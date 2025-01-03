import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import time
import os
import tensorflow as tf
from kgcnn.data.utils import save_pickle_file, load_pickle_file
from datetime import timedelta
from tensorflow_addons import optimizers
from kgcnn.data.transform.scaler.standard import StandardLabelScaler
from kgcnn.data.transform.scaler.molecule import QMGraphLabelScaler
import kgcnn.training.scheduler
from kgcnn.training.history import save_history_score, load_history_list
from kgcnn.metrics.metrics import ScaledMeanAbsoluteError, ScaledRootMeanSquaredError
from sklearn.model_selection import KFold
from kgcnn.training.hyper import HyperParameter
from kgcnn.data.serial import deserialize as deserialize_dataset
from kgcnn.model.serial import deserialize as deserialize_model
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from kgcnn.utils.devices import set_devices_gpu
from sklearn.model_selection import StratifiedKFold,train_test_split

class GNNTrainConfig:
    def __init__(self, 
                 hyper="hyper/hyper_mp_e_form.py", 
                 category= None, 
                 model=None, 
                 dataset=None, 
                 make= None, 
                 gpu=None,
                 fold=None,
                 seed=42):
        self.hyper = hyper
        self.category = category
        self.model = model
        self.dataset = dataset
        self.make = make
        self.gpu = gpu 
        self.seed = seed
        self.fold = fold

    def to_dict(self):
        return vars(self)

# Usage:
config = GNNTrainConfig(hyper="training/hyper/hyper_sample_all_materials_data_form_processed.py", model='Megnet', 
                        make='make_crystal_model', dataset='SampleAllMPFormDataset', seed=42)
print("Input of argparse:", config.to_dict())
args = config.to_dict()


# Set seed.
np.random.seed(args["seed"])
tf.random.set_seed(args["seed"])
tf.keras.utils.set_random_seed(args["seed"])


# Assigning GPU.
set_devices_gpu(args["gpu"])

# A class `HyperParameter` is used to expose and verify hyperparameter.
# The hyperparameter is stored as a dictionary with sectiomegnet 'model', 'dataset' and 'training'.
hyper = HyperParameter(
    hyper_info=args["hyper"], hyper_category=args["category"],
    model_name=args["model"], model_class=args["make"], dataset_class=args["dataset"])
hyper.verify()


dataset = deserialize_dataset(hyper["dataset"])
filepath = hyper.results_file_path()
save_pickle_file(dataset, os.path.join(filepath, "Megnet_data.pickle"))

# datapath = "/home/deep/gcnn_keras-master/results/MatProjectMultifidelityDataset/Schnet_make_crystal_model-pbe/Schnet_data.pickle"
# dataset = load_pickle_file(datapath)



data_length = len(dataset)  # Length of the cleaned dataset.
print("--------------------------------------------------------------------------")
print(data_length)


ALL_FIDELITIES = [ "pbe", "gllb-sc", "hse","scan"]
TEST_FIDELITIES = ["pbe", "gllb-sc", "hse","scan"]

# Check if dataset has the required properties for model input. This includes a quick shape comparison.
# The name of the keras `Input` layer of the model is directly connected to property of the dataset.
# Example 'edge_indices' or 'node_attributes'. This couples the keras model to the dataset.
dataset.assert_valid_model_input(hyper["model"]["config"]["inputs"])

# Filter the dataset for invalid graphs. At the moment invalid graphs are graphs which do not have the property set,
# which is required by the model's input layers, or if a tensor-like property has zero length.
dataset.clean(hyper["model"]["config"]["inputs"])

# Train on graph labels. Must be defined by the dataset.
label_names = dataset.label_names
label_units = dataset.label_units


# Training on multiple targets for regression.
multi_target_indices = hyper["training"]["multi_target_indices"] if "multi_target_indices" in hyper[
    "training"] else None
if multi_target_indices is not None:
    # labels = labels[:, multi_target_indices]
    if label_names is not None:
        label_names = [label_names[i] for i in multi_target_indices]
    if label_units is not None:
        label_units = [label_units[i] for i in multi_target_indices]

# Make output directory
filepath = hyper.results_file_path()
postfix_file = hyper["info"]["postfix_file"]

# Training on splits. Since training on crystal datasets can be expensive, there is a 'execute_splits' parameter to not
# train on all splits for testing.
execute_folds = args["fold"]
# if "execute_folds" in hyper["training"]:
#     execute_folds = hyper["training"]["execute_folds"]
model, hist, x_test, y_test, scaler, atoms_test = None, None, None, None, None, None


pbe_indices = np.array([i for i, fid in enumerate([str(id).split("_")[1] for id in dataset.obtain_property('materials_ids')]) 
                        if fid in ALL_FIDELITIES])
dataset_filter = dataset[pbe_indices]


# For Crystals, also the atomic number is required to properly pre-scale extensive quantities like total energy.
atoms = dataset_filter.obtain_property("node_number")
material_ids = [str(i['materials_ids']) for i in dataset_filter]   
fidelity_list = [i.split("_")[1] for i in material_ids]  

# 初始化 StratifiedKFold 分类器
kfold = StratifiedKFold(**hyper["training"]["cross_validation"]["config"])
# 遍历各个折叠
train_test_indices = [
    (train_index, test_index) for train_index, test_index in kfold.split(material_ids, fidelity_list )]


num_folds = len(train_test_indices)
splits_done = 0
time_list = []
train_indices_all, test_indices_all = [], []

for current_fold, (train_index, test_index) in enumerate(train_test_indices):
    test_indices_all.append(test_index)
    train_indices_all.append(train_index)

    test_dataset = dataset_filter[test_index]
    material_ids = [str(i['materials_ids']) for i in test_dataset] 
    fidelity_list_test = [i.split("_")[1] for i in material_ids] 
    test_index, val_index = train_test_split(test_index, stratify=fidelity_list_test, test_size=0.5, random_state=42)  

   #  remove pbe from validation
    # val_index = [i for i in val_index if not str(list(dataset_filter)[i]['materials_ids']).endswith("pbe")]

    print("Train, val and test data sizes are ", len(train_index), len(val_index), len(test_index))

    # Only do execute_splits out of the k-folds of cross-validation.
    if execute_folds:
        if current_fold not in execute_folds:
            continue
    print("Running training on fold: %s" % current_fold)

    # Make the model for current split using model kwargs from hyperparameter.
    # They are always updated on top of the models default kwargs.
    model = deserialize_model(hyper["model"])
 

    # First select training and test graphs from indices, then convert them into tensorflow tensor
    # representation. Which property of the dataset and whether the tensor will be ragged is retrieved from the
    # kwargs of the keras `Input` layers ('name' and 'ragged').
    x_train, y_train = dataset_filter[train_index].tensor(hyper["model"]["config"]["inputs"]),  np.expand_dims( np.array(dataset_filter[train_index].get("graph_labels")), axis=-1)
    x_val, y_val = dataset_filter[val_index].tensor(hyper["model"]["config"]["inputs"]),   np.expand_dims( np.array(dataset_filter[val_index].get("graph_labels")) , axis=-1)
    # Also keep the same information for atomic numbers of the structures.
    atoms_val = [atoms[i] for i in val_index]
    atoms_train = [atoms[i] for i in train_index]


    # Normalize training and test targets via a sklearn `StandardScaler`. No other scaler are used at the moment.
    # Scaler is applied to target if 'scaler' appears in hyperparameter. Only use for regression.
    if "scaler" in hyper["training"]:
        print("Using StandardScaler.")
        if hyper["training"]["scaler"]["class_name"] == "QMGraphLabelScaler":
            scaler = QMGraphLabelScaler(**hyper["training"]["scaler"]["config"])
        else:
            scaler = StandardLabelScaler(**hyper["training"]["scaler"]["config"])

        y_train = scaler.fit_transform(y=y_train, atomic_number=atoms_train)
        y_val = scaler.transform(y=y_val, atomic_number=atoms_val)
        scaler_scale = scaler.get_scaling()

        # If scaler was used we add rescaled standard metrics to compile, since otherwise the keras history will not
        # directly log the original target values, but the scaled ones.
        mae_metric = ScaledMeanAbsoluteError(scaler_scale.shape, name="scaled_mean_absolute_error")
        rms_metric = ScaledRootMeanSquaredError(scaler_scale.shape, name="scaled_root_mean_squared_error")
        if scaler_scale is not None:
            mae_metric.set_scale(scaler_scale)
            rms_metric.set_scale(scaler_scale)
        metrics = [mae_metric, rms_metric]

        # Save scaler to file
        scaler.save(os.path.join(filepath, f"scaler{postfix_file}_fold_{current_fold}"))

    else:
        print("TRAINING: Not using StandardScaler for regression.")
        metrics = None

    # Compile model with optimizer and loss
    model.compile(**hyper.compile(loss="mean_absolute_error", metrics=metrics))
    print(model.summary())

    import time
    # Define a custom LambdaCallback to format the output
    custom_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=lambda epoch, logs: setattr(custom_callback, 'epoch_start_time', time.time()),
        on_epoch_end=lambda epoch, logs: print(
            f"Epoch {epoch + 1}/{hyper.fit()['epochs']}, "
            f"loss: {logs['loss']:.6f} - scaled_mean_absolute_error: {logs['scaled_mean_absolute_error']:.6f} "
            f"- scaled_root_mean_squared_error: {logs['scaled_root_mean_squared_error']:.6f} "
            f"- {(time.time() - custom_callback.epoch_start_time):.3f}s/epoch"
        )
    )

    # Define the ModelCheckpoint callback to save the best model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model.hdf5",
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    fit_args = hyper.fit()
    if 'callbacks' in fit_args:
        del fit_args['callbacks']
        
    # Start and time training
    start = time.time()
    hist = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    callbacks=[custom_callback, model_checkpoint_callback],   # Add the custom callback here
                    # **hyper.fit()
                    **fit_args
                    )
    stop = time.time()
    
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))
    time_list.append(str(timedelta(seconds=stop - start)))
    # Get loss from history
    save_pickle_file(hist.history, os.path.join(filepath, f"history{postfix_file}_fold_{current_fold}.pickle"))

    # Plot prediction
    x_test, y_test = dataset_filter[test_index].tensor(hyper["model"]["config"]["inputs"]), np.expand_dims( np.array(dataset_filter[test_index].get("graph_labels")) , axis=-1)
    atoms_test = [atoms[i] for i in test_index]
    y_test = scaler.transform(y=y_test, atomic_number=atoms_test)
    predicted_y = model.predict(x_test)  
    true_y = y_test

    if scaler:
        predicted_y = scaler.inverse_transform(y=predicted_y, atomic_number=atoms_test)
        true_y = scaler.inverse_transform(y=true_y, atomic_number=atoms_test)

    
    plot_predict_true(predicted_y, true_y,
                      filepath=filepath, data_unit=label_units,
                      model_name='CHGNet', dataset_name=hyper.dataset_class, target_names=label_names,
                      file_name=f"predict{postfix_file}_fold_{current_fold}.png", show_fig=False)
    # hyper.model_name
    # 保存每个材料的ID及对应的预测结果和标签值
    results = {
        # 'material_ids': [dataset.material_ids[i] for i in test_indices],
        'predicted_y': predicted_y,
        'true_y': true_y,
        'label_units':label_units,
        'model_name':hyper.model_name,
        'dataset_name':hyper.dataset_class,
        'target_names':label_names,
        'file_name':f"predict{postfix_file}_fold_{current_fold}.png"
    }

    save_pickle_file(results, os.path.join(filepath, f"prediction_results_save{postfix_file}.pickle"))


    # Save keras-model to output-folder.
    # model.save(os.path.join(filepath, f"model{postfix_file}_fold_{current_fold}"))

    # Save complete model (architecture + weights) with a different file extension
    model.save(os.path.join(filepath, f"model{postfix_file}_fold_{current_fold}.h5"))

    model.save(os.path.join(filepath, f"model{postfix_file}_fold_{current_fold}.keras"))

    # Save weights with a specific suffix for clarity
    model.save_weights(os.path.join(filepath, f"model{postfix_file}_fold_{current_fold}_weights_only.h5"))

    splits_done = splits_done + 1

history_list = load_history_list(os.path.join(filepath, f"history{postfix_file}_fold_(i).pickle"), num_folds)

# Plot training- and test-loss vs epochs for all splits.
data_unit = hyper["data"]["data_unit"] if "data_unit" in hyper["data"] else ""
plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                     model_name=hyper.model_name, data_unit=data_unit, dataset_name=hyper.dataset_class,
                     filepath=filepath, file_name=f"loss{postfix_file}.png")

# Save original data indices of the splits.
np.savez(os.path.join(filepath, f"{hyper.model_name}_test_indices_{postfix_file}.npz"), *test_indices_all)
np.savez(os.path.join(filepath, f"{hyper.model_name}_train_indices_{postfix_file}.npz"), *train_indices_all)

# Save hyperparameter again, which were used for this fit.
hyper.save(os.path.join(filepath, f"{hyper.model_name}_hyper{postfix_file}.json"))

# Save score of fit result for as text file.
save_history_score(history_list, loss_name=None, val_loss_name=None,
                   model_name=hyper.model_name, data_unit=data_unit, dataset_name=hyper.dataset_class,
                   model_class=hyper.model_class, multi_target_indices=multi_target_indices,
                   execute_folds=execute_folds,seed=args["seed"],
                   filepath=filepath, file_name=f"score{postfix_file}.yaml", time_list=time_list)




