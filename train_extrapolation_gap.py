# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kgcnn.data.utils import save_pickle_file, load_pickle_file

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
from kgcnn.training.hyper import HyperParameter
from kgcnn.data.serial import deserialize as deserialize_dataset
from kgcnn.model.serial import deserialize as deserialize_model
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from kgcnn.utils.devices import set_devices_gpu


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

# 初始化配置类并设置参数
config = GNNTrainConfig(hyper="training/hyper/multifidelity_band_gap.py", model='coGN', make='make_model', dataset='MatProjectMultifidelityDataset', seed=42)
args = config.to_dict()
np.random.seed(args["seed"])
tf.random.set_seed(args["seed"])
tf.keras.utils.set_random_seed(args["seed"])
set_devices_gpu(args["gpu"])

hyper = HyperParameter(
    hyper_info=args["hyper"], hyper_category=args["category"],
    model_name=args["model"], model_class=args["make"], dataset_class=args["dataset"])
hyper.verify()

# 加载数据集

# dataset = deserialize_dataset(hyper["dataset"])
# filepath = hyper.results_file_path()
# save_pickle_file(dataset, os.path.join(filepath, "Schnet_data.pickle"))

ALL_FIDELITIES = [ "pbe", "gllb-sc", "hse","scan"]
TEST_FIDELITIES = ["pbe", "gllb-sc", "hse","scan"]

datapath = r"/home/deep/gcnn_keras-master/results/MatProjectMultifidelityDataset/coGN_make_model-pbe/coGN_data.pickle"
dataset = load_pickle_file(datapath)

# 检查数据集是否满足模型输入要求
dataset.assert_valid_model_input(hyper["model"]["config"]["inputs"])

# 过滤无效图形
dataset.clean(hyper["model"]["config"]["inputs"])

# Train on graph labels. Must be defined by the dataset.
label_names = dataset.label_names
label_units = dataset.label_units


# Make output directory
filepath = hyper.results_file_path()
postfix_file = hyper["info"]["postfix_file"]
data_length = len(dataset)  # Length of the cleaned dataset.

# Training on multiple targets for regression.
multi_target_indices = hyper["training"]["multi_target_indices"] if "multi_target_indices" in hyper[
    "training"] else None
if multi_target_indices is not None:
    # labels = labels[:, multi_target_indices]
    if label_names is not None:
        label_names = [label_names[i] for i in multi_target_indices]
    if label_units is not None:
        label_units = [label_units[i] for i in multi_target_indices]

pbe_indices = np.array([i for i, fid in enumerate([str(id).split("_")[1] for id in dataset.obtain_property('materials_ids')]) 
                        if fid in ALL_FIDELITIES])
dataset_filter = dataset[pbe_indices]


# 获取原子编号
atoms = dataset_filter.obtain_property("node_number")
# 获取标签
labels = np.array(dataset_filter.obtain_property("graph_labels"))
if len(labels.shape) <= 1:
    labels = np.expand_dims(labels, axis=-1)
print("Labels '%s' in '%s' have shape '%s'." % (label_names, label_units, labels.shape))


# 根据目标值范围分割数据集
threshold = np.percentile(labels, 90)
# 根据阈值分割数据
high_labels_indices = np.where(labels >= threshold)[0]
low_labels_indices = np.where(labels < threshold)[0]

test_size_high = 0.1  # 10% of high_labels_indices
test_indices, _ = train_test_split(high_labels_indices, test_size=test_size_high, random_state=args["seed"])


# 从 low_labels_indices 中采样 val_indices
val_size_low = 0.1  # 10% of low_labels_indices
remaining_low_indices, val_indices = train_test_split(low_labels_indices, test_size=val_size_low, random_state=args["seed"])

# 从 remaining_low_indices 中进一步采样 train_indices 和 train_validation_indices
train_val_size_low = 0.1  # 10% of remaining_low_indices
train_indices, train_validation_indices = train_test_split(remaining_low_indices, test_size=train_val_size_low, random_state=args["seed"])

# 检查分割结果
total_samples = len(np.concatenate((low_labels_indices, high_labels_indices)))

print(f"threshold value of datasets: {threshold}")
print(f"Total number of samples: {total_samples}")
print(f"Number of train samples: {len(train_indices)} ({len(train_indices) / total_samples:.2f})")
print(f"Number of train validation samples: {len(train_validation_indices)} ({len(train_validation_indices) / total_samples:.2f})")
print(f"Number of validation samples: {len(val_indices)} ({len(val_indices) / total_samples:.2f})")
print(f"Number of test samples: {len(test_indices)} ({len(test_indices) / total_samples:.2f})")


# 准备训练、验证和测试数据
x_train, y_train = dataset_filter[train_indices].tensor(hyper["model"]["config"]["inputs"]), labels[train_indices]
x_train_val, y_train_val = dataset_filter[train_validation_indices].tensor(hyper["model"]["config"]["inputs"]), labels[train_validation_indices]

x_val, y_val = dataset_filter[val_indices].tensor(hyper["model"]["config"]["inputs"]), labels[val_indices]
x_test, y_test = dataset_filter[test_indices].tensor(hyper["model"]["config"]["inputs"]), labels[test_indices]

atoms_train = [atoms[i] for i in train_indices]
atoms_train_val = [atoms[i] for i in train_validation_indices]
atoms_test = [atoms[i] for i in test_indices]
atoms_val = [atoms[i] for i in val_indices]

# 数据标准化
scaler = StandardLabelScaler(**hyper["training"]["scaler"]["config"])
y_train = scaler.fit_transform(y=y_train, atomic_number=atoms_train)
y_train_val = scaler.fit_transform(y=y_train_val, atomic_number=atoms_train_val)
y_val = scaler.transform(y=y_val, atomic_number=atoms_val)
y_test = scaler.transform(y=y_test, atomic_number=atoms_test)

# 构建模型并训练
model = deserialize_model(hyper["model"])

model.compile(**hyper.compile(loss="mean_absolute_error", metrics=['mae']))
print(model.summary())


# load from the best model
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
        filepath="best_model.hdf5", 
        monitor="val_loss",  # 或者根据实际情况选择其他监控指标，如 "val_mean_absolute_error"
        mode="min",  # 对于损失函数，我们通常希望其值越小越好
        save_best_only=True,  # 只保存最优模型
    )
fit_args = hyper.fit()
if 'callbacks' in fit_args:
        del fit_args['callbacks']
hist = model.fit(x_train, y_train,
                    validation_data=(x_train_val, y_train_val),
                    callbacks=[ checkpoint_callback],  # Add the custom callback here
                    # **hyper.fit()
                    **fit_args
                    )


# Load the best model after training
model.load_weights("best_model.hdf5")

# 预测并反标准化
predicted_val = scaler.inverse_transform(model.predict(x_val), atomic_number=atoms_val)
predicted_test = scaler.inverse_transform(model.predict(x_test), atomic_number=atoms_test)
true_val = scaler.inverse_transform(y_val, atomic_number=atoms_val)
true_test = scaler.inverse_transform(y_test, atomic_number=atoms_test)

# 计算绝对误差
val_abs_error = np.abs(predicted_val - true_val)
test_abs_error = np.abs(predicted_test - true_test)

# 计算 MAE
mae_interpolation = np.mean(val_abs_error)
mae_extrapolation = np.mean(test_abs_error)


import matplotlib.pyplot as plt
# 绘制预测结果绝对误差图
plt.figure(figsize=(5, 4))

# 合并 Interpolation 和 Extrapolation 图
plt.scatter(true_val, predicted_val, label='Interpolation', color='lightseagreen')
plt.scatter(true_test, predicted_test, label='Extrapolation', color='orangered')

# 设置共享的 x 轴和 y 轴标签
plt.xlabel('MatProjectMultifidelity DFT',  fontsize=14)
plt.ylabel('MatProjectMultifidelity Predicted',  fontsize=14)

# 设置标题
plt.title('CHGNet MAE', fontsize=14)

# 添加图例
plt.legend(fontsize=12)

# 在图例中添加 MAE 值
plt.text(0.05, 0.95, f'Interpolation: {mae_interpolation:.3f}\nExtrapolation: {mae_extrapolation:.3f}',
         transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

# 设置 y 轴的显示范围
y_min = min(min(predicted_val), min(predicted_test))
y_max = max(max(predicted_val), max(predicted_test))
plt.ylim(y_min -  abs(y_min), y_max + 0.5* abs(y_max))


# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig(os.path.join(filepath, f"abs_error_plots_CHGNet.png"))


# 保存每个材料的ID及对应的预测结果和标签值
results = {
    # 'material_ids': [dataset.material_ids[i] for i in test_indices],
    'interpolation_predictions': predicted_val.flatten(),
    'extrapolation_predictions': predicted_test.flatten(),
    'true_val': true_val.flatten(),
    'true_values': true_test.flatten(),

}

# 保存 MAE 值
results['mae_interpolation'] = mae_interpolation
results['mae_extrapolation'] = mae_extrapolation

save_pickle_file(results, os.path.join(filepath, f"prediction_results{postfix_file}.pickle"))




