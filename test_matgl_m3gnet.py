
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
import matplotlib
from functools import partial
import os
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
import numpy as np
from pymatgen.core import Structure
import matgl
from dgl.data.utils import Subset
import lightning as pl
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
from matgl.models import M3GNet
from matgl.utils.training import ModelLightningModule

matplotlib.use('Agg')


def plot_predict_true(y_predict, y_true, data_unit: str = "", model_name: str = "",
                      filepath: str = "", file_name: str = "", dataset_name: str = "", target_names: str = "",
                      figsize: list = None, dpi: float = None, show_fig: bool = False):
    r"""Make a scatter plot of predicted versus actual targets. Not for k-splits.

    Args:
        y_predict (np.ndarray): Numpy array of shape `(N_samples, n_targets)` or `(N_samples, )`.
        y_true (np.ndarray): Numpy array of shape `(N_samples, n_targets)` or `(N_samples, )`.
        data_unit (str): Name of the data's unit.
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        target_names (str): Name of the targets.
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.

    Returns:
        matplotlib.pyplot.figure: Figure of the scatter plot.
    """
    if len(y_predict.shape) == 1:
        y_predict = np.expand_dims(y_predict, axis=-1)
    if len(y_true.shape) == 1:
        y_true = np.expand_dims(y_true, axis=-1)
    num_targets = y_true.shape[1]

    if data_unit is None:
        data_unit = ""
    if isinstance(data_unit, str):
        data_unit = [data_unit] * num_targets
    if len(data_unit) != num_targets:
        print("WARNING:kgcnn: Targets do not match units for plot.")
    if target_names is None:
        target_names = ""
    if isinstance(target_names, str):
        target_names = [target_names] * num_targets
    if len(target_names) != num_targets:
        print("WARNING:kgcnn: Targets do not match names for plot.")

    if figsize is None:
        # figsize = [6, 5]
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 300.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for i in range(num_targets):
        delta_valid = y_true[:, i] - y_predict[:, i]
        delta_valid_value = delta_valid[~np.isnan(delta_valid)]

        mae_valid = np.mean(np.abs(delta_valid_value))
        rmse_valid = np.sqrt(np.mean(delta_valid_value ** 2))
        label = f"{target_names[i]} MAE: {mae_valid:.4f}, RMSE: {rmse_valid:.4f} [{data_unit[i]}]"

        plt.scatter(y_predict[:, i], y_true[:, i], label=label)

    min_max = np.amin(y_true[~np.isnan(y_true)]).astype("float"), np.amax(y_true[~np.isnan(y_true)]).astype("float")
    plt.plot(np.arange(*min_max, 0.05), np.arange(*min_max, 0.05), color='red')
    plt.xlabel('Predicted')
    plt.ylabel('DFT')
    plt.title("Prediction of " + model_name + " for " + dataset_name)
    plt.legend(loc='upper left', fontsize='medium')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name))
    if show_fig:
        plt.show()
    return fig


# 加载数据
with open("/home/datasets/sample_all_materials_data_form_processed/sample_all_materials_data_form_processed.json", "r") as f:
    exp_band_gap_data = json.load(f)

# 提取材料ID和数据
material_ids = exp_band_gap_data["index"]
data = exp_band_gap_data["data"]

# 初始化预测值和真实值列表
predictions = []
true_labels = []

# 遍历数据，提取结构信息并进行预测
X = []
y = []
for entry in data:
    struct_info = entry[0]
    exp_band_gap = entry[1]
    
    # 将结构信息转换为 pymatgen.Structure 对象
    struct = Structure.from_dict(struct_info)
    
    # 检查结构是否为有序结构
    if struct.is_ordered:
        X.append(struct)
        y.append(exp_band_gap)


# 转换为 numpy 数组
y = np.array(y)

# 设置交叉验证
kf = KFold(n_splits=5, random_state=42, shuffle=True)
train_test_indices = [(train_index, test_index) for train_index, test_index in kf.split(X, y)]

# 初始化存储每个 fold 的预测值和真实值
all_predictions = []
all_true_labels = []

# 读取 prediction_results.pickle 文件
# path = r"/home/deep/gcnn_keras-master/results/MPShearModulusVrhDataset/150/coGN_make_model/prediction_results_save.pickle"
path = r"/home/deep/gcnn_keras-master/results/SampleAllMPFormDataset/coGN_make_model/prediction_results_save.pickle"
with open(path, 'rb') as file:
    results = pickle.load(file)


# 提取数据
label_units = results['label_units']
dataset_name = results['dataset_name']
target_names = results['target_names']

# 对每个 fold 进行训练和测试
for current_fold, (train_index, test_index) in enumerate(train_test_indices):
    train_structure, val_structure = [X[i] for i in train_index], [X[i] for i in test_index]
    train_labels, val_labels = y[train_index], y[test_index]
    
    # 使用模型进行预测
    all_structures = train_structure + val_structure 
    # 获取元素类型
    elem_list = get_element_list(all_structures)

    # 设置图转换器
    converter = Structure2Graph(element_types=elem_list, cutoff=5.0)

    # 转换原始数据集为M3GNetDataset
    train_mp_dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=train_structure,
        converter=converter,
        labels={"gvrh": train_labels},
        include_line_graph=True,
    )

    val_mp_dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=val_structure,
        converter=converter,
        labels={"gvrh": val_labels},
        include_line_graph=True,
    )
    

    # 创建子集
    train_data = Subset(train_mp_dataset, np.arange(len(train_mp_dataset)))
    val_data = Subset(val_mp_dataset, np.arange(len(val_mp_dataset)))

    # 定义数据加载器
    my_collate_fn = partial(collate_fn_graph, include_line_graph=True)
    train_loader, val_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=None,
        collate_fn=my_collate_fn,
        batch_size=256,
        num_workers=0,
    )

    # 设置 M3GNet 模型
    model = M3GNet(
        element_types=elem_list,
        is_intensive=True,
        readout_type="set2set",
    )

    # 设置 PyTorch Lightning 模块
    lit_module = ModelLightningModule(model=model, include_line_graph=True, loss="mae_loss")

    # 设置日志记录器和训练器,M3GNET
    logger = CSVLogger("logs", name="M3GNet_training")
    # trainer = pl.Trainer(max_epochs=300, accelerator="cpu", logger=logger)
    trainer = pl.Trainer(
        max_epochs=140,
        accelerator="gpu",  # 使用GPU
        devices=1,  # 使用1个GPU
        logger=logger
    )
    # 开始训练
    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


    model_save_path = f"./M3GNet-MP-Gvrh-{current_fold}/"
    lit_module.model.save(model_save_path)

    # 加载模型
    model = matgl.load_model(model_save_path)
    # model = matgl.load_model("./pretrained_models/CHGNet-MP-Gvrh-mean-1")


    # predict
    fold_predictions = []
    fold_true_labels = []
    for struct, exp_band_gap in zip(val_structure, val_labels):
        gvrh = model.predict_structure(struct)
        fold_predictions.append(float(gvrh))
        fold_true_labels.append(exp_band_gap)
    
    # 转换为 numpy 数组
    fold_predictions = np.array(fold_predictions)
    fold_true_labels = np.array(fold_true_labels)
    
    # 过滤掉包含 NaN 的数据
    valid_indices = ~np.isnan(fold_predictions) & ~np.isnan(fold_true_labels)
    fold_predictions = fold_predictions[valid_indices]
    fold_true_labels = fold_true_labels[valid_indices]
    
    # 存储每个 fold 的预测值和真实值
    all_predictions.extend(fold_predictions)
    all_true_labels.extend(fold_true_labels)
    
    # 计算当前 fold 的 MAE 和 RMSE
    mae = np.mean(np.abs(fold_predictions - fold_true_labels))
    rmse = np.sqrt(mean_squared_error(fold_true_labels, fold_predictions))
    
    print(f"Fold {current_fold + 1} Test MAE: {mae:.4f}")
    print(f"Fold {current_fold + 1} Test RMSE: {rmse:.4f}")
    
    # 绘制当前 fold 的预测结果图并保存
    postfix_file = ""  # 你可以根据需要设置这个后缀
    file_name = f"predict{postfix_file}_fold_{current_fold}.png"
    plot_predict_true(fold_predictions, fold_true_labels, data_unit=label_units, model_name='M3GNet', 
                      filepath="./", file_name=file_name, dataset_name=dataset_name, 
                      target_names=target_names, show_fig=True)


