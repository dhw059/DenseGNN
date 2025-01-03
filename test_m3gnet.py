
import tensorflow as tf
from m3gnet.models import M3GNet
from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester
import json
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from kgcnn.data.utils import save_pickle_file, load_pickle_file
import pickle
import matplotlib
import warnings

# 忽略 TensorFlow 的警告
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

# 关闭 TensorFlow 的自动追踪优化警告
tf.config.experimental.enable_tensor_float_32_execution(False)
tf.config.optimizer.set_jit(True)
"""
# 忽略警告
warnings.filterwarnings("ignore")
"""

# 修改 TensorFlow 的日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        figsize = [6, 5]
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



# 加载训练好的模型
model_path = r"/home/deep/m3gnet-main/pretrained/m3gnet_models/matbench_log_gvrh/3/m3gnet"
model = M3GNet.load(model_path)



# 加载数据
with open("/home/datasets/all_materials_gvrh_data_processed/all_materials_gvrh_data_processed.json", "r") as f:
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
path = r"/home/deep/gcnn_keras-master/results/MPShearModulusVrhDataset/150/coGN_make_model/prediction_results_save.pickle"
with open(path, 'rb') as file:
    results = pickle.load(file)


# 提取数据
label_units = results['label_units']
dataset_name = results['dataset_name']
target_names = results['target_names']

# 对每个 fold 进行训练和测试
for current_fold, (train_index, test_index) in enumerate(train_test_indices):
    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 使用模型进行预测

    fold_predictions = []
    fold_true_labels = []
    for struct, exp_band_gap in zip(X_test, y_test):
        log_prediction = model.predict_structure(struct)
        gvrh = 10 ** log_prediction
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


# # 转换为 numpy 数组
# all_predictions = np.array(all_predictions)
# all_true_labels = np.array(all_true_labels)

# # 检查数据维度
# print(f"Overall Predictions shape: {all_predictions.shape}")
# print(f"Overall True labels shape: {all_true_labels.shape}")

# # 计算总体的 MAE 和 RMSE
# overall_mae = np.mean(np.abs(all_predictions - all_true_labels))
# overall_rmse = np.sqrt(mean_squared_error(all_true_labels, all_predictions))

# print(f"Overall Test MAE: {overall_mae:.4f}")
# print(f"Overall Test RMSE: {overall_rmse:.4f}")

# # 绘制总体的预测结果图并保存
# file_name = "predict_overall.png"
# plot_predict_true(all_predictions, all_true_labels, data_unit='GPa', model_name='M3GNet', 
#                   filepath="./", file_name=file_name, dataset_name='ContainLiCompoundsDataset', 
#                   target_names='Shear modulus vrh', show_fig=True)
