import pathlib
import joblib
import numpy as np
import pandas as pd
import tensorflow
import logging
import create_topology as top
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import random


random.seed()
rand_num = int(100 * random.random())
print("seed=", rand_num)

# GPU usage
# tensorflow.get_logger().setLevel(logging.INFO)
# physical_devices = tensorflow.config.list_physical_devices("GPU")
# tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

# Access path where dataset is stored
PATH = pathlib.Path(pathlib.Path(__file__).parent.resolve()).parent.resolve()
directory = str(PATH) + r'\data\python_data'
csv_file = directory + r'\HyperOptResults_DistanceID'
best_models_loc = csv_file + r'\best_distance_id_v04.csv'
model_class_loc = csv_file + r'\model_class_v04_'


def shuffle_dataset(dataset, output_class):
    x_train, x_test, y_train, y_test = train_test_split(dataset, output_class, test_size=0.2, random_state=rand_num)
    return x_train, x_test, y_train, y_test


def reshape_2d(dataset, window_size):
    phases = 3
    metrics = int(dataset[0].shape[1]/phases)
    num_windows = int(dataset[0].shape[0] / window_size)
    new_dataset = np.empty([len(dataset), window_size, metrics, phases, num_windows], dtype=float)

    for i in range(len(dataset)):
        for j in range(window_size):
            for k in range(metrics):
                for m in range(phases):
                    for n in range(num_windows):
                        new_dataset[i, j, k, m, n] = dataset[i][window_size * n + j, phases * k + m]
    del dataset
    return new_dataset


def rmse(y_true, y_pred):
    mse = np.mean(np.square(y_true - y_pred))
    rmse_out = np.sqrt(mse)

    return rmse_out


def reshape_class(y_hat):
    re_y = [0 for _ in range(len(y_hat))]
    segments = len(y_hat[0])
    for i in range(len(y_hat)):
        re_y[i] = float(np.argmax(y_hat[i]))/(segments-1)

    return np.array(re_y)


def get_metrics(dataset, output, model, model_df):

    metrics = [[] for _ in range(3)]
    for i in range(3):
        train_x, test_x, train_y, test_y = shuffle_dataset(dataset, output)
        train_y, test_y = reshape_class(train_y), reshape_class(test_y)
        loaded_model = tensorflow.keras.models.load_model(model + str(i+1))
        model_params = model_df.iloc[i]
        window = int(model_params['window'])
        test_x = reshape_2d(test_x, window)


        yp = [x for [x] in loaded_model.predict(test_x)]
        yp = np.array(yp)
        me = np.square(test_y - yp)
        rme = np.sqrt(me)
        model_rmse = rmse(test_y, yp)
        
        metrics[i] = [me, rme, model_rmse]

    return metrics


tree, nodes, grid_length = top.create_grid()
leaf_nodes = top.give_leaves(nodes[0])

# Find all leaf nodes
grid_path = []
branch_distance = []
for leaf in leaf_nodes:
    arr, dist = top.give_path(nodes[0], leaf)
    grid_path.append(arr)
    branch_distance.append(dist)
feeders_num = len(nodes[0].children)  # Number of root's children: 3
branches_num = len(grid_path)         # Number of grid's branches: 9
metrics_num = len(nodes)              # Number of voltage meters: 33
distance_sections = 5                 # Number of sections a branch is divided: 5


class_path = directory + r'\Distance_Class_sorted.joblib'
V_branch_noDC_path = directory + r'\V_branch_noDC.joblib'
V_branch_noDC = joblib.load(V_branch_noDC_path)
Distance_Class_sorted = np.array(joblib.load(class_path))
best_models = pd.read_csv(best_models_loc, index_col=0)


Metrics = get_metrics(V_branch_noDC, Distance_Class_sorted, model_class_loc, best_models)


joblib.dump(Metrics, csv_file + r'\Metrics_v04.joblib')
