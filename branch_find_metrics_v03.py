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

# # GPU usage
# tensorflow.get_logger().setLevel(logging.INFO)
# physical_devices = tensorflow.config.list_physical_devices("GPU")
# tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

# Access path where dataset is stored
PATH = pathlib.Path(pathlib.Path(__file__).parent.resolve()).parent.resolve()
directory = str(PATH) + r'\data\python_data'
csv_file = directory + r'\HyperOptResults_BranchID'
best_models_f1 = csv_file + r'\best_branch_id_v03_1.csv'
best_models_f2 = csv_file + r'\best_branch_id_v03_2.csv'
best_models_f3 = csv_file + r'\best_branch_id_v03_3.csv'
model_f1_loc = csv_file + r'\model_v03_f1_'
model_f2_loc = csv_file + r'\model_v03_f2_'
model_f3_loc = csv_file + r'\model_v03_f3_'


def shuffle_dataset(dataset, output_class):
    x_train, x_test, y_train, y_test = train_test_split(dataset, output_class, test_size=0.2, random_state=rand_num)
    return x_train, x_test, y_train, y_test


def reshape_2d(dataset, window_size):
    num_windows = int(dataset[0].shape[1] / window_size)
    new_dataset = np.empty([len(dataset), window_size, dataset[0].shape[0],
                            dataset[0].shape[2], num_windows], dtype=float)
    for i in range(len(dataset)):
        for j in range(window_size):
            for k in range(dataset[0].shape[0]):
                for m in range(num_windows):
                    new_dataset[i, j, k, ::, m] = dataset[i][k, window_size*m+j, ::]
    del dataset
    return new_dataset


def get_metrics(dataset, output, model, model_df):

    matrix, metrics = [[] for _ in range(3)], [[] for _ in range(3)]
    multilabel_conf_matrix = [[] for _ in range(3)]
    for i in range(3):
        train_x, test_x, train_y, test_y = shuffle_dataset(dataset, output)
        loaded_model = tensorflow.keras.models.load_model(model + str(i+1))
        model_params = model_df.iloc[i]
        window = int(model_params['window'])
        test_x = reshape_2d(test_x, window)

        # predict crisp classes for test set
        yhat_classes = np.argmax(loaded_model.predict(test_x, verbose=0), axis=-1)
        yhat_classes = to_categorical(yhat_classes)

        # confusion matrix
        multilabel_conf_matrix[i] = confusion_matrix(test_y, yhat_classes)
        matrix[i] = multilabel_confusion_matrix(test_y, yhat_classes)
        print(matrix[i])

        rep_i = classification_report(test_y, yhat_classes, output_dict=True)
        df_i = pd.DataFrame(rep_i).transpose()[:test_y.shape[1]]
        print(df_i)
        metrics[i] = df_i

    return matrix, multilabel_conf_matrix, metrics


def confusion_matrix(actual_class, predicted_class):
    
    idx_true = np.argmax(actual_class, axis=-1)
    idx_predicted = np.argmax(predicted_class, axis=-1)
    dim = max(idx_true)+1
    matrix = np.zeros((dim, dim), dtype=int)
    for i in range(len(idx_true)):
        matrix[idx_true[i], idx_predicted[i]] += 1
    
    return matrix


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

V_feeder_noDC, Branch_Output_sorted, best_models = [[] for _ in range(feeders_num)], [[] for _ in range(feeders_num)], [[] for _ in range(feeders_num)]
for index in range(feeders_num):
    data_path = directory + r'\V_feeder_noDC_' + str(index+1) + '.joblib'
    class_path = directory + r'\Branch_Output_sorted_' + str(index+1) + '.joblib'
    V_feeder_noDC[index] = joblib.load(data_path)
    Branch_Output_sorted[index] = np.array(joblib.load(class_path))
    best_models[index] = pd.read_csv(csv_file + r'\best_branch_id_v03_' + str(index+1) + '.csv', index_col=0)


Matrix, Metrics = [[] for _ in range(3)], [[] for _ in range(3)]
Multilabel_Confusion_Matrix = [[] for _ in range(3)]
Matrix[0], Multilabel_Confusion_Matrix[0], Metrics[0] = get_metrics(V_feeder_noDC[0], Branch_Output_sorted[0], model_f1_loc, best_models[0])
Matrix[1], Multilabel_Confusion_Matrix[1], Metrics[1] = get_metrics(V_feeder_noDC[1], Branch_Output_sorted[1], model_f2_loc, best_models[1])
Matrix[2], Multilabel_Confusion_Matrix[2], Metrics[2] = get_metrics(V_feeder_noDC[2], Branch_Output_sorted[2], model_f3_loc, best_models[2])


joblib.dump(Matrix, csv_file + r'\Matrix_v03.joblib')
joblib.dump(Multilabel_Confusion_Matrix, csv_file + r'\Multilabel_Confusion_Matrix_v03.joblib')
joblib.dump(Metrics, csv_file + r'\Metrics_v03.joblib')
