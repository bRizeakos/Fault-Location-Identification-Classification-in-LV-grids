import pathlib
import pandas as pd
import numpy as np
import tensorflow
import logging
import joblib
from ast import literal_eval
import create_topology as top
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Flatten, BatchNormalization, ConvLSTM2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import random
from tqdm.keras import TqdmCallback

random.seed()
rand_num = int(100 * random.random())
print("seed=", rand_num)

# GPU usage
tensorflow.get_logger().setLevel(logging.INFO)
physical_devices = tensorflow.config.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

# Access path where dataset is stored
PATH = pathlib.Path(pathlib.Path(__file__).parent.resolve()).parent.resolve()
directory = str(PATH) + r'\data\python_data'
csv_file = directory + r'\HyperOptResults_DistanceID'
best_models_loc = csv_file + r'\best_distance_id_v03.csv'

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
branches_num = len(grid_path)  # Number of grid's branches: 9
metrics_num = len(nodes)  # Number of voltage meters: 33
distance_sections = 5  # Number of sections a branch is divided: 5

class_path = directory + r'\Distance_Class_sorted.joblib'
V_branch_noDC_path = directory + r'\V_branch_noDC.joblib'
V_branch_noDC = joblib.load(V_branch_noDC_path)
Distance_Class_sorted = np.array(joblib.load(class_path))
best_models = pd.read_csv(best_models_loc, index_col=0)


def shuffle_dataset(dataset, output_class):
    x_train, x_test, y_train, y_test = train_test_split(dataset, output_class, test_size=0.2, random_state=rand_num)
    return x_train, x_test, y_train, y_test


def reshape_class(y_hat):
    re_y = [0 for _ in range(len(y_hat))]
    segments = len(y_hat[0])
    for i in range(len(y_hat)):
        re_y[i] = float(np.argmax(y_hat[i]))/(segments-1)

    return np.array(re_y)


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


def evaluate_model(dataset, output, model_df):
    epochs = 200
    history = np.empty((len(model_df), epochs))
    x_axis = range(1, epochs + 1)
    for i in range(len(model_df)):
        model = model_df.iloc[i]
        print(model)
        batch_size, filters, layers = int(model['batch_size']), int(model['filters']), int(model['layers'])
        window, rate, units = int(model['window']), literal_eval(model['rate']), literal_eval(model['units'])
        x_train, x_test, y_train, y_test = shuffle_dataset(dataset, output)
        x_train = reshape_2d(x_train, window)
        y_train, y_test = reshape_class(y_train), reshape_class(y_test)
        num_nodes, num_phases, num_steps = x_train.shape[2], x_train.shape[3], x_train.shape[4]
        # fit and evaluate the best model

        # define model
        verbose = 1

        # define model
        best_model = Sequential()
        if layers == 2:
            # First layer specifies input_shape
            best_model.add(ConvLSTM2D(filters=filters, kernel_size=(num_nodes - 1, num_phases - 1),
                                      activation='relu',
                                      input_shape=(window, num_nodes, num_phases, num_steps),
                                      data_format='channels_last'))
            best_model.add(Dropout(rate=rate[0]))
            best_model.add(BatchNormalization())
            best_model.add(Flatten())
        else:
            # First layer specifies input_shape
            best_model.add(ConvLSTM2D(filters=filters, kernel_size=(num_nodes - 1, num_phases - 1),
                                      activation='relu',
                                      input_shape=(window, num_nodes, num_phases, num_steps),
                                      data_format='channels_last'))
            best_model.add(Dropout(rate=rate[0]))
            best_model.add(BatchNormalization())
            best_model.add(Flatten())

            # Middle layers return sequences
            for idx in range(layers - 2):
                best_model.add(Dense(units=units[idx + 1]))
                best_model.add(Dropout(rate=rate[idx + 1]))
                best_model.add(BatchNormalization())

            # Last layer doesn't return anything
        best_model.add(Dense(1, activation='sigmoid'))
        best_model.compile(optimizer='adam', loss="mean_squared_error")
        best_model.summary()

        es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=25)

        history_i = best_model.fit(x_train, y_train, validation_split=0.2, epochs=epochs,
                                   batch_size=batch_size, verbose=verbose, callbacks=[es, TqdmCallback(verbose=0)])

        best_model.save(csv_file + r'\model_class_v03_' + str(i + 1))
        len_of_history = len(history_i.history['loss'])
        history[i, :len_of_history] = history_i.history['loss']
        if len_of_history < epochs:
            history[i, len_of_history:] = [history[i, len_of_history - 1] for _ in range(epochs - len_of_history)]

    plt.rc('font', size=7)
    plt.plot(100 * history[0, :], label='1st model', color='red')
    plt.plot(100 * history[1, :], label='2nd model', color='green')
    plt.plot(100 * history[2, :], label='3rd model', color='blue')
    plt.title('Loss of models')
    plt.ylabel('Loss (%)')
    plt.xlabel('Trials')
    plt.legend(loc='upper right', fontsize='small')
    plt.show()

    plt.savefig(csv_file + r'\model_v03.jpg')

    return history


# Load Data
best_models = pd.read_csv(best_models_loc, index_col=0)

history_distance = evaluate_model(V_branch_noDC, Distance_Class_sorted, best_models)
joblib.dump(history_distance, csv_file + r'\history_distance_v03.joblib')
