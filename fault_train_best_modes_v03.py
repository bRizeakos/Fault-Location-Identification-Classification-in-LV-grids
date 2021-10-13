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
csv_file = directory + r'\HyperOptResults_ClassID'
best_models_loc = csv_file + r'\best_fault_id_v03.csv'

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

class_path = directory + r'\Fault_Class_sorted.joblib'
V_branch_noDC_path = directory + r'\V_branch_noDC.joblib'
V_branch_noDC = joblib.load(V_branch_noDC_path)
Fault_Class_sorted = np.array(joblib.load(class_path))
best_models = pd.read_csv(best_models_loc, index_col=0)


def shuffle_dataset(dataset, output_class):
    x_trn, x_tst, y_trn, y_tst = [], [], [], []

    scenario_length = [9535, 9535, 9535, 1022, 1022, 1022, 2384, 2384, 2384, 817, 1225]

    for idx in range(len(scenario_length)):
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[sum(scenario_length[:idx]):sum(scenario_length[:idx + 1])],
            output_class[sum(scenario_length[:idx]):sum(scenario_length[:idx + 1])],
            test_size=0.2, random_state=rand_num)
        for j in x_train:
            x_trn.append(j)
        for j in x_test:
            x_tst.append(j)
        for j in y_train:
            y_trn.append(j)
        for j in y_test:
            y_tst.append(j)

    temp1 = list(zip(x_trn, y_trn))
    temp2 = list(zip(x_tst, y_tst))

    random.shuffle(temp1), random.shuffle(temp1)

    x_trn, y_trn = zip(*temp1)
    x_tst, y_tst = zip(*temp2)

    x_trn = np.stack(x_trn, axis=0)
    x_tst = np.stack(x_tst, axis=0)
    y_trn = np.stack(y_trn, axis=0)
    y_tst = np.stack(y_tst, axis=0)

    return x_trn, x_tst, y_trn, y_tst


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
    history = np.empty((len(model_df), 4, epochs))
    x_axis = range(1, epochs + 1)
    for i in range(len(model_df)):
        model = model_df.iloc[i]
        print(model)
        batch_size, filters, layers = int(model['batch_size']), int(model['filters']), int(model['layers'])
        window, rate, units = int(model['window']), literal_eval(model['rate']), literal_eval(model['units'])
        x_train, x_test, y_train, y_test = shuffle_dataset(dataset, output)
        num_outputs = len(y_train[0])
        x_train = reshape_2d(x_train, window)
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
                best_model.add(Dense(units=units[idx + 1], activation='relu'))
                best_model.add(Dropout(rate=rate[idx + 1]))
                best_model.add(BatchNormalization())

            # Last layer doesn't return anything
        best_model.add(Dense(num_outputs, activation='softmax'))
        if num_outputs < 3:
            loss_fun = 'binary_crossentropy'
        else:
            loss_fun = 'categorical_crossentropy'
        best_model.compile(optimizer='adam', loss=loss_fun, metrics=["accuracy"])
        best_model.summary()

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)

        history_i = best_model.fit(x_train, y_train, validation_split=0.2, epochs=epochs,
                                   batch_size=batch_size, verbose=verbose, callbacks=[es, TqdmCallback(verbose=0)])

        best_model.save(csv_file + r'\model_class_v03_' + str(i + 1))
        len_of_history = len(history_i.history['accuracy'])
        history[i, 0, :len_of_history] = history_i.history['accuracy']
        if len_of_history < epochs:
            history[i, 0, len_of_history:] = [history[i, 0, len_of_history - 1] for _ in range(epochs - len_of_history)]
        history[i, 1, :len_of_history] = history_i.history['loss']
        if len_of_history < epochs:
            history[i, 1, len_of_history:] = [history[i, 1, len_of_history - 1] for _ in range(epochs - len_of_history)]
        history[i, 2, :len_of_history] = history_i.history['val_accuracy']
        if len_of_history < epochs:
            history[i, 2, len_of_history:] = [history[i, 2, len_of_history - 1] for _ in range(epochs - len_of_history)]
        history[i, 3, :len_of_history] = history_i.history['val_loss']
        if len_of_history < epochs:
            history[i, 3, len_of_history:] = [history[i, 3, len_of_history - 1] for _ in range(epochs - len_of_history)]

    max_loss = np.amax(history[:, 1, :])
    max_val_loss = np.amax(history[:, 3, :])

    plt.rc('font', size=7)
    fig, axs = plt.subplots(nrows=2, ncols=2, dpi=250, sharex=True)
    fig.suptitle('History Data of Model')

    axs[0, 0].plot(x_axis, 100 * history[0, 0, :], label='1st model', color='red')
    axs[0, 0].plot(x_axis, 100 * history[1, 0, :], label='2nd model', color='green')
    axs[0, 0].plot(x_axis, 100 * history[2, 0, :], label='3rd model', color='blue')
    axs[0, 0].set_title('Accuracy of models')
    axs[0, 0].set_ylim([-5, 105])
    axs[0, 0].set_xlabel('Trials')
    axs[0, 0].set_ylabel('Accuracy (%)')
    axs[0, 0].legend(loc='lower right', fontsize='small')

    axs[0, 1].plot(x_axis, 100 * history[0, 1, :], label='1st model', color='red')
    axs[0, 1].plot(x_axis, 100 * history[1, 1, :], label='2nd model', color='green')
    axs[0, 1].plot(x_axis, 100 * history[2, 1, :], label='3rd model', color='blue')
    axs[0, 1].set_title('Loss of models')
    # axs[0, 1].set_ylim([-5, 100*max_loss + 5])
    axs[0, 1].set_ylim([-5, 205])
    axs[0, 1].set_xlabel('Trials')
    axs[0, 1].set_ylabel('Loss (%)')
    axs[0, 1].legend(loc='upper right', fontsize='small')

    axs[1, 0].plot(x_axis, 100 * history[0, 2, :], label='1st model', color='red')
    axs[1, 0].plot(x_axis, 100 * history[1, 2, :], label='2nd model', color='green')
    axs[1, 0].plot(x_axis, 100 * history[2, 2, :], label='3rd model', color='blue')
    axs[1, 0].set_title('Validation Accuracy of models')
    axs[1, 0].set_ylim([-5, 105])
    axs[1, 0].set_xlabel('Trials')
    axs[1, 0].set_ylabel('Validation Accuracy (%)')
    axs[1, 0].legend(loc='lower right', fontsize='small')

    axs[1, 1].plot(x_axis, 100 * history[0, 3, :], label='1st model', color='red')
    axs[1, 1].plot(x_axis, 100 * history[1, 3, :], label='2nd model', color='green')
    axs[1, 1].plot(x_axis, 100 * history[2, 3, :], label='3rd model', color='blue')
    axs[1, 1].set_title('Validation Loss of models')
    # axs[1, 1].set_ylim([-5, 100*max_val_loss + 5])
    axs[1, 1].set_ylim([-5, 205])
    axs[1, 1].set_xlabel('Trials')
    axs[1, 1].set_ylabel('Validation Loss (%)')
    axs[1, 1].legend(loc='upper right', fontsize='small')

    plt.savefig(csv_file + r'\model_v03.jpg')

    return history


# Load Data
best_models = pd.read_csv(best_models_loc, index_col=0)

history_class = evaluate_model(V_branch_noDC, Fault_Class_sorted, best_models)
joblib.dump(history_class, csv_file + r'\history_class_v03.joblib')
