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
csv_file = directory + r'\HyperOptResults_BranchID'
best_models_loc = [[] for _ in range(3)]
best_models_loc[0] = csv_file + r'\best_branch_id_v03_1.csv'
best_models_loc[1] = csv_file + r'\best_branch_id_v03_2.csv'
best_models_loc[2] = csv_file + r'\best_branch_id_v03_3.csv'

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

V_feeder_noDC, Branch_Output_sorted, best_models = [[] for _ in range(feeders_num)], \
                                                   [[] for _ in range(feeders_num)], [[] for _ in range(feeders_num)]
for index in range(feeders_num):
    data_path = directory + r'\V_feeder_noDC_' + str(index+1) + '.joblib'
    class_path = directory + r'\Branch_Output_sorted_' + str(index+1) + '.joblib'
    V_feeder_noDC[index] = joblib.load(data_path)
    Branch_Output_sorted[index] = np.array(joblib.load(class_path))
    best_models[index] = pd.read_csv(best_models_loc[index], index_col=0)
    
    
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


def evaluate_model(number, dataset, output, model_df):

    epochs = 200
    history = np.empty((len(model_df), 4, epochs))
    x_axis = range(1, epochs+1)
    for i in range(len(model_df)):
        model = model_df.iloc[i]
        print(model)
        batch_size, filters, layers = int(model['batch_size']), int(model['filters']), int(model['layers'])
        window, rate, units = int(model['window']), literal_eval(model['rate']), literal_eval(model['units'])
        x_train, x_test, y_train, y_test = shuffle_dataset(dataset, output)
    
        x_train = reshape_2d(x_train, window)
        num_steps, num_branches, num_nodes = x_train.shape[4], x_train.shape[2], x_train.shape[3]
        n_outputs = len(y_train[0])
        # fit and evaluate the best model

        # define model
        verbose = 1

        # define model
        best_model = Sequential()
        if layers == 2:
            # First layer specifies input_shape
            best_model.add(ConvLSTM2D(filters=filters, kernel_size=(num_branches - 1, num_steps - 1),
                                      activation='relu',
                                      input_shape=(window, num_branches, num_nodes, num_steps),
                                      data_format='channels_last'))

            best_model.add(Dropout(rate=rate[0]))
            best_model.add(BatchNormalization())
            best_model.add(Flatten())
        else:
            # First layer specifies input_shape
            best_model.add(ConvLSTM2D(filters=filters, kernel_size=(num_branches - 1, num_steps - 1),
                                      activation='relu',
                                      input_shape=(window, num_branches, num_nodes, num_steps),
                                      data_format='channels_last'))
            best_model.add(Dropout(rate=rate[0]))
            best_model.add(BatchNormalization())
            best_model.add(Flatten())
    
            # Middle layers return sequences
            for idx in range(layers - 2):
                best_model.add(Dense(units=units[idx+1], activation='relu'))
                best_model.add(Dropout(rate=rate[idx+1]))
                best_model.add(BatchNormalization())
    
            # Last layer doesn't return anything
        best_model.add(Dense(n_outputs, activation='softmax'))
        if n_outputs < 3:
            loss_fun = 'binary_crossentropy'
        else:
            loss_fun = 'categorical_crossentropy'
        best_model.compile(optimizer='adam', loss=loss_fun, metrics=["accuracy"])
        best_model.summary()

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)

        history_i = best_model.fit(x_train, y_train, validation_split=0.2, epochs=epochs,
                                   batch_size=batch_size, verbose=verbose, callbacks=[es, TqdmCallback(verbose=0)])

        best_model.save(csv_file + r'\model_v03_f' + str(number) + '_' + str(i+1))
        len_of_history = len(history_i.history['accuracy'])
        history[i, 0, :len_of_history] = history_i.history['accuracy']
        if len_of_history < epochs:
            history[i, 0, len_of_history:] = [history[i, 0, len_of_history-1] for _ in range(epochs - len_of_history)]
        history[i, 1, :len_of_history] = history_i.history['loss']
        if len_of_history < epochs:
            history[i, 1, len_of_history:] = [history[i, 1, len_of_history-1] for _ in range(epochs - len_of_history)]
        history[i, 2, :len_of_history] = history_i.history['val_accuracy']
        if len_of_history < epochs:
            history[i, 2, len_of_history:] = [history[i, 2, len_of_history-1] for _ in range(epochs - len_of_history)]
        history[i, 3, :len_of_history] = history_i.history['val_loss']
        if len_of_history < epochs:
            history[i, 3, len_of_history:] = [history[i, 3, len_of_history-1] for _ in range(epochs - len_of_history)]

    max_loss = np.amax(history[:, 1, :])
    max_val_loss = np.amax(history[:, 3, :])

    plt.rc('font', size=7)
    fig, axs = plt.subplots(nrows=2, ncols=2, dpi=250, sharex=True)
    if number == 1:
        fig.suptitle('History Data of Model: Feeder A')
    elif number == 2:
        fig.suptitle('History Data of Model: Feeder B')
    elif number == 3:
        fig.suptitle('History Data of Model: Feeder C')
    else:
        return False

    axs[0, 0].plot(x_axis, 100*history[0, 0, :], label='1st model', color='red')
    axs[0, 0].plot(x_axis, 100*history[1, 0, :], label='2nd model', color='green')
    axs[0, 0].plot(x_axis, 100*history[2, 0, :], label='3rd model', color='blue')
    axs[0, 0].set_title('Accuracy of models')
    axs[0, 0].set_ylim([-5, 105])
    axs[0, 0].set_xlabel('Trials')
    axs[0, 0].set_ylabel('Accuracy (%)')
    axs[0, 0].legend(loc='lower right', fontsize='small')

    axs[0, 1].plot(x_axis, 100*history[0, 1, :], label='1st model', color='red')
    axs[0, 1].plot(x_axis, 100*history[1, 1, :], label='2nd model', color='green')
    axs[0, 1].plot(x_axis, 100*history[2, 1, :], label='3rd model', color='blue')
    axs[0, 1].set_title('Loss of models')
    # axs[0, 1].set_ylim([-5, 100*max_loss + 5])
    axs[0, 1].set_ylim([-5, 205])
    axs[0, 1].set_xlabel('Trials')
    axs[0, 1].set_ylabel('Loss (%)')
    axs[0, 1].legend(loc='upper right', fontsize='small')

    axs[1, 0].plot(x_axis, 100*history[0, 2, :], label='1st model', color='red')
    axs[1, 0].plot(x_axis, 100*history[1, 2, :], label='2nd model', color='green')
    axs[1, 0].plot(x_axis, 100*history[2, 2, :], label='3rd model', color='blue')
    axs[1, 0].set_title('Validation Accuracy of models')
    axs[1, 0].set_ylim([-5, 105])
    axs[1, 0].set_xlabel('Trials')
    axs[1, 0].set_ylabel('Validation Accuracy (%)')
    axs[1, 0].legend(loc='lower right', fontsize='small')

    axs[1, 1].plot(x_axis, 100*history[0, 3, :], label='1st model', color='red')
    axs[1, 1].plot(x_axis, 100*history[1, 3, :], label='2nd model', color='green')
    axs[1, 1].plot(x_axis, 100*history[2, 3, :], label='3rd model', color='blue')
    axs[1, 1].set_title('Validation Loss of models')
    # axs[1, 1].set_ylim([-5, 100*max_val_loss + 5])
    axs[1, 1].set_ylim([-5, 205])
    axs[1, 1].set_xlabel('Trials')
    axs[1, 1].set_ylabel('Validation Loss (%)')
    axs[1, 1].legend(loc='upper right', fontsize='small')

    plt.savefig(csv_file + r'\model_v03' + str(number) + r'.jpg')

    return history


# Load Data
best_models_f1 = pd.read_csv(best_models_loc[0], index_col=0)
best_models_f2 = pd.read_csv(best_models_loc[1], index_col=0)
best_models_f3 = pd.read_csv(best_models_loc[2], index_col=0)

history_f1 = evaluate_model(1, V_feeder_noDC[0], Branch_Output_sorted[0], best_models_f1)
joblib.dump(history_f1, csv_file + r'\history_v03_f1.joblib')
history_f2 = evaluate_model(2, V_feeder_noDC[1], Branch_Output_sorted[1], best_models_f2)
joblib.dump(history_f2, csv_file + r'\history_v03_f2.joblib')
history_f3 = evaluate_model(3, V_feeder_noDC[2], Branch_Output_sorted[2], best_models_f3)
joblib.dump(history_f3, csv_file + r'\history_v03_f3.joblib')
