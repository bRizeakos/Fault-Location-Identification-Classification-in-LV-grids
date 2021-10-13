# Finds the Best HyperParameters for the Fault Type Identification NN

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import pathlib
import joblib
import tensorflow
import numpy as np
import pandas as pd
import create_topology as top
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Flatten, ConvLSTM2D, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import logging
from tqdm.keras import TqdmCallback
import random
import time

random.seed()
rand_num = int(100 * random.random())
print("seed=", rand_num)
cce = tensorflow.keras.losses.CategoricalCrossentropy()

# GPU usage
tensorflow.get_logger().setLevel(logging.INFO)
physical_devices = tensorflow.config.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

# Load data
PATH = pathlib.Path(pathlib.Path(__file__).parent.resolve()).parent.resolve()
directory = str(PATH) + r'\data\python_data'

V_branch_noDC_path = directory + r'\V_branch_noDC.joblib'
Fault_Class_sorted_path = directory + r'\Fault_Class_sorted.joblib'
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

V_branch_noDC = joblib.load(V_branch_noDC_path)
Fault_Class_sorted = np.array(joblib.load(Fault_Class_sorted_path))


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


filters_space = [8, 16, 32, 64, 128, 256]
window_space = [25, 50]


def set_space(num_of_max_layers):
    end_space = {'batch_size': scope.int(hp.quniform('batch_size', 100, 250, 25)),
                 'layers': scope.int(hp.quniform('layers', 2, max_layers, 1)),
                 'filters': hp.choice('filters', filters_space),
                 'window': hp.choice('window', window_space)}

    for idx in range(num_of_max_layers):
        unit_name = 'units' + str(idx + 1)
        rate_name = 'rate' + str(idx + 1)
        end_space[unit_name] = scope.int(hp.quniform(unit_name, 20, 200, 20))
        end_space[rate_name] = hp.uniform(rate_name, 0.01, 0.5)

    return end_space


# Create the variables space of exploration
max_layers = 7
space = set_space(max_layers)
results_of_fnn = []


# Define the function that defines model
def f_nn(params):
    dataset, output_class = V_branch_noDC, Fault_Class_sorted
    x_train, x_test, y_train, y_test = shuffle_dataset(dataset, output_class)

    # define model
    num_outputs = len(y_train[0])
    # reshape into subsequences (samples, window_size, nodes, phases, sequences)

    b_size, layer, num_length, num_filters = params['batch_size'], params['layers'], params['window'], params['filters']
    unt = [0 for _ in range(layer)]
    unt[0] = num_length
    if layer > 2:
        unt[1:-1] = [params['units' + str(idx + 1)] for idx in range(1, layer - 1)]
    unt[-1] = num_outputs

    rt = [params['rate' + str(idx + 1)] for idx in range(layer - 1)]
    rt.append(0)

    x_train = reshape_2d(x_train, num_length)
    num_nodes, num_phases, num_steps = x_train.shape[2], x_train.shape[3], x_train.shape[4]

    # Keras LSTM model
    model = Sequential()

    if layer == 2:
        # First layer specifies input_shape
        model.add(ConvLSTM2D(filters=num_filters, kernel_size=(num_nodes - 1, num_phases - 1),
                             activation='relu',
                             input_shape=(num_length, num_nodes, num_phases, num_steps),
                             data_format='channels_last'))
        model.add(Dropout(rate=rt[0]))
        model.add(BatchNormalization())
        model.add(Flatten())
    else:
        # First layer specifies input_shape
        model.add(ConvLSTM2D(filters=num_filters, kernel_size=(num_nodes - 1, num_phases - 1),
                             activation='relu',
                             input_shape=(num_length, num_nodes, num_phases, num_steps),
                             data_format='channels_last'))
        model.add(Dropout(rate=rt[0]))
        model.add(BatchNormalization())
        model.add(Flatten())

        # Middle layers return sequences

        for idx in range(layer - 2):
            model.add(Dense(units=unt[idx + 1], activation='relu'))
            model.add(Dropout(rate=rt[idx + 1]))
            model.add(BatchNormalization())

        # Last layer doesn't return anything
    model.add(Dense(num_outputs, activation='softmax'))
    if num_outputs < 3:
        loss_fun = 'binary_crossentropy'
    else:
        loss_fun = 'categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss_fun, metrics=["accuracy"])
    model.summary()

    es = EarlyStopping(monitor='val_loss', mode='min',
                       verbose=0, patience=25)
    start_time = time.time()
    result = model.fit(x_train, y_train,
                       verbose=0,
                       validation_split=0.2,
                       batch_size=b_size,
                       epochs=200,
                       callbacks=[es, TqdmCallback(verbose=0)])
    end_time = time.time()
    dt = end_time - start_time
    number_of_epochs_it_ran = len(result.history['loss'])
    avg_time = dt / number_of_epochs_it_ran
    # Get the highest accuracy of the training epochs
    acc = round(np.amax(result.history['accuracy']), 5)
    # Get the lowest loss of the training epochs
    loss = round(np.amin(result.history['loss']), 5)
    # Get the highest validation accuracy of the training epochs
    val_acc = round(np.amax(result.history['val_accuracy']), 5)
    # Get the lowest validation loss of the training epochs
    val_loss = round(np.amin(result.history['val_loss']), 5)

    print('Best validation loss of epoch:', val_loss)

    val_length = int(0.2 * len(y_test))
    x_val, y_val = x_train[-val_length:], y_train[-val_length:]
    yp = model.predict(x_val)
    cross = cce(y_val, yp).numpy()\

    print('Cross-entropy loss of eval:', cross)
    _, score = model.evaluate(x_train, y_train, verbose=0)

    print("Train accuracy: %.2f%%" % (100 * score))
    results_of_fnn.append([100 * acc, 100 * loss, 100 * val_acc, 100 * val_loss, b_size, unt, rt, layer, num_length,
                           num_filters, number_of_epochs_it_ran, avg_time, dt])

    names = ['Accuracy (%)', 'Loss (%)', 'Val_ Accuracy (%)', 'Val_Loss (%)', 'batch_size', 'units', 'rate', 'layers',
             'window', 'filters', 'Number of Epochs', 'Average Time per Epoch (s)', 'Total Time']
    df = pd.DataFrame(results_of_fnn, columns=names)
    try:
        outfile = open(directory + r'\fault_id_v03.csv', 'wb')
        df.to_csv(outfile)
        outfile.close()
    except IOError as error:
        print(error)

    return {'loss': cross, 'status': STATUS_OK, 'model': model, 'params': params}


trials = Trials()
best = fmin(f_nn,
            space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)  # max_evals=200
# print(results_of_fnn)

best_model = trials.results[np.argmin([r['loss'] for r in
                                       trials.results])]['model']
best_params = trials.results[np.argmin([r['loss'] for r in
                                        trials.results])]['params']
worst_model = trials.results[np.argmax([r['loss'] for r in
                                        trials.results])]['model']
worst_params = trials.results[np.argmax([r['loss'] for r in
                                         trials.results])]['params']

print(trials.trials)
print(best_model)
print(best_params)
print(worst_model)
print(worst_params)

print("Best estimate parameters", best)
batch_size, epochs, filters, layers = int(best['batch_size']), 200, filters_space[best['filters']], int(best['layers'])
window = window_space[best['window']]
trainX, testX, trainY, testY = shuffle_dataset(V_branch_noDC, Fault_Class_sorted)
n_outputs = len(trainY[0])

rate = [best['rate' + str(idx + 1)] for idx in range(layers - 1)]
rate.append(0)

units = [0 for _ in range(layers)]
units[0] = window
if layers > 2:
    units[1:-1] = [int(best['units' + str(idx + 1)]) for idx in range(1, layers - 1)]
units[-1] = n_outputs

# fit and evaluate the best model

# define model
verbose = 1
# reshape into subsequences (samples, time steps, rows, cols, channels)
n_length = window
trainX = reshape_2d(trainX, n_length)
testX = reshape_2d(testX, n_length)
n_nodes, n_phases, n_steps = trainX.shape[2], trainX.shape[3], trainX.shape[4]
print('batch_size=%s' % batch_size)
print('epochs=%s' % epochs)
print('filters=%s' % filters)
print('layers=%s' % layers)
print('rate=%s' % rate)
print('units=%s' % units)
print('window=%s' % window)
print('n_outputs=%s' % n_outputs)
print('n_length=%s' % n_length)
print('n_steps=%s' % n_steps)

# define model
last_model = Sequential()
if layers == 2:
    last_model.add(ConvLSTM2D(filters=filters, kernel_size=(n_nodes - 1, n_phases - 1), activation='relu',
                              input_shape=(n_length, n_nodes, n_phases, n_steps), data_format='channels_last'))
    last_model.add(Dropout(rate=rate[0]))
    last_model.add(BatchNormalization())
    last_model.add(Flatten())
else:
    # First layer specifies input_shape and returns sequences
    last_model.add(ConvLSTM2D(filters=filters, kernel_size=(n_nodes - 1, n_phases - 1), activation='relu',
                              input_shape=(n_length, n_nodes, n_phases, n_steps), data_format='channels_last'))
    last_model.add(Dropout(rate=rate[0]))
    last_model.add(BatchNormalization())
    last_model.add(Flatten())

    # Middle layers return sequences
    for index in range(layers - 2):
        last_model.add(Dense(units=units[index + 1], activation='relu'))
        last_model.add(Dropout(rate=rate[index + 1]))
        last_model.add(BatchNormalization())

    # Last layer doesn't return anything
last_model.add(Dense(n_outputs, activation='softmax'))
if n_outputs < 3:
    loss_func = 'binary_crossentropy'
else:
    loss_func = 'categorical_crossentropy'
last_model.compile(optimizer='adam', loss=loss_func, metrics=["accuracy"])
last_model.summary()
last_model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
_, accuracy = last_model.evaluate(testX, testY, batch_size=batch_size, verbose=0)

print("Accuracy of the best model: %.3f%%" % (100 * accuracy))
