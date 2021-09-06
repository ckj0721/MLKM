#!/bin/python

# Import Library
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys


# ARGV
MACHINE = str(sys.argv[1])
DEVICES = str(sys.argv[2])
BATCHING = int(sys.argv[3])
LEARNING_RATE = float(sys.argv[4])
SYSTEM_SIZE = int(sys.argv[5])
COUPLING = float(sys.argv[6])
LENGTH_IN = int(sys.argv[7])
MODEL = str(sys.argv[8])
LABEL = int(sys.argv[9])
DATA = int(sys.argv[10])
MAX_EPOCH = int(sys.argv[11])
PREV_TRIAL = int(sys.argv[12])
TRIAL = int(sys.argv[13])


# CONSTANT
# REGULARIZATION_RATE = LEARNING_RATE
DIM = int(2 * SYSTEM_SIZE)
SHIFT = 6.28
FEET = 1000
DT = 0.005
TIME_SERIES_TRAIN = 1
TIME_SERIES_TEST = 2
ENSEMBLE_SIZE = 2000
TRAIN_INI = 4
TRAIN_END = 94
LENGTH_TEST = 9800



def ann(model, label):
    
    if model == "FFFF":
        input = tf.keras.Input(shape = (DIM * LENGTH_IN,), dtype = tf.float32)
        x = tf.keras.layers.Dense(DIM * label, activation = tf.keras.activations.tanh)(input)
        x = tf.keras.layers.Dense(DIM * label, activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.Dense(DIM * label, activation = tf.keras.activations.tanh)(x)
        output = tf.keras.layers.Dense(DIM, activation=tf.keras.activations.tanh)(x)

    elif model == "11FFF":
        input = tf.keras.Input(shape = (DIM * LENGTH_IN,), dtype = tf.float32)
        x = tf.keras.layers.Reshape((LENGTH_IN, DIM,), input_shape=(DIM * LENGTH_IN,))(input)
        x = tf.keras.layers.Conv1D(DIM * label, 3, activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.Conv1D(DIM * label, 3, activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(DIM * label, activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.Dense(DIM * label, activation = tf.keras.activations.tanh)(x)
        output = tf.keras.layers.Dense(DIM, activation=tf.keras.activations.tanh)(x)

    elif model == "22FFF":
        input = tf.keras.Input(shape = (DIM * LENGTH_IN,), dtype = tf.float32)
        x = tf.keras.layers.Reshape((LENGTH_IN, SYSTEM_SIZE, 2,), input_shape=(DIM * LENGTH_IN,))(input)
        x = tf.keras.layers.Conv2D(2 * label, (3, 3), activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.Conv2D(2 * label, (3, 3), activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(DIM * label, activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.Dense(DIM * label, activation = tf.keras.activations.tanh)(x)
        output = tf.keras.layers.Dense(DIM, activation=tf.keras.activations.tanh)(x)

    elif model == "LL":
        input = tf.keras.Input(shape = (DIM * LENGTH_IN,), dtype = tf.float32)
        x = tf.keras.layers.Reshape((LENGTH_IN, DIM,), input_shape=(DIM * LENGTH_IN,))(input)
        x = tf.keras.layers.LSTM(SYSTEM_SIZE * label, return_sequences = True)(x)
        output = tf.keras.layers.LSTM(DIM)(x)
        
    model = tf.keras.Model(input, output)
    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE), loss = tf.keras.losses.MSE)

    return model



save = '/pds/pds151/ckj/MLKM/MLKM_Tracer_5/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/save_{trial:d}/save.ckpt'

#parameters depends on machine
if MACHINE == 'GPUs':
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICES
    strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE_REP = BATCHING
    BATCH_SIZE = BATCH_SIZE_REP * strategy.num_replicas_in_sync
    SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 4
    THREAD_DATA = 2 * strategy.num_replicas_in_sync
    with strategy.scope():
        model = ann(MODEL, LABEL)
        model.load_weights(save.format(N = SYSTEM_SIZE, L = LENGTH_IN, K = COUPLING, model = MODEL, label = LABEL, trial = PREV_TRIAL))
elif MACHINE == 'CPU' or  MACHINE == 'GPU':
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICES
    BATCH_SIZE = BATCHING
    SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 4
    THREAD_DATA = 2
    model = ann(MODEL, LABEL)
    model.load_weights(save.format(N = SYSTEM_SIZE, L = LENGTH_IN, K = COUPLING, model = MODEL, label = LABEL, trial = PREV_TRIAL))
else:
    print("Select MACHINE : CPU/GPU/GPUs")
    raise SystemExit




file_image = '/pds/pds{data:d}/ckj/MLKM/train_prediction/N{N:d}K{K:0.4f}L{L:d}/Phaset_w{w:0.2f}_N{N:d}_K{K:0.4f}_ft{ft:d}_dt{dt:0.3f}_E{e:d}_{L:d}_{E:d}.dat'
def list_of_file_image(Emin, Emax):
    list = []
    for E in np.arange(Emin, Emax + 0.5):
        list.append(file_image.format(N = SYSTEM_SIZE, K = COUPLING, w = SHIFT, ft = FEET, dt = DT, e = TIME_SERIES_TRAIN, L = LENGTH_IN, E = int(E), data = DATA))
    return list

file_label = '/pds/pds{data:d}/ckj/MLKM/train_prediction/N{N:d}K{K:0.4f}L{L:d}/Phaset_w{w:0.2f}_N{N:d}_K{K:0.4f}_ft{ft:d}_dt{dt:0.3f}_E{e:d}_{L:d}_{E:d}.label'
def list_of_file_label(Emin, Emax):
    list = []
    for E in np.arange(Emin, Emax + 0.5):
        list.append(file_label.format(N = SYSTEM_SIZE, K = COUPLING, w = SHIFT, ft = FEET, dt = DT, e = TIME_SERIES_TRAIN, L = LENGTH_IN, E = int(E), data = DATA))
    return list


def make_dataset_train(size_of_ensemble, Emin, Emax):
    list_of_image = list_of_file_image(Emin, Emax)
    list_of_label = list_of_file_label(Emin, Emax)

    number_of_files = len(list_of_label)
    number_of_total = number_of_files * size_of_ensemble

    # labels = tf.data.Dataset.from_tensor_slices(list_of_label).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    # images = tf.data.Dataset.from_tensor_slices(list_of_image).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    labels = tf.data.Dataset.from_tensor_slices(list_of_label).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = THREAD_DATA)
    images = tf.data.Dataset.from_tensor_slices(list_of_image).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = THREAD_DATA)
    dataset = tf.data.Dataset.zip((images, labels))

    # map and cache
    def decorder(image, label):
        image = tf.strings.split(image)
        image = tf.strings.to_number(image, out_type = tf.float32)
        label = tf.strings.split(label)
        label = tf.strings.to_number(label, out_type = tf.float32)
        return image, label

    dataset = dataset.map(decorder, num_parallel_calls = THREAD_DATA).cache()

    # split training and validation
    number_of_train = int(number_of_total * 0.9)
    number_of_valid = number_of_total - number_of_train
    dataset_valid = dataset.take(number_of_valid)
    dataset_train = dataset.skip(number_of_valid)
    
    # repeat and shuffle
    dataset_train = dataset_train.shuffle(SHUFFLE_BUFFER_SIZE).repeat()
    dataset_valid = dataset_valid.shuffle(SHUFFLE_BUFFER_SIZE).repeat()

    
    # dataset_train = dataset_train.map(decorder, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    # dataset_valid = dataset_valid.map(decorder, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    # dataset_train = dataset_train.map(decorder, num_parallel_calls = THREAD_DATA)
    # dataset_valid = dataset_valid.map(decorder, num_parallel_calls = THREAD_DATA)

    # batch and prefetch
    dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder = True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    dataset_valid = dataset_valid.batch(BATCH_SIZE, drop_remainder = True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
  
    return dataset_train, dataset_valid, number_of_train, number_of_valid



dataset_train, dataset_valid, number_of_train, number_of_valid = make_dataset_train(ENSEMBLE_SIZE, TRAIN_INI, TRAIN_END)


history = model.fit(dataset_train, verbose = 0, epochs = MAX_EPOCH, steps_per_epoch = int(number_of_train / BATCH_SIZE), validation_data = dataset_valid, validation_steps = int(number_of_valid / BATCH_SIZE))

# earlystop_callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = MAX_EPOCH, restore_best_weights = True)]
# earlystop_callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = MAX_EPOCH)]
# history = model.fit(dataset_train, verbose = 0, epochs = int(10 * MAX_EPOCH), steps_per_epoch = int(number_of_train / BATCH_SIZE), validation_data = dataset_valid, validation_steps = int(number_of_valid / BATCH_SIZE), callbacks = earlystop_callbacks)




history_pdf = '/pds/pds151/ckj/MLKM/MLKM_Tracer_5/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/history_{trial:d}.pdf'
history_dat = '/pds/pds151/ckj/MLKM/MLKM_Tracer_5/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/history_{trial:d}.dat'

def plot_history(hist, filename):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(hist[:,0], label='Train')
    plt.plot(hist[:,1], label = 'Validation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)

# hist = np.append(np.loadtxt(history_dat.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, trial = PREV_TRIAL)), np.column_stack((history.history['loss'], history.history['val_loss']))[:-MAX_EPOCH], axis = 0)
hist = np.append(np.loadtxt(history_dat.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, trial = PREV_TRIAL)), np.column_stack((history.history['loss'], history.history['val_loss'])), axis = 0)

plot_history(hist, history_pdf.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, trial = TRIAL))
np.savetxt(history_dat.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, trial = TRIAL), hist, fmt = "%e")



save_dir = '/pds/pds151/ckj/MLKM/MLKM_Tracer_5/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/save_{trial:d}/'
if not os.path.exists(save_dir.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, trial = TRIAL)):
    os.mkdir(save_dir.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, trial = TRIAL))
model.save_weights(save.format(N = SYSTEM_SIZE, L = LENGTH_IN, K = COUPLING, model = MODEL, label = LABEL, trial = TRIAL))



file = '/pds/pds121/KMML/train_prediction/Phaset_w{w:0.2f}_N{N:d}_K{K:0.4f}_ft{ft:d}_dt{dt:0.3f}_E{e:d}.dat'

def test_same_init(offset, length_test):

    data = np.loadtxt(file.format(N = SYSTEM_SIZE, K = COUPLING, w = SHIFT, ft = FEET, dt = DT, e = TIME_SERIES_TRAIN))
    test_output = np.zeros((length_test, DIM), dtype=np.float32)

    test_image = data[offset:offset + LENGTH_IN]
    for i in np.arange(length_test):
        if MACHINE == 'GPUs':
            test_dataset = tf.data.Dataset.from_tensor_slices(np.reshape(test_image, (1, LENGTH_IN * DIM)))
            test_dataset = test_dataset.repeat()
            test_dataset = test_dataset.batch(strategy.num_replicas_in_sync, drop_remainder = True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
            test_output[i] = model.predict(test_dataset, steps = 1)[0]
        elif MACHINE == 'CPU' or MACHINE == 'GPU':
            test_output[i] = model.predict(np.reshape(test_image, (1, LENGTH_IN * DIM)))[0]
        test_image = np.delete(test_image, 0, 0)
        test_image = np.append(test_image, np.reshape(test_output[i], (1, DIM)), axis=0)
        

    output_dat = '/pds/pds151/ckj/MLKM/MLKM_Tracer_5/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/output_same_{offset:d}_{length:d}_{trial:d}.dat'
    np.savetxt(output_dat.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, offset = offset, length = length_test, trial = TRIAL), test_output, fmt = "%e")

    fig = plt.figure(figsize=(10,6))
    plt.subplot(3,1,1)
    plt.xlabel('t')
    plt.ylabel('x,y')
    plt.pcolor(data[offset:offset + LENGTH_IN + length_test].T, vmin = -1, vmax = 1, cmap='Spectral')
    plt.colorbar()
    plt.tight_layout


    plt.subplot(3,1,2)
    plt.xlabel('t')
    plt.ylabel('x,y')
    plt.pcolor(np.append(data[offset:offset + LENGTH_IN], test_output, axis = 0).T, vmin = -1, vmax = 1, cmap='Spectral')
    plt.colorbar()
    plt.tight_layout


    plt.subplot(3,1,3)
    plt.xlabel('t')
    plt.ylabel('x,y')
    plt.pcolor(data[offset:offset + LENGTH_IN + length_test].T - np.append(data[offset:offset + LENGTH_IN], test_output, axis = 0).T, vmin = -2, vmax = 2, cmap='Spectral')
    plt.colorbar()
    plt.tight_layout

    output_png = '/pds/pds151/ckj/MLKM/MLKM_Tracer_5/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/output_same_{offset:d}_{length:d}_{trial:d}.png'
    plt.savefig(output_png.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, offset = offset, length = length_test, trial = TRIAL))


def test_diff_init(offset, length_test):

    data = np.loadtxt(file.format(N = SYSTEM_SIZE, K = COUPLING, w = SHIFT, ft = FEET, dt = DT, e = TIME_SERIES_TEST))
    test_output = np.zeros((length_test, DIM), dtype=np.float32)

    test_image = data[offset:offset + LENGTH_IN]
    for i in np.arange(length_test):
        if MACHINE == 'GPUs':
            test_dataset = tf.data.Dataset.from_tensor_slices(np.reshape(test_image, (1, LENGTH_IN * DIM)))
            test_dataset = test_dataset.repeat()
            test_dataset = test_dataset.batch(strategy.num_replicas_in_sync, drop_remainder = True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
            test_output[i] = model.predict(test_dataset, steps = 1)[0]
        elif MACHINE == 'CPU' or MACHINE == 'GPU':
            test_output[i] = model.predict(np.reshape(test_image, (1, LENGTH_IN * DIM)))[0]
        test_image = np.delete(test_image, 0, 0)
        test_image = np.append(test_image, np.reshape(test_output[i], (1, DIM)), axis=0)

    output_dat = '/pds/pds151/ckj/MLKM/MLKM_Tracer_5/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/output_diff_{offset:d}_{length:d}_{trial:d}.dat'
    np.savetxt(output_dat.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, offset = offset, length = length_test, trial = TRIAL), test_output, fmt = "%e")

    fig = plt.figure(figsize=(10,6))
    plt.subplot(3,1,1)
    plt.xlabel('t')
    plt.ylabel('x,y')
    plt.pcolor(data[offset:offset + LENGTH_IN + length_test].T, vmin = -1, vmax = 1, cmap='Spectral')
    plt.colorbar()
    plt.tight_layout


    plt.subplot(3,1,2)
    plt.xlabel('t')
    plt.ylabel('x,y')
    plt.pcolor(np.append(data[offset:offset + LENGTH_IN], test_output, axis = 0).T, vmin = -1, vmax = 1, cmap='Spectral')
    plt.colorbar()
    plt.tight_layout


    plt.subplot(3,1,3)
    plt.xlabel('t')
    plt.ylabel('x,y')
    plt.pcolor(data[offset:offset + LENGTH_IN + length_test].T - np.append(data[offset:offset + LENGTH_IN], test_output, axis = 0).T, vmin = -2, vmax = 2, cmap='Spectral')
    plt.colorbar()
    plt.tight_layout

    output_png = '/pds/pds151/ckj/MLKM/MLKM_Tracer_5/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/output_diff_{offset:d}_{length:d}_{trial:d}.png'
    plt.savefig(output_png.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, offset = offset, length = length_test, trial = TRIAL))





# test_same_init(0, LENGTH_TEST)
test_same_init(100000, LENGTH_TEST)
test_same_init(190000, LENGTH_TEST)
# test_diff_init(0, LENGTH_TEST)
# test_diff_init(190000, LENGTH_TEST)

