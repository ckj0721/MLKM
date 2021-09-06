#!/bin/python

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
SHIFT = float(sys.argv[7])
FEET = int(sys.argv[8])
DT = float(sys.argv[9])
MODEL = str(sys.argv[10])
LABEL = int(sys.argv[11])
DATA = int(sys.argv[12])
MAX_EPOCH = int(sys.argv[13])
PREV_TRIAL = int(sys.argv[14])
TRIAL = int(sys.argv[15])
TRAIN_LABEL_INI = int(sys.argv[16])
TRAIN_LABEL_END = int(sys.argv[17])

# CONSTANT
# REGULARIZATION_RATE = LEARNING_RATE
LENGTH_IN = int(FEET / DT)
DIM_IN = int(SYSTEM_SIZE * 2)
DIM_OUT = int(SYSTEM_SIZE * SYSTEM_SIZE)
ENSEMBLE_SIZE = 1000
TEST_LABEL = 0
VALID_LABEL = 2001


def ann(model, label):

    if model == "FFF":
        input = tf.keras.Input(shape = (LENGTH_IN * DIM_IN,), dtype = tf.float32)
        x = tf.keras.layers.Dense(DIM_OUT * label, activation = tf.keras.activations.tanh)(input)
        x = tf.keras.layers.Dense(DIM_OUT * label, activation = tf.keras.activations.tanh)(x)
        output = tf.keras.layers.Dense(DIM_OUT, activation = tf.keras.activations.sigmoid)(x)

    elif model == "1F":
        input = tf.keras.Input(shape = (LENGTH_IN * DIM_IN,), dtype = tf.float32)
        x = tf.keras.layers.Reshape((LENGTH_IN, DIM_IN,), input_shape=(LENGTH_IN * DIM_IN,))(input)
        x = tf.keras.layers.Conv1D(DIM_IN * label, 6, activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.Flatten()(x)
        output = tf.keras.layers.Dense(DIM_OUT, activation = tf.keras.activations.sigmoid)(x)

    elif model == "2F":
        input = tf.keras.Input(shape = (LENGTH_IN * DIM_IN,), dtype = tf.float32)
        x = tf.keras.layers.Reshape((LENGTH_IN, SYSTEM_SIZE, 2,), input_shape=(LENGTH_IN * DIM_IN,))(input)
        x = tf.keras.layers.Conv2D(2 * label, (6, 3), activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.Flatten()(x)
        output = tf.keras.layers.Dense(DIM_OUT, activation = tf.keras.activations.sigmoid)(x)
        
    elif model == "LF":
        input = tf.keras.Input(shape = (LENGTH_IN * DIM_IN,), dtype = tf.float32)
        x = tf.keras.layers.Reshape((LENGTH_IN, DIM_IN,), input_shape = (LENGTH_IN * DIM_IN,))(input)
        x = tf.keras.layers.LSTM(DIM_OUT * label)(x)
        output = tf.keras.layers.Dense(DIM_OUT, activation = tf.keras.activations.sigmoid)(x)

    elif model == "RF":
        input = tf.keras.Input(shape = (LENGTH_IN * DIM_IN,), dtype = tf.float32)
        x = tf.keras.layers.Reshape((LENGTH_IN, DIM_IN,), input_shape = (LENGTH_IN * DIM_IN,))(input)
        x = tf.keras.layers.SimpleRNN(DIM_OUT * label)(x)
        output = tf.keras.layers.Dense(DIM_OUT, activation = tf.keras.activations.sigmoid)(x)

#     elif model == "LLF":
#         input = tf.keras.Input(shape = (LENGTH_IN * DIM_IN,), dtype = tf.float32)
#         x = tf.keras.layers.Reshape((LENGTH_IN, DIM_IN,), input_shape = (LENGTH_IN * DIM_IN,))(input)
#         x = tf.keras.layers.LSTM(SYSTEM_SIZE * label, return_sequences = True)(x)
#         x = tf.keras.layers.LSTM(DIM_OUT)(x)
#         output = tf.keras.layers.Dense(DIM_OUT, activation = tf.keras.activations.sigmoid)(x)
    
    model = tf.keras.Model(input, output)

    return model

save = '/pds/pds151/ckj/MLKM/MLKM_Detector_6/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/save_{trial:d}/save.ckpt'

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
        model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE), loss = tf.keras.losses.MSE)
elif MACHINE == 'CPU' or MACHINE == 'GPU':
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICES
    BATCH_SIZE = BATCHING
    SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 4
    THREAD_DATA = 2
    model = ann(MODEL, LABEL)
    model.load_weights(save.format(N = SYSTEM_SIZE, L = LENGTH_IN, K = COUPLING, model = MODEL, label = LABEL, trial = PREV_TRIAL))
    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE), loss = tf.keras.losses.MSE)
else:
    print("Select MACHINE : CPUs/GPUs/GPU")
    raise SystemExit





# file_image = '/pds/pds151/ckj/MLKM/Phase_to_Net/N{N:d}_2/Phaset_N{N:d}_K{K:0.4f}_ft{ft:d}_dt{dt:0.3f}_{E:d}.dat'
# file_image = '/pds/pds112/ckj/MLKM/Phase_to_Net/N{N:d}_2/Phaset_N{N:d}_K{K:0.4f}_ft{ft:d}_dt{dt:0.3f}_{E:d}.dat'
file_image = '/pds/pds{data:d}/ckj/MLKM/Phase_to_Net/N{N:d}_2/Phaset_N{N:d}_K{K:0.4f}_ft{ft:d}_dt{dt:0.3f}_{E:d}.dat'
def list_of_file_image(Emin, Emax):
    list = []
    for E in np.arange(Emin, Emax + 0.5):
        list.append(file_image.format(N = SYSTEM_SIZE, K = COUPLING, w = SHIFT, ft = FEET, dt = DT, E = int(E), data = DATA))
    return list

# file_label = '/pds/pds151/ckj/MLKM/Phase_to_Net/N{N:d}_2/network_N{N:d}_K{K:0.4f}_ft{ft:d}_dt{dt:0.3f}_{E:d}.dat'
# file_label = '/pds/pds112/ckj/MLKM/Phase_to_Net/N{N:d}_2/network_N{N:d}_K{K:0.4f}_ft{ft:d}_dt{dt:0.3f}_{E:d}.dat'
file_label = '/pds/pds{data:d}/ckj/MLKM/Phase_to_Net/N{N:d}_2/network_N{N:d}_K{K:0.4f}_ft{ft:d}_dt{dt:0.3f}_{E:d}.dat'
def list_of_file_label(Emin, Emax):
    list = []
    for E in np.arange(Emin, Emax + 0.5):
        list.append(file_label.format(N = SYSTEM_SIZE, K = COUPLING, w = SHIFT, ft = FEET, dt = DT, E = int(E), data = DATA))
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

dataset_train, dataset_valid, number_of_train, number_of_valid = make_dataset_train(ENSEMBLE_SIZE, TRAIN_LABEL_INI, TRAIN_LABEL_END)


# history = model.fit(dataset_train, verbose = 0, epochs = MAX_EPOCH, steps_per_epoch = int(number_of_train / BATCH_SIZE), validation_data = dataset_valid, validation_steps = int(number_of_valid / BATCH_SIZE))


earlystop_callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = MAX_EPOCH, restore_best_weights = True)]
history = model.fit(dataset_train, verbose = 0, epochs = int(10 * MAX_EPOCH), steps_per_epoch = int(number_of_train / BATCH_SIZE), validation_data = dataset_valid, validation_steps = int(number_of_valid / BATCH_SIZE), callbacks = earlystop_callbacks)



history_dat = '/pds/pds151/ckj/MLKM/MLKM_Detector_6/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/history_{trial:d}.dat'
history_pdf = '/pds/pds151/ckj/MLKM/MLKM_Detector_6/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/history_{trial:d}.pdf'


def plot_history(hist, filename):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(hist[:,0], label='Train')
    plt.plot(hist[:,1], label = 'Validation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)


if len(history.history['loss']) == int(10 * MAX_EPOCH):
    hist = np.append(np.loadtxt(history_dat.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, trial = PREV_TRIAL)), np.column_stack((history.history['loss'], history.history['val_loss'])), axis = 0)
else:
    hist = np.append(np.loadtxt(history_dat.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, trial = PREV_TRIAL)), np.column_stack((history.history['loss'], history.history['val_loss']))[:-MAX_EPOCH], axis = 0)

plot_history(hist, history_pdf.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, trial = TRIAL))
np.savetxt(history_dat.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, trial = TRIAL), hist, fmt = "%e")

save_dir = '/pds/pds151/ckj/MLKM/MLKM_Detector_6/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/save_{trial:d}/'
if not os.path.exists(save_dir.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, trial = TRIAL)):
    os.mkdir(save_dir.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, trial = TRIAL))

model.save_weights(save.format(N = SYSTEM_SIZE, L = LENGTH_IN, K = COUPLING, model = MODEL, label = LABEL, trial = TRIAL))



def make_dataset_test(size_of_ensemble, Emin, Emax):
    list_of_image = list_of_file_image(Emin, Emax)
    list_of_label = list_of_file_label(Emin, Emax)

    number_of_files = len(list_of_label)
    number_of_test = number_of_files * size_of_ensemble

#     labels = tf.data.Dataset.from_tensor_slices(list_of_label).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
#     images = tf.data.Dataset.from_tensor_slices(list_of_image).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    labels = tf.data.Dataset.from_tensor_slices(list_of_label).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = number_of_files)
    images = tf.data.Dataset.from_tensor_slices(list_of_image).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = number_of_files)
    dataset = tf.data.Dataset.zip((images, labels))

    # map and cache
    def decorder(image, label):
        image = tf.strings.split(image)
        image = tf.strings.to_number(image, out_type = tf.float32)
        label = tf.strings.split(label)
        label = tf.strings.to_number(label, out_type = tf.float32)
        return image, label
    
#     dataset = dataset.map(decorder, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(decorder, num_parallel_calls = THREAD_DATA).cache()

    # batch and prefetch
    dataset = dataset.batch(BATCH_SIZE, drop_remainder = True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
  
    return dataset, number_of_test


def test(target):
    dataset_test, number_of_test = make_dataset_test(ENSEMBLE_SIZE, target, target)
    outputs = model.predict(dataset_test, steps = int(number_of_test / BATCH_SIZE))
    outputs_dat = '/pds/pds151/ckj/MLKM/MLKM_Detector_6/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/output_{target:d}_{trial:d}.dat'
    np.savetxt(outputs_dat.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, target = target, trial = TRIAL), outputs, fmt = "%e")
    outputs_round = tf.math.round(outputs)
    outputs_round_dat = '/pds/pds151/ckj/MLKM/MLKM_Detector_6/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/output_{target:d}_{trial:d}_round.dat'
    np.savetxt(outputs_round_dat.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, target = target, trial = TRIAL), outputs_round, fmt = "%e")

    

# test(VALID_LABEL)
test(TEST_LABEL)


def plot_image(images, filename):
    plt.figure(figsize=(12, 8))
    plt.subplot(2,1,1)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.pcolor(np.reshape(images, (LENGTH_IN, DIM_IN)).T[::2], vmin = -1, vmax = 1, cmap='Spectral')
    plt.colorbar()
    plt.tight_layout

    plt.subplot(2,1,2)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.pcolor(np.reshape(images, (LENGTH_IN, DIM_IN)).T[1::2], vmin = -1, vmax = 1, cmap='Spectral')
    plt.colorbar()
    plt.tight_layout

    plt.savefig(filename)
    plt.close()

def plot_label(images, labels, outputs, outputs_round, filename):
    plt.figure(figsize=(12, 8))
    plt.subplot(2,3,1)
    plt.xlabel('')
    plt.ylabel('')
    plt.pcolor(np.reshape(labels, (SYSTEM_SIZE, SYSTEM_SIZE)), vmin = 0, vmax = 1, cmap='binary')
    plt.axis('square')
    plt.tick_params(axis='both', labelsize=0, length = 0)
    plt.tight_layout

    plt.subplot(2,3,2)
    plt.xlabel('')
    plt.ylabel('')
    plt.pcolor(np.reshape(outputs, (SYSTEM_SIZE, SYSTEM_SIZE)), vmin = 0, vmax = 1, cmap='binary')
    plt.axis('square')
    plt.tick_params(axis='both', labelsize=0, length = 0)
    plt.tight_layout

    plt.subplot(2,3,3)
    plt.xlabel('')
    plt.ylabel('')
    plt.pcolor(np.reshape(labels - outputs, (SYSTEM_SIZE, SYSTEM_SIZE)), vmin = -1, vmax = 1, cmap='seismic')
    plt.axis('square')
    plt.tick_params(axis='both', labelsize=0, length = 0)
    plt.tight_layout

    plt.subplot(2,3,4)
    plt.xlabel('')
    plt.ylabel('')
    plt.pcolor(np.reshape(labels, (SYSTEM_SIZE, SYSTEM_SIZE)), vmin = 0, vmax = 1, cmap='binary')
    plt.axis('square')
    plt.tick_params(axis='both', labelsize=0, length = 0)
    plt.tight_layout

    plt.subplot(2,3,5)
    plt.xlabel('')
    plt.ylabel('')
    plt.pcolor(np.reshape(outputs_round, (SYSTEM_SIZE, SYSTEM_SIZE)), vmin = 0, vmax = 1, cmap='binary')
    plt.axis('square')
    plt.tick_params(axis='both', labelsize=0, length = 0)
    plt.tight_layout

    plt.subplot(2,3,6)
    plt.xlabel('')
    plt.ylabel('')
    plt.pcolor(np.reshape(labels - outputs_round, (SYSTEM_SIZE, SYSTEM_SIZE)), vmin = -1, vmax = 1, cmap='seismic')
    plt.axis('square')
    plt.tick_params(axis='both', labelsize=0, length = 0)
    plt.tight_layout
    plt.savefig(filename)
    plt.close()

    

def plot_sample(target, number_of_sample):
    images = np.loadtxt(file_image.format(N = SYSTEM_SIZE, K = COUPLING, w = SHIFT, ft = FEET, dt = DT, E = int(target), data = DATA))
    labels = np.loadtxt(file_label.format(N = SYSTEM_SIZE, K = COUPLING, w = SHIFT, ft = FEET, dt = DT, E = int(target), data = DATA))
    outputs_dat = '/pds/pds151/ckj/MLKM/MLKM_Detector_6/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/output_{target:d}_{trial:d}.dat'
    outputs = np.loadtxt(outputs_dat.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, target = target, trial = TRIAL))
    outputs_round_dat = '/pds/pds151/ckj/MLKM/MLKM_Detector_6/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/output_{target:d}_{trial:d}_round.dat'
    outputs_round = np.loadtxt(outputs_round_dat.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, target = target, trial = TRIAL))

    fig_dir = '/pds/pds151/ckj/MLKM/MLKM_Detector_6/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/output_{target:d}_fig_{trial:d}/'
    if not os.path.exists(fig_dir.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, target = target, trial = TRIAL)):
        os.mkdir(fig_dir.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, target = target, trial = TRIAL))

    fig_image = '/pds/pds151/ckj/MLKM/MLKM_Detector_6/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/output_{target:d}_fig_{trial:d}/sample_{i:d}_image.png'
    fig_label = '/pds/pds151/ckj/MLKM/MLKM_Detector_6/saves/N{N:d}K{K:0.4f}L{L:d}/{model:s}_{label:d}/output_{target:d}_fig_{trial:d}/sample_{i:d}_label.png'

    for i in range(number_of_sample):
        plot_image(images[i], fig_image.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, target = target, trial = TRIAL, i = i))
        plot_label(images[i], labels[i], outputs[i], outputs_round[i], fig_label.format(N = SYSTEM_SIZE, K = COUPLING, L = LENGTH_IN, model = MODEL, label = LABEL, target = target, trial = TRIAL, i = i))

            

# plot_sample(VALID_LABEL, 100)
plot_sample(TEST_LABEL, 100)
