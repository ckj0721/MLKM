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
LENGTH_OF_TIME = int(sys.argv[5])
MODEL = str(sys.argv[6])
LABEL = int(sys.argv[7])
MAX_EPOCH = int(sys.argv[8])
PREV_TRIAL = int(sys.argv[9])
TRIAL = int(sys.argv[10])

# REGULARIZATION_RATE = LEARNING_RATE
ENSEMBLE_SIZE = 10000
KMIN = 0.0200
KMAX = 2.0000
TRAIN_INI = 1
TRAIN_FIN = 2


def ann(model, label):

    if model == 'FFF':
        input = tf.keras.Input(shape = (LENGTH_OF_TIME,), dtype = tf.float32)
        x = tf.keras.layers.Dense(64 * label, activation = tf.keras.activations.tanh)(input)
        x = tf.keras.layers.Dense(64 * label, activation = tf.keras.activations.tanh)(x)
        output = tf.keras.layers.Dense(1, activation = tf.keras.activations.relu)(x)

    elif model == 'FFFF':
        input = tf.keras.Input(shape = (LENGTH_OF_TIME,), dtype = tf.float32)
        x = tf.keras.layers.Dense(64 * label, activation = tf.keras.activations.tanh)(input)
        x = tf.keras.layers.Dense(64 * label, activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.Dense(64 * label, activation = tf.keras.activations.tanh)(x)
        output = tf.keras.layers.Dense(1, activation=tf.keras.activations.relu)(x)

    elif model == '11FFF':
        input = tf.keras.Input(shape = (LENGTH_OF_TIME,), dtype = tf.float32)
        x = tf.keras.layers.Reshape((LENGTH_OF_TIME, 1), input_shape=(LENGTH_OF_TIME,))(input)
        x = tf.keras.layers.Conv1D(2 * label, 3, activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.Conv1D(2 * label, 3, activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64 * label, activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.Dense(64 * label, activation = tf.keras.activations.tanh)(x)
        output = tf.keras.layers.Dense(1, activation=tf.keras.activations.relu)(x)

    elif model == 'LLF':
        input = tf.keras.Input(shape = (LENGTH_OF_TIME,), dtype = tf.float32)
        x = tf.keras.layers.Reshape((LENGTH_OF_TIME, 1), input_shape=(LENGTH_OF_TIME,))(input)
        x = tf.keras.layers.LSTM(64 * label, return_sequences = True)(x)
        x = tf.keras.layers.LSTM(64 * label)(x)
        output = tf.keras.layers.Dense(1, activation=tf.keras.activations.relu)(x)

    model = tf.keras.Model(input, output)
    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE), loss = tf.keras.losses.MAE)

    return model

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
elif MACHINE == 'CPU' or MACHINE == 'GPU':
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICES
    BATCH_SIZE = BATCHING
    SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 4
    THREAD_DATA = 2
    model = ann(MODEL, LABEL)
else:
    print("Select MACHINE : CPUs/GPUs/GPU")
    raise SystemExit

save = '/pds/pds151/ckj/MLKM/MLKM_Estimator_4/saves/L{L:d}/{model:s}_{label:d}/save_{trial:d}/save.ckpt'
model.load_weights(save.format(L = LENGTH_OF_TIME, model = MODEL, label = LABEL, trial = PREV_TRIAL))


file_image = '/pds/pds121/KMML/train3/Rt_K{K:0.4f}_e{e:d}.dat'
def list_of_file_image(Kmin, Kmax, emin, emax):
    list = []
    for e in np.arange(emin, emax + 0.5):
        for K in np.arange(Kmin, Kmax + 0.0100, 0.0200):
            list.append(file_image.format(K = K, e = int(e)))
    return list

file_label = '/pds/pds121/ckj/KMML/train3/Rt_K{K:0.4f}_e{e:d}.label'
def list_of_file_label(Kmin, Kmax, emin, emax):
    list = []
    for e in np.arange(emin, emax + 0.5):
        for K in np.arange(Kmin, Kmax + 0.0100, 0.0200):
            list.append(file_label.format(K = K, e = int(e)))
    return list

test_image = '/pds/pds121/KMML/test3/Rt_test.dat'

def make_dataset_test():
    return np.loadtxt(test_image)

def make_dataset_train(size_of_ensemble, Kmin, Kmax, emin, emax):
    list_of_image = list_of_file_image(Kmin, Kmax, emin, emax)
    list_of_label = list_of_file_label(Kmin, Kmax, emin, emax)

    number_of_files = len(list_of_label)
    number_of_total = number_of_files * size_of_ensemble

    # labels = tf.data.Dataset.from_tensor_slices(list_of_label).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    # images = tf.data.Dataset.from_tensor_slices(list_of_image).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    labels = tf.data.Dataset.from_tensor_slices(list_of_label).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = THREAD_DATA)
    images = tf.data.Dataset.from_tensor_slices(list_of_image).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = THREAD_DATA)
    dataset = tf.data.Dataset.zip((images, labels))

    # map and cache
    @tf.function
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




dataset_train, dataset_valid, number_of_train, number_of_valid = make_dataset_train(ENSEMBLE_SIZE, KMIN, KMAX, TRAIN_INI, TRAIN_FIN)


history = model.fit(dataset_train, verbose = 0, epochs = MAX_EPOCH, steps_per_epoch = int(number_of_train / BATCH_SIZE), validation_data = dataset_valid, validation_steps = int(number_of_valid / BATCH_SIZE))

earlystop_callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = MAX_EPOCH, restore_best_weights = True)]
# earlystop_callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = MAX_EPOCH)]
history = model.fit(dataset_train, verbose = 0, epochs = int(10 * MAX_EPOCH), steps_per_epoch = int(number_of_train / BATCH_SIZE), validation_data = dataset_valid, validation_steps = int(number_of_valid / BATCH_SIZE), callbacks = earlystop_callbacks)



history_pdf = '/pds/pds151/ckj/MLKM/MLKM_Estimator_4/saves/L{L:d}/{model:s}_{label:d}/history_{trial:d}.pdf'
history_dat = '/pds/pds151/ckj/MLKM/MLKM_Estimator_4/saves/L{L:d}/{model:s}_{label:d}/history_{trial:d}.dat'

def plot_history(hist, filename):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(hist[:,0], label='Train')
    plt.plot(hist[:,1], label = 'Validation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)

# hist = np.append(np.loadtxt(history_dat.format(L = LENGTH_OF_TIME, model = MODEL, label = LABEL, trial = PREV_TRIAL)), np.column_stack((history.history['loss'], history.history['val_loss'])), axis = 0)
hist = np.append(np.loadtxt(history_dat.format(L = LENGTH_OF_TIME, model = MODEL, label = LABEL, trial = PREV_TRIAL)), np.column_stack((history.history['loss'], history.history['val_loss']))[:-MAX_EPOCH], axis = 0)

plot_history(hist, history_pdf.format(L = LENGTH_OF_TIME, model = MODEL, label = LABEL, trial = TRIAL))
np.savetxt(history_dat.format(L = LENGTH_OF_TIME, model = MODEL, label = LABEL, trial = TRIAL), hist, fmt = "%f")


save_dir = '/pds/pds151/ckj/MLKM/MLKM_Estimator_4/saves/L{L:d}/{model:s}_{label:d}/save_{trial:d}/'
if not os.path.exists(save_dir.format(L = LENGTH_OF_TIME, model = MODEL, label = LABEL, trial = TRIAL)):
    os.mkdir(save_dir.format(L = LENGTH_OF_TIME, model = MODEL, label = LABEL, trial = TRIAL))
model.save_weights(save.format(L = LENGTH_OF_TIME, model = MODEL, label = LABEL, trial = TRIAL))



dataset_test = make_dataset_test()
outputs_dat = '/pds/pds151/ckj/MLKM/MLKM_Estimator_4/saves/L{L:d}/{model:s}_{label:d}/outputs_{trial:d}.dat'
outputs = model.predict(dataset_test)
np.savetxt(outputs_dat.format(L = LENGTH_OF_TIME, model = MODEL, label = LABEL, trial = TRIAL), outputs, fmt = "%f")

