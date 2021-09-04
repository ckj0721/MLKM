#!/bin/python

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


# ARGV
MACHINE = str(sys.argv[1])
BATCHING = int(sys.argv[2])
LENGTH = int(sys.argv[3])
KC = 1.0000 #float(sys.argv[5])
KC_MIN = float(sys.argv[4])
KC_MAX = float(sys.argv[5])
LABEL = int(sys.argv[6])

dir_root = '/pds/pds151/ckj/MLKM/MLKM_Discriminator/saves/'
dir_save = '/pds/pds151/ckj/MLKM/MLKM_Discriminator/saves/N{N:d}_{Kcmin:0.4f}_{Kcmax:0.4f}/'

if not os.path.exists(dir_root):
    os.mkdir(dir_root)
if not os.path.exists(dir_save.format(N = LENGTH, Kcmin = KC_MIN, Kcmax = KC_MAX)):
    os.mkdir(dir_save.format(N = LENGTH, Kcmin = KC_MIN, Kcmax = KC_MAX))

def ann(length, label):
    input = tf.keras.Input(shape = (length,), dtype = tf.float32)
    # x = tf.keras.layers.Reshape((length, 1), input_shape=(length,))(input)

    if label == 1:
        x = tf.math.cos(input)
        y = tf.math.sin(input)
        z = tf.keras.layers.Concatenate(axis = 1)([x, y])
        x = tf.keras.layers.Dense(256, activation = tf.keras.activations.tanh)(z)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation = tf.keras.activations.tanh)(x)
        x = tf.keras.layers.BatchNormalization()(x)
    # elif label == 2:
    #     x = tf.keras.layers.LSTM(256)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    #     x = tf.keras.layers.Dense(256, activation = tf.keras.activations.tanh)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    #     x = tf.keras.layers.Dense(256, activation = tf.keras.activations.tanh)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    # elif label == 3:
    #     x = tf.keras.layers.GRU(256)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    #     x = tf.keras.layers.Dense(256, activation = tf.keras.activations.tanh)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    #     x = tf.keras.layers.Dense(256, activation = tf.keras.activations.tanh)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    # elif label == 4:
    #     x = tf.keras.layers.SimpleRNN(256)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    #     x = tf.keras.layers.Dense(256, activation = tf.keras.activations.tanh)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    #     x = tf.keras.layers.Dense(256, activation = tf.keras.activations.tanh)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    # elif label == 5:
    #     x = tf.keras.layers.Conv1D(16, 8, activation = tf.keras.activations.tanh)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    #     x = tf.keras.layers.LSTM(256)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    #     x = tf.keras.layers.Dense(256, activation = tf.keras.activations.tanh)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    #     x = tf.keras.layers.Dense(256, activation = tf.keras.activations.tanh)(x)
    #     x = tf.keras.layers.BatchNormalization()(x)


    output = tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)(x)
        
    model = tf.keras.Model(input, output)
    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-6), loss = tf.keras.losses.binary_crossentropy, metrics = ['accuracy'])
    return model

#parameters depends on machine
if MACHINE == 'GPUs':
    strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE_REP = BATCHING
    BATCH_SIZE = BATCH_SIZE_REP * strategy.num_replicas_in_sync
    SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 8
    def make_model(length, label):
        with strategy.scope():
            return ann(length, label)
elif MACHINE == 'CPUs':
    BATCH_SIZE = BATCHING
    SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 8
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    def make_model(length, label):
        return ann(length, label)
elif MACHINE == 'GPU':
    BATCH_SIZE = BATCHING
    SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 8
    def make_model(length, label):
        return ann(length, label)
else:
    print("Select MACHINE : CPUs/GPUs/GPU")
    raise SystemExit

file_image = '/pds/pds121/KMML/phase_snapshot/N{N:d}/Phase_snapshot_N{N:d}_K{K:0.4f}_{e:d}.dat'
def list_of_file_image(Kmin, Kmax, emin, emax):
    list = []
    for e in np.arange(emin, emax + 0.5):
        for K in np.arange(Kmin, Kmax + 0.0050, 0.0100):
            list.append(file_image.format(N = LENGTH, K = K, e = int(e)))
    return list

file_label = '/pds/pds121/ckj/KMML/phase_snapshot/Phase_snapshot_K{K:0.4f}_{e:d}.label'
def list_of_file_label(Kmin, Kmax, emin, emax):
    list = []
    for e in np.arange(emin, emax + 0.5):
        for K in np.arange(Kmin, Kmax + 0.0050, 0.0100):
            list.append(file_label.format(K = K, e = int(e)))
    return list



def make_dataset_test(size_of_ensemble, Kc, Kmin, Kmax, emin, emax):
    list_of_image = list_of_file_image(Kmin, Kmax, emin, emax)
    list_of_label = list_of_file_label(Kmin, Kmax, emin, emax)

    number_of_files = len(list_of_label)
    number_of_total = number_of_files * size_of_ensemble

    labels = tf.data.Dataset.from_tensor_slices(list_of_label).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    images = tf.data.Dataset.from_tensor_slices(list_of_image).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)


    # shuffle and repeat 
    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).repeat()

    # map and cache
    # @tf.function
    def decorder(image, label):
        image = tf.strings.split(image)
        image = tf.strings.to_number(image, out_type = tf.float32)
        label = tf.strings.split(label)
        label = tf.strings.to_number(label, out_type = tf.float32)
        def up(): return tf.constant([1, 0], dtype = tf.int64)
        def down(): return tf.constant([0, 1], dtype = tf.int64)
        label = tf.cond(label[0] > Kc, up, down)
        return image, label
    
    dataset = dataset.map(decorder, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    # batch and prefetch
    dataset = dataset.batch(BATCH_SIZE, drop_remainder = True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    
    return dataset, number_of_total

def make_dataset_train(size_of_ensemble, Kc, Kcmin, Kcmax, emin, emax):
    list_of_image = []
    list_of_label = []
    list_of_image += list_of_file_image(0.0100, Kcmin, emin, emax)
    list_of_image += list_of_file_image(Kcmax, 2.2000, emin, emax)
    list_of_label += list_of_file_label(0.0100, Kcmin, emin, emax)
    list_of_label += list_of_file_label(Kcmax, 2.2000, emin, emax)

    number_of_files = len(list_of_label)
    number_of_total = number_of_files * size_of_ensemble

    labels = tf.data.Dataset.from_tensor_slices(list_of_label).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    images = tf.data.Dataset.from_tensor_slices(list_of_image).interleave(tf.data.TextLineDataset, cycle_length = number_of_files, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    dataset = tf.data.Dataset.zip((images, labels))

    # split training and validation
    number_of_train = int(number_of_total * 0.9)
    number_of_valid = number_of_total - number_of_train
    dataset_valid = dataset.take(number_of_valid)
    dataset_train = dataset.skip(number_of_valid)
    
    # repeat and shuffle
    dataset_train = dataset_train.shuffle(SHUFFLE_BUFFER_SIZE).repeat()
    dataset_valid = dataset_valid.shuffle(SHUFFLE_BUFFER_SIZE).repeat()

    # map and cache
    # @tf.function
    def decorder(image, label):
        image = tf.strings.split(image)
        image = tf.strings.to_number(image, out_type = tf.float32)
        label = tf.strings.split(label)
        label = tf.strings.to_number(label, out_type = tf.float32)
        def up(): return tf.constant([1, 0], dtype = tf.int64)
        def down(): return tf.constant([0, 1], dtype = tf.int64)
        label = tf.cond(label[0] > Kc, up, down)
        return image, label
    
    dataset_train = dataset_train.map(decorder, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    dataset_valid = dataset_valid.map(decorder, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    # batch and prefetch
    dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder = True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    dataset_valid = dataset_valid.batch(BATCH_SIZE, drop_remainder = True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    
    return dataset_train, dataset_valid, number_of_train, number_of_valid

dataset_train, dataset_valid, number_of_train, number_of_valid = make_dataset_train(10000, KC, KC_MIN, KC_MAX, 1, 2)

save = '/pds/pds151/ckj/MLKM/MLKM_Discriminator/saves/N{N:d}_{Kcmin:0.4f}_{Kcmax:0.4f}/Test_dis_{label:d}_save'
if not os.path.exists(save.format(N = LENGTH, Kcmin = KC_MIN, Kcmax = KC_MAX, label = LABEL)):
    os.mkdir(save.format(N = LENGTH, Kcmin = KC_MIN, Kcmax = KC_MAX, label = LABEL))
struct = '/pds/pds151/ckj/MLKM/MLKM_Discriminator/saves/N{N:d}_{Kcmin:0.4f}_{Kcmax:0.4f}/Test_dis_{label:d}_struct.png'
train_history = '/pds/pds151/ckj/MLKM/MLKM_Discriminator/saves/N{N:d}_{Kcmin:0.4f}_{Kcmax:0.4f}/Test_dis_{label:d}_history.pdf'

def plot_history(history, filename):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.clf()
    plt.subplot(2, 1, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'], label='Train')
    plt.plot(hist['epoch'], hist['val_accuracy'], label = 'Validation')
    plt.legend()
    plt.show()

    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train')
    plt.plot(hist['epoch'], hist['val_loss'], label = 'Validation')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)

    
earlystop_callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)]

model = make_model(LENGTH, LABEL)

tf.keras.utils.plot_model(model, to_file = struct.format(N = LENGTH, Kcmin = KC_MIN, Kcmax = KC_MAX, label = LABEL), show_shapes = True, show_layer_names = False, rankdir = 'LR')

history = model.fit(dataset_train, verbose = 0, epochs = 100, steps_per_epoch = int(number_of_train / BATCH_SIZE), validation_data = dataset_valid, validation_steps = int(number_of_valid / BATCH_SIZE), callbacks = earlystop_callbacks)
model.save(save.format(N = LENGTH, Kcmin = KC_MIN, Kcmax = KC_MAX, label = LABEL))
plot_history(history, train_history.format(N = LENGTH, Kcmin = KC_MIN, Kcmax = KC_MAX, label = LABEL))

outputs = np.empty((220, 3))
for i in range(220):
    K = 0.0100 + 0.0100 * i
    dataset, number_of_total = make_dataset_test(10000, KC, K, K, 1, 2)
    output = tf.reduce_mean(model.predict(dataset, steps = int(number_of_total / BATCH_SIZE)), 0)
    outputs[i, 0] = K
    outputs[i, 1] = output.numpy()[0]
    outputs[i, 2] = output.numpy()[1]


dat = '/pds/pds151/ckj/MLKM/MLKM_Discriminator/saves/N{N:d}_{Kcmin:0.4f}_{Kcmax:0.4f}/Test_{label:d}_data.dat'
np.savetxt(dat.format(N = LENGTH, Kcmin = KC_MIN, Kcmax = KC_MAX, label = LABEL), outputs, fmt = "%f")

def plot_outputs(outputs, Kc, Kcmin, Kcmax, filename):
    plt.clf()
    plt.xlabel('K')
    plt.ylabel('output')
    plt.axvspan(0.0100, Kcmin, color = 'C0', alpha = 0.3)
    plt.axvspan(Kcmax, 2.2000, color = 'C3', alpha = 0.3)
    plt.axvline(x = Kc, color = 'C1', linestyle = ':')
    plt.axhline(y = 0.5, color = 'C2', linestyle = ':')
    plt.axvline(x = (KC_MIN + KC_MAX) * 0.5, color = 'C3', linestyle = ':')
    plt.plot(outputs[:,0], outputs[:,1], label = 'output 1', color = 'C4')
    plt.plot(outputs[:,0], outputs[:,2], label = 'output 2', color = 'C5')
    plt.legend()
    plt.savefig(filename)

fig = '/pds/pds151/ckj/MLKM/MLKM_Discriminator/saves/N{N:d}_{Kcmin:0.4f}_{Kcmax:0.4f}/Test_{label:d}_test.pdf'
plot_outputs(outputs, KC, KC_MIN, KC_MAX, fig.format(N = LENGTH, Kcmin = KC_MIN, Kcmax = KC_MAX, label = LABEL))
