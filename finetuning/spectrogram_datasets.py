import numpy as np
from transplant.datasets.icentia11k_spectogram import spectogram_preprocessor
import tensorflow as tf

def finetuning_spectogram_beat_generator(x_data, y_data, frame_size=2048, window_size=256, stride=64, n_freqs = 64, fs=250., shuffle = True):

    # initialize array of indices
    indices = np.arange(x_data.shape[0])
    while True:
        # data is a dictionary with data['x'] and data['y'] where data['x'] is a list of 1706 ECG signals of size (1706, 16384, 1)
        # lets loop through each of these getting x and y
        if shuffle:
            np.random.shuffle(indices)

        for i in indices: # loop through each sample
            x = np.squeeze(x_data[i])  # Get the ith sample from data['x']
            
            y = y_data[i]  # Get the ith label from data['y']
            x = spectogram_preprocessor(
                x,
                frame_size=frame_size,
                window_size=window_size,
                stride=stride,
                n_freqs=n_freqs,
                fs=fs
                )

            yield x, y
# Now lets define the fine-tuning dataloaders
def create_spectogram_dataset_from_data(data, frame_size=2048, window_size = 256, stride = 64,n_freqs = 64):
    time_dim = frame_size//stride
    freq_dim = window_size//2 if n_freqs == -1 else n_freqs
    print("time_dim: ", time_dim, "freq_dim: ", freq_dim)
    dataset = tf.data.Dataset.from_generator(
        generator=finetuning_spectogram_beat_generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape((freq_dim, time_dim, 1)), tf.TensorShape((4))),
        args=(data['x'], data['y'], frame_size, window_size, stride, n_freqs))
    return dataset