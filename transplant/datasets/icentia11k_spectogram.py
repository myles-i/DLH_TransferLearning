import numpy as np
from scipy.signal import ShortTimeFFT
from transplant.datasets.icentia11k import *
import tensorflow as tf


# Lets define the data generator
def _str(s):
    return s.decode() if isinstance(s, bytes) else str(s)

def spectogram_preprocessor(signal, frame_size=2048, window_size=256, stride=64, n_freqs = -1, fs=250.):
    # this function will take first "frame_size" elements in the input signal to compute the spectogram
    # and return a 3D tensor of shape (n_freqs, N, 1) where N = frame_size//stride
    # for simplicity, frame_size and window_size need to both be divisible by stride
    # will have some spectogram slices that contain zero padded data on the edges in order
    # to make the output exactly match the desired shape. if n_freqs = -1, it will return all
    # of teh computed frequencies which will be of length window_size//2

    # enforce that window_size is divisible by stride for simplicity
    # print all inputs
    assert frame_size % stride == 0, "frame_size must be divisible by stride"
    assert window_size % stride == 0, "window_size must be divisible by stride"

    if (len(signal) > frame_size):
        signal = signal[:frame_size]  # truncate signal to frame_size
    signal = np.squeeze(signal)

    if n_freqs == -1:
        n_freqs = window_size//2

    # calculate output size
    N = frame_size//stride

    # now figure out how many padded frames we needed
    # preilintialize the ShortTimeFFT object and related values
    w = np.hanning(window_size)
    SFT = ShortTimeFFT(w, hop=int(stride), fs=fs);


    # do complicated logic to fight left and right borders of non-padded spectogram slices (plb and pub)
    # then add sufficient padded slices on left and right such that the output length is correctly N
    plb = SFT.lower_border_end[1]
    pub = SFT.upper_border_begin(frame_size)[1] + 1

    padded_slices_needed = (N - (pub - plb))
    N_left_0padded_slices = padded_slices_needed//2
    N_right_0padded_slices = padded_slices_needed - N_left_0padded_slices
    lower_lim = plb - N_left_0padded_slices
    upper_lim = pub + N_right_0padded_slices

    # lets do the actual spectogram (series of fft frequency decompositions over time)
    x = SFT.stft(signal)

    # assume we want the output to be Nxn_freqs and we want the lower frequency
    x = x[ 1:(n_freqs+1), lower_lim:upper_lim] # we skip the first frequency component, as it is the DC component

    # now lets take the abs and normalize to between 0 and 1
    x = np.abs(x)
    x = x / np.max(x)

    x = np.expand_dims(x, axis=2)  # add channel dimension
    return x

def spectogram_beat_generator(patient_generator, samples_per_patient=1, frame_size=2048, window_size=256, stride = 64, n_freqs=-1):

    for _, (signal, labels) in patient_generator:
        num_segments, segment_size = signal.shape
        patient_beat_labels = labels['btype']  # note: variables in a .npz file are only loaded when accessed
        for _ in range(samples_per_patient):

            # randomly choose a frame that lies within the segment i.e. no zero-padding is necessary
            segment_index = np.random.randint(num_segments)
            frame_start = np.random.randint(segment_size - frame_size)
            frame_end = frame_start + frame_size
            raw_data = signal[segment_index, frame_start:frame_end]

            x = spectogram_preprocessor(raw_data, frame_size=frame_size, window_size = window_size, stride = stride, n_freqs = n_freqs)
            # calculate the count of each beat type in the frame and determine the final label
            beat_ends, beat_labels = patient_beat_labels[segment_index]
            _, frame_beat_labels = get_complete_beats(
                beat_ends, beat_labels, frame_start, frame_end)
            y = get_beat_label(frame_beat_labels)
            yield x, y

def spectogram_beat_data_generator(db_dir, patient_ids, frame_size,
                   unzipped=False, samples_per_patient=1, window_size=256, stride = 64, n_freqs=-1, ):

    patient_generator = uniform_patient_generator(
        _str(db_dir), patient_ids, unzipped=bool(unzipped))
    data_generator = spectogram_beat_generator(
        patient_generator, frame_size=int(frame_size), window_size = window_size, samples_per_patient=int(samples_per_patient),
        stride = int(stride), n_freqs = int(n_freqs))
    return data_generator

def spectogram_beat_dataset(db_dir, patient_ids, frame_size,
                   unzipped=False, samples_per_patient=1, window_size = 256, stride = 64, n_freqs=-1):
    
    #expected output size
    time_dim = frame_size//stride
    freq_dim = window_size//2 if n_freqs == -1 else n_freqs
    dataset = tf.data.Dataset.from_generator(
        generator=spectogram_beat_data_generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape((freq_dim, time_dim, 1)), tf.TensorShape(())),
        args=(db_dir, patient_ids, frame_size, unzipped, samples_per_patient, window_size, stride, n_freqs))
    return dataset