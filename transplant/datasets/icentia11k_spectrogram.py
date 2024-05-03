import numpy as np
from scipy.signal import ShortTimeFFT
from transplant.datasets.icentia11k import *
import tensorflow as tf
import librosa

# Lets define the data generator
def _str(s):
    return s.decode() if isinstance(s, bytes) else str(s)

def spectrogram_preprocessor(signal, frame_size=None, window_size=256, stride=64, n_freqs = -1, fs=250., output_db = False, ref = np.min, amin = 1e-5, top_db = 80):
    if not frame_size:
        frame_size = len(signal)
    n_slices = frame_size//stride
    # ch1 = librosa.feature.melspectrogram(n_mels = N,y=signal, n_fft = window_size, hop_length = stride, sr =fs)
    x =librosa.stft(signal, n_fft=window_size, hop_length=stride)
    x = np.abs(x) # take the magnitude, ignore the phase
    if output_db:
        # express in db so mag is in log scale
        x = librosa.amplitude_to_db(x, ref = ref, amin = amin, top_db = top_db) 
        # equivalent math:
        # https://librosa.org/doc/0.8.1/generated/librosa.amplitude_to_db.html
        # x = np.maximum(x, amin)
        # x = 20.0 * np.log10(x / ref(x)) #if ref is not a function, then do x/ref instead
        # x = np.maximum(x - np.max(x), -top_db) + np.max # threshold the output at top_db below the peak

    # x = 20*np.log10(x + 1e-6) # convert to decibels
    x = x[1:n_freqs+1, 0:n_slices]
    # x = x/np.max(x) # normalize to between 0 and 1
    x = np.expand_dims(x, axis=2)  # add channel dimension
    return x

def spectrogram_beat_generator(patient_generator, samples_per_patient=1, frame_size=2048, window_size=256, stride = 64, n_freqs=-1):

    for _, (signal, labels) in patient_generator:
        num_segments, segment_size = signal.shape
        patient_beat_labels = labels['btype']  # note: variables in a .npz file are only loaded when accessed
        for _ in range(samples_per_patient):

            # randomly choose a frame that lies within the segment i.e. no zero-padding is necessary
            segment_index = np.random.randint(num_segments)
            frame_start = np.random.randint(segment_size - frame_size)
            frame_end = frame_start + frame_size
            raw_data = signal[segment_index, frame_start:frame_end]

            x = spectrogram_preprocessor(raw_data, frame_size=frame_size, window_size = window_size, stride = stride, n_freqs = n_freqs)
            
            # calculate the count of each beat type in the frame and determine the final label
            beat_ends, beat_labels = patient_beat_labels[segment_index]
            _, frame_beat_labels = get_complete_beats(
                beat_ends, beat_labels, frame_start, frame_end)
            y = get_beat_label(frame_beat_labels)
            yield x, y

def spectrogram_beat_data_generator(db_dir, patient_ids, frame_size,
                   unzipped=False, samples_per_patient=1, window_size=256, stride = 64, n_freqs=-1, ):

    patient_generator = uniform_patient_generator(
        _str(db_dir), patient_ids, unzipped=bool(unzipped))
    data_generator = spectrogram_beat_generator(
        patient_generator, frame_size=int(frame_size), window_size = window_size, samples_per_patient=int(samples_per_patient),
        stride = int(stride), n_freqs = int(n_freqs))
    return data_generator

def spectrogram_beat_dataset(db_dir, patient_ids, frame_size,
                   unzipped=False, samples_per_patient=1, window_size = 256, stride = 64, n_freqs=-1):
    
    #expected output size
    time_dim = frame_size//stride
    freq_dim = window_size//2 if n_freqs == -1 else n_freqs
    dataset = tf.data.Dataset.from_generator(
        generator=spectrogram_beat_data_generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape((freq_dim, time_dim, 1)), tf.TensorShape(())),
        args=(db_dir, patient_ids, frame_size, unzipped, samples_per_patient, window_size, stride, n_freqs))
    return dataset