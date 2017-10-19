# -*- coding: utf-8 -*-
# /usr/bin/python2

from utils import *
import glob
import tensorflow as tf
from random import sample

phn2idx, idx2phn = load_vocab()
V = len(phn2idx)


def load_data(mode):
    '''Loads the list of sound files.
    mode: A string. One of the phases below:
      `train1`: TIMIT TRAIN waveform -> mfccs (inputs) -> PGGs -> phones (target) (ce loss)
      `test1`: TIMIT TEST waveform -> mfccs (inputs) -> PGGs -> phones (target) (accuracy)
      `train2`: ARCTIC SLT waveform -> mfccs -> PGGs (inputs) -> spectrogram (target)(l2 loss)
      `test2`: ARCTIC SLT waveform -> mfccs -> PGGs (inputs) -> spectrogram (target)(accuracy)
      `convert`: ARCTIC BDL waveform -> mfccs (inputs) -> PGGs -> spectrogram -> waveform (output)
    '''
    if mode == "train1":
        wav_files = glob.glob(hp.train1.data_path)
    elif mode == "test1":
        wav_files = glob.glob(hp.test1.data_path)
    elif mode == "train2":
        wav_files = glob.glob(hp.train2.data_path)
    elif mode == "test2":
        wav_files = glob.glob(hp.test2.data_path)
    elif mode == "convert":
        wav_files = glob.glob(hp.convert.data_path)
    return wav_files


def get_batch_queue(mode, batch_size):
    '''Loads data and put them in mini batch queues.
    mode: A string. Either `train1` | `test1` | `train2` | `test2` | `convert`.
    '''

    if mode not in ('train1', 'test1', 'train2', 'test2', 'convert'):
        raise Exception("invalid mode={}".format(mode))

    with tf.device('/cpu:0'):
        # Load data
        wav_files = load_data(mode=mode)

        # calc total batch count
        num_batch = len(wav_files) // batch_size

        # Convert to tensor
        wav_files = tf.convert_to_tensor(wav_files)

        # Create Queues
        wav_file, = tf.train.slice_input_producer([wav_files, ], shuffle=True, capacity=128)

        if mode in ('train1', 'test1'):
            # Get inputs and target
            x, y = get_mfccs_and_phones_queue(inputs=wav_file,
                                              dtypes=[tf.float32, tf.int32],
                                              capacity=2048,
                                              num_threads=32)

            # create batch queues
            x, y = tf.train.batch([x, y],
                                  shapes=[(None, hp.n_mfcc), (None,)],
                                  num_threads=32,
                                  batch_size=batch_size,
                                  capacity=batch_size * 32,
                                  dynamic_pad=True)

        else:
            # Get inputs and target
            x, y = get_mfccs_and_spectrogram_queue(inputs=wav_file,
                                                   dtypes=[tf.float32, tf.float32],
                                                   capacity=2048,
                                                   num_threads=64)

            # create batch queues
            x, y = tf.train.batch([x, y],
                                shapes=[(None, hp.n_mfcc), (None, 1+hp.n_fft//2)],
                                num_threads=64,
                                batch_size=batch_size,
                                capacity=batch_size * 64,
                                dynamic_pad=True)

        return x, y, num_batch


def get_batch(mode, batch_size):
    '''Loads data.
    mode: A string. Either `train1` | `test1` | `train2` | `test2` | `convert`.
    '''

    if mode not in ('train1', 'test1', 'train2', 'test2', 'convert'):
        raise Exception("invalid mode={}".format(mode))

    with tf.device('/cpu:0'):
        # Load data
        wav_files = load_data(mode=mode)

        target_wavs = sample(wav_files, batch_size)

        if mode in ('train1', 'test1'):
            x, y = map(_get_zero_padded, zip(*map(lambda w: get_mfccs_and_phones(w, hp.sr), target_wavs)))
        else:
            def load_and_get_mfccs_and_spectrogram(wav_file):
                wav, sr = librosa.load(wav_file, sr=hp.sr)
                get_mfccs_and_spectrogram(wav, sr, duration=hp.duration)

            x, y = map(_get_zero_padded, zip(*map(lambda w: load_and_get_mfccs_and_spectrogram(w), target_wavs)))
    return x, y


# TODO generalize for all mode
def get_batch_per_wav(mode, batch_size):
    with tf.device('/cpu:0'):
        # Load data
        wav_files = load_data(mode=mode)
        wav_file = sample(wav_files, 1)[0]
        wav, sr = librosa.load(wav_file, sr=hp.sr)

        total_duration = hp.duration * batch_size
        total_len = hp.sr * total_duration
        wav = librosa.util.fix_length(wav, total_len)
        len = hp.duration * hp.sr
        batched = np.reshape(wav, (batch_size, len))

        x, y = map(_get_zero_padded, zip(*map(lambda w: get_mfccs_and_spectrogram(w, sr, duration=hp.duration), batched)))
    return x, y


def _get_zero_padded(list_of_arrays):
    '''
    :param list_of_arrays
    :return: zero padded array
    '''
    batch = []
    max_len = 0
    for d in list_of_arrays:
        max_len = max(len(d), max_len)
    for d in list_of_arrays:
        num_pad = max_len - len(d)
        pad_width = [(0, num_pad)]
        for _ in range(d.ndim - 1):
            pad_width.append((0, 0))
        pad_width = tuple(pad_width)
        padded = np.pad(d, pad_width=pad_width, mode="constant", constant_values=0)
        batch.append(padded)
    return np.array(batch)


@producer_func
def get_mfccs_and_phones_queue(wav_file):
    '''From a single `wav_file` (string), which has been fetched from slice queues,
       extracts mfccs (inputs), and phones (target), then enqueue them again.
       This is applied in `train1` or `test1` phase.
    '''
    mfccs, phns = get_mfccs_and_phones(wav_file, hp.sr)
    return mfccs, phns


@producer_func
def get_mfccs_and_spectrogram_queue(wav_file):
    '''From `wav_file` (string), which has been fetched from slice queues,
       extracts mfccs and spectrogram, then enqueue them again.
       This is applied in `train2` or `test2` phase.
    '''
    wav, sr = librosa.load(wav_file, sr=hp.sr)
    mfccs, spectrogram = get_mfccs_and_spectrogram(wav, sr, duration=hp.duration)
    return mfccs, spectrogram