# -*- coding: utf-8 -*-
# /usr/bin/python2

import glob
import threading
from functools import wraps
from random import sample

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from utils import *
from utils import preemphasis
from hparams import Hyperparams as hp


def load_vocab():
    '''
    len(phns) is aliased as V.
    :return:
    '''
    phns = ['PAD', 'UNK', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
     'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
     'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
     'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
     'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
    phn2idx = {phn:idx for idx, phn in enumerate(phns)}
    idx2phn = {idx:phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn


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
        testset_size = hp.test2.batch_size * 4
        wav_files = glob.glob(hp.train2.data_path)[testset_size:][:1000]
    elif mode == "test2":
        testset_size = hp.test2.batch_size * 4
        wav_files = glob.glob(hp.train2.data_path)[:testset_size]
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
                return get_mfccs_and_spectrogram(wav, sr, duration=hp.duration)

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


# Adapted from the `sugartensor` code.
# https://github.com/buriburisuri/sugartensor/blob/master/sugartensor/sg_queue.py
def producer_func(func):
    r"""Decorates a function `func` as producer_func.

    Args:
      func: A function to decorate.
    """

    @wraps(func)
    def wrapper(inputs, dtypes, capacity, num_threads):
        r"""
        Args:
            inputs: A inputs queue list to enqueue
            dtypes: Data types of each tensor
            capacity: Queue capacity. Default is 32.
            num_threads: Number of threads. Default is 1.
        """

        # enqueue function
        def enqueue_func(sess, op):
            # read data from source queue
            data = func(sess.run(inputs))
            # create feeder dict
            feed_dict = {}
            for ph, col in zip(placeholders, data):
                feed_dict[ph] = col
            # run session
            sess.run(op, feed_dict=feed_dict)

        # create place holder list
        placeholders = []
        for dtype in dtypes:
            placeholders.append(tf.placeholder(dtype=dtype))

        # create FIFO queue
        queue = tf.FIFOQueue(capacity, dtypes=dtypes)

        # enqueue operation
        enqueue_op = queue.enqueue(placeholders)

        # create queue runner
        runner = _FuncQueueRunner(enqueue_func, queue, [enqueue_op] * num_threads)

        # register to global collection
        tf.train.add_queue_runner(runner)

        # return de-queue operation
        return queue.dequeue()

    return wrapper


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


def get_mfccs_and_phones(wav_file, sr, trim=True, random_crop=False, crop_timesteps=hp.sr/hp.hop_length):
    '''This is applied in `train1` or `test1` phase.
    '''

    # Load
    wav, sr = librosa.load(wav_file, sr=sr)

    # Get MFCCs
    mfccs, _ = get_mfccs_and_spectrogram(wav, sr, trim=False, random_crop=False)

    # timesteps
    num_timesteps = mfccs.shape[0]

    # phones (targets)
    phn_file = wav_file.replace("WAV.wav", "PHN").replace("wav", "PHN")
    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    for line in open(phn_file, 'r').read().splitlines():
        start_point, _, phn = line.split()
        bnd = int(start_point) // hp.hop_length
        phns[bnd:] = phn2idx[phn]
        bnd_list.append(bnd)

    # Trim
    if trim:
        start, end = bnd_list[1], bnd_list[-1]
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Random crop
    if random_crop:
        start = np.random.choice(range(np.maximum(1, len(mfccs) - crop_timesteps)), 1)[0]
        end = start + crop_timesteps
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    return mfccs, phns


def get_mfccs_and_spectrogram(wav, sr, trim=True, duration=1, random_crop=False):
    '''This is applied in `train2` or `test2` phase.
    '''

    # Trim
    if trim:
        wav, _ = librosa.effects.trim(wav)

    # Fix duration
    if random_crop:
        wav = wav_random_crop(wav, hp.sr, duration)
    else:
        len = sr * duration
        wav = librosa.util.fix_length(wav, len)

    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=hp.preemphasis)

    # Get spectrogram
    D = librosa.stft(y=y_preem,
                     n_fft=hp.n_fft,
                     hop_length=hp.hop_length,
                     win_length=hp.win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels) # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag**1) # (n_mels, t) # mel spectrogram

    # Get mfccs
    mel = librosa.power_to_db(mel)
    mfccs = np.dot(librosa.filters.dct(hp.n_mfcc, mel.shape[0]), mel)

    return mfccs.T, mag.T # (t, n_mfccs),  (t, 1+n_fft/2)


class _FuncQueueRunner(tf.train.QueueRunner):
    def __init__(self, func, queue=None, enqueue_ops=None, close_op=None,
                 cancel_op=None, queue_closed_exception_types=None,
                 queue_runner_def=None):
        # save ad-hoc function
        self.func = func
        # call super()
        super(_FuncQueueRunner, self).__init__(queue, enqueue_ops, close_op, cancel_op,
                                               queue_closed_exception_types, queue_runner_def)

    # pylint: disable=broad-except
    def _run(self, sess, enqueue_op, coord=None):

        if coord:
            coord.register_thread(threading.current_thread())
        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    self.func(sess, enqueue_op)  # call enqueue function
                except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1


def wav_random_crop(wav, sr, duration):
    assert(wav.ndim <= 2)

    target_len = sr * duration
    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - target_len)), 1)[0]
    end = start + target_len
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav