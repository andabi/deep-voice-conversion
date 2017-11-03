# -*- coding: utf-8 -*-
# /usr/bin/python2

import glob
import sys
import threading
from functools import wraps
from random import sample

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

import hparams as hp
from hparams import Default as hp_default
from utils import *
from utils import preemphasis


def get_mfccs_and_phones(wav_file, sr, trim=False, random_crop=True,
                         length=int(hp_default.duration / hp_default.frame_shift + 1)):
    '''This is applied in `train1` or `test1` phase.
    '''

    # Load
    wav, sr = librosa.load(wav_file, sr=sr)

    mfccs, _, _ = _get_mfcc_log_spec_and_log_mel_spec(wav, hp_default.preemphasis, hp_default.n_fft,
                                                      hp_default.win_length,
                                                      hp_default.hop_length)

    # timesteps
    num_timesteps = mfccs.shape[0]

    # phones (targets)
    phn_file = wav_file.replace("WAV.wav", "PHN").replace("wav", "PHN")
    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    for line in open(phn_file, 'r').read().splitlines():
        start_point, _, phn = line.split()
        bnd = int(start_point) // hp_default.hop_length
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
        start = np.random.choice(range(np.maximum(1, len(mfccs) - length)), 1)[0]
        end = start + length
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Padding or crop
    mfccs = librosa.util.fix_length(mfccs, length, axis=0)
    phns = librosa.util.fix_length(phns, length, axis=0)

    return mfccs, phns


def get_mfccs_and_spectrogram(wav_file, trim=True, duration=None, random_crop=False):
    '''This is applied in `train2`, `test2` or `convert` phase.
    '''

    # Load
    wav, _ = librosa.load(wav_file, sr=hp_default.sr)

    # Trim
    if trim:
        wav, _ = librosa.effects.trim(wav)

    if random_crop:
        wav = wav_random_crop(wav, hp_default.sr, duration)

    # Padding or crop
    if duration:
        length = hp_default.sr * duration
        wav = librosa.util.fix_length(wav, length)

    return _get_mfcc_log_spec_and_log_mel_spec(wav, hp_default.preemphasis, hp_default.n_fft,
                                               hp_default.win_length, hp_default.hop_length)


def _get_mfcc_log_spec_and_log_mel_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):
    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(hp_default.sr, hp_default.n_fft, hp_default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs
    db = librosa.amplitude_to_db(mel)
    mfccs = np.dot(librosa.filters.dct(hp_default.n_mfcc, db.shape[0]), db)

    # Log
    mag = np.log(mag + sys.float_info.epsilon)
    mel = np.log(mel + sys.float_info.epsilon)

    # Normalization
    # self.y_log_spec = (y_log_spec - hp.mean_log_spec) / hp.std_log_spec
    # self.y_log_spec = (y_log_spec - hp.min_log_spec) / (hp.max_log_spec - hp.min_log_spec)

    return mfccs.T, mag.T, mel.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)


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
    mfccs, phns = get_mfccs_and_phones(wav_file, hp_default.sr)
    return mfccs, phns


@producer_func
def get_mfccs_and_spectrogram_queue(wav_file):
    '''From `wav_file` (string), which has been fetched from slice queues,
       extracts mfccs and spectrogram, then enqueue them again.
       This is applied in `train2` or `test2` phase.
    '''
    mfccs, spec, mel = get_mfccs_and_spectrogram(wav_file, duration=hp_default.duration)
    return mfccs, spec, mel


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
            mfcc, ppg = get_mfccs_and_phones_queue(inputs=wav_file,
                                                   dtypes=[tf.float32, tf.int32],
                                                   capacity=2048,
                                                   num_threads=32)

            # create batch queues
            mfcc, ppg = tf.train.batch([mfcc, ppg],
                                       shapes=[(None, hp_default.n_mfcc), (None,)],
                                       num_threads=32,
                                       batch_size=batch_size,
                                       capacity=batch_size * 32,
                                       dynamic_pad=True)
            return mfcc, ppg, num_batch
        else:
            # Get inputs and target
            mfcc, spec, mel = get_mfccs_and_spectrogram_queue(inputs=wav_file,
                                                              dtypes=[tf.float32, tf.float32, tf.float32],
                                                              capacity=2048,
                                                              num_threads=64)

            # create batch queues
            mfcc, spec, mel = tf.train.batch([mfcc, spec, mel],
                                             shapes=[(None, hp_default.n_mfcc), (None, 1 + hp_default.n_fft // 2),
                                                     (None, hp_default.n_mels)],
                                             num_threads=64,
                                             batch_size=batch_size,
                                             capacity=batch_size * 64,
                                             dynamic_pad=True)
            return mfcc, spec, mel, num_batch


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
            mfcc, ppg = map(_get_zero_padded, zip(*map(lambda w: get_mfccs_and_phones(w, hp_default.sr), target_wavs)))
            return mfcc, ppg
        else:
            mfcc, spec, mel = map(_get_zero_padded, zip(*map(
                lambda wav_file: get_mfccs_and_spectrogram(wav_file, duration=hp_default.duration), target_wavs)))
            return mfcc, spec, mel


# TODO generalize for all mode
def get_wav_batch(mode, batch_size):
    with tf.device('/cpu:0'):
        # Load data
        wav_files = load_data(mode=mode)
        wav_file = sample(wav_files, 1)[0]
        wav, _ = librosa.load(wav_file, sr=hp_default.sr)

        total_duration = hp_default.duration * batch_size
        total_len = hp_default.sr * total_duration
        wav = librosa.util.fix_length(wav, total_len)

        length = hp_default.sr * hp_default.duration
        batched = np.reshape(wav, (batch_size, length))

        mfcc, spec, mel = map(_get_zero_padded, zip(
            *map(lambda w: _get_mfcc_log_spec_and_log_mel_spec(w, hp_default.preemphasis, hp_default.n_fft,
                                                               hp_default.win_length, hp_default.hop_length), batched)))
    return mfcc, spec, mel


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


def load_vocab():
    phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
            'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
            'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
            'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
            'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

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
        wav_files = glob.glob(hp.Train1.data_path)
    elif mode == "test1":
        wav_files = glob.glob(hp.Test1.data_path)
    elif mode == "train2":
        testset_size = hp.Test2.batch_size * 4
        wav_files = glob.glob(hp.Train2.data_path)[testset_size:]
    elif mode == "test2":
        testset_size = hp.Test2.batch_size * 4
        wav_files = glob.glob(hp.Train2.data_path)[:testset_size]
    elif mode == "convert":
        wav_files = glob.glob(hp.Convert.data_path)
    return wav_files
