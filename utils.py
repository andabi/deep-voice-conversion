# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/vc
'''
from __future__ import print_function

import copy

import librosa
import numpy as np
import tensorflow as tf

from hyperparams import Hyperparams as hp

from functools import wraps
import threading

from tensorflow.python.platform import tf_logging as logging


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


def get_mfccs_and_spectrogram(wav_file, trim=True, random_crop=True, crop_duration=1):
    '''This is applied in `train2` or `test2` phase.
    '''
    # Load
    y, sr = librosa.load(wav_file, sr=hp.sr)

    # Trim
    if trim:
        y, _ = librosa.effects.trim(y)

    # Random crop
    if random_crop:
        y = wav_random_crop(y, hp.sr, crop_duration)

    # Get spectrogram
    D = librosa.stft(y=y,
                     n_fft=hp.n_fft,
                     hop_length=hp.hop_length,
                     win_length=hp.win_length)
    mag = np.abs(D)

    # MFCCs
    y_preem = preemphasis(y, coeff=hp.preemphasis)
    mfccs = librosa.feature.mfcc(y=y_preem,
                                 sr=hp.sr,
                                 n_mfcc=hp.n_mfcc,
                                 n_fft=hp.n_fft,
                                 hop_length=hp.hop_length,
                                 n_mels=hp.n_mels) # (n_mfccs, t)

    return mfccs.T, mag.T # (t, n_mfccs),  (t, 1+n_fft/2)


def get_mfccs_and_phones(wav_file, trim=True, random_crop=True, crop_timesteps=201):
    '''This is applied in `train1` or `test1` phase.
    '''
    # Get MFCCs
    mfccs, _ = get_mfccs_and_spectrogram(wav_file, trim=False, random_crop=False)

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


def spectrogram2wav(mag, n_fft, win_length, hop_length, num_iters, phase_angle=None, length=None):
    '''
    
    :param mag: [f, t]
    :param n_fft: n_fft
    :param win_length: window length
    :param hop_length: hop length
    :param num_iters: num of iteration when griffin-lim reconstruction
    :param phase_angle: phase angle
    :param length: length of wav
    :return: 
    '''
    assert(num_iters > 0)
    if phase_angle is None:
        phase_angle = np.pi * np.random.rand(*mag.shape)
    spec = mag * np.exp(1.j * phase_angle)
    for i in range(num_iters):
        wav = librosa.istft(spec, win_length=win_length, hop_length=hop_length, length=length)
        if i != num_iters - 1:
            spec = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            _, phase = librosa.magphase(spec)
            phase_angle = np.angle(phase)
            spec = mag * np.exp(1.j * phase_angle)
    return wav


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")


def preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


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