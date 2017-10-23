# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By Dabi Ahn. andabi412@gmail.com.
https://www.github.com/andabi
'''

from pydub import AudioSegment
import os
import librosa
import soundfile as sf
import numpy as np
from scipy import signal


def read(path, sr, mono=False):
    wav, _ = librosa.load(path, mono=mono, sr=sr)
    return wav


def write(wav, sr, path, format='wav', subtype='PCM_16'):
    sf.write(path, wav, sr, format=format, subtype=subtype)


def rewrite_mp3_to_wav(source_path, target_path):
    '''
    Necessary libraries: ffmpeg, libav
    :param source_path: 
    :param target_path: 
    :return: 
    '''
    basepath, filename = os.path.split(source_path)
    os.chdir(basepath)
    AudioSegment.from_mp3(source_path).export(target_path, format='wav')


def split_path(path):
    '''
    'a/b/c.wav' => ('a/b', 'c', 'wav')
    :param path: filepath = 'a/b/c.wav'
    :return: basename, filename, and extension = ('a/b', 'c', 'wav')
    '''
    basepath, filename = os.path.split(path)
    filename, extension = os.path.splitext(filename)
    return basepath, filename, extension


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
    assert (num_iters > 0)
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


def preemphasis(x, coeff=0.97):
    return signal.lfilter([1, -coeff], [1], x)


def inv_preemphasis(x, coeff=0.97):
    return signal.lfilter([1], [1, -coeff], x)
