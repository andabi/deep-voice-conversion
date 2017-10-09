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
