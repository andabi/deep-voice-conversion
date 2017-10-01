# -*- coding: utf-8 -*-
#!/usr/bin/env python

import librosa
import glob
import os


# src_path = '/Users/avin/git/vc/datasets/arctic/bdl'
src_path = '/Users/avin/git/vc/datasets/kate/sense_and_sensibility'
target_path = '{}_split'.format(src_path)
sr = 22050
top_db = 30


def split(wav, top_db):
    intervals = librosa.effects.split(wav, top_db=top_db)
    wavs = map(lambda i: wav[i[0]: i[1]], intervals)
    return wavs


def read(path, sr):
    wav, _ = librosa.load(path, mono=True, sr=sr)
    return wav


def write(wav, sr, path):
    librosa.output.write_wav(path, wav, sr)


def split_path(path):
    '''
    'a/b/c.wav' => ('a/b', 'c', 'wav')
    :param path: filepath = 'a/b/c.wav'
    :return: basename, filename, and extension = ('a/b', 'c', 'wav')
    '''
    basepath, filename = os.path.split(path)
    filename, extension = os.path.splitext(filename)
    return basepath, filename, extension


num_files = num_files_split = 0
max_len = max_len_split = 0
min_len = min_len_split = float('inf')
for filepath in glob.glob('{}/*.wav'.format(src_path)):
    wav = read(filepath, sr)
    split_wavs = split(wav, top_db)
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    _, filename, _ = split_path(filepath)
    map(lambda (i, w): write(w, sr, '{}/{}_{}.wav'.format(target_path, filename, i)), enumerate(split_wavs))

    num_files += 1
    num_files_split += len(split_wavs)

    max_len = max(max_len, len(wav))
    max_len_split = max(max_len_split, max(map(lambda w: len(w), split_wavs)))

    min_len = min(min_len, len(wav))
    min_len_split = min(min_len_split, min(map(lambda w: len(w), split_wavs)))

print('num_files: {}, num_files_split: {}'.format(num_files, num_files_split))
print('max_len: {}, max_len_split: {}'.format(max_len, max_len_split))
print('min_len: {}, min_len_split: {}'.format(min_len, min_len_split))

# range of arctic/bdl: 20000~120000