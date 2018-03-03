# -*- coding: utf-8 -*-
# !/usr/bin/env python

from scipy import signal
from pydub import AudioSegment
import os
import librosa
import soundfile as sf
import numpy as np


def read_wav(path, sr, duration=None, mono=True):
    wav, _ = librosa.load(path, mono=mono, sr=sr, duration=duration)
    return wav


def write_wav(wav, sr, path, format='wav', subtype='PCM_16'):
    sf.write(path, wav, sr, format=format, subtype=subtype)


def read_mfcc(prefix):
    filename = '{}.mfcc.npy'.format(prefix)
    mfcc = np.load(filename)
    return mfcc


def write_mfcc(prefix, mfcc):
    filename = '{}.mfcc'.format(prefix)
    np.save(filename, mfcc)


def read_spectrogram(prefix):
    filename = '{}.spec.npy'.format(prefix)
    spec = np.load(filename)
    return spec


def write_spectrogram(prefix, spec):
    filename = '{}.spec'.format(prefix)
    np.save(filename, spec)


def split_wav(wav, top_db):
    intervals = librosa.effects.split(wav, top_db=top_db)
    wavs = map(lambda i: wav[i[0]: i[1]], intervals)
    return wavs


def trim_wav(wav):
    wav, _ = librosa.effects.trim(wav)
    return wav


def fix_length(wav, length):
    if len(wav) != length:
        wav = librosa.util.fix_length(wav, length)
    return wav


def crop_random_wav(wav, length):
    """
    Randomly cropped a part in a wav file.
    :param wav: a waveform
    :param length: length to be randomly cropped.
    :return: a randomly cropped part of wav.
    """
    assert (wav.ndim <= 2)
    assert (type(length) == int)

    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - length)), 1)[0]
    end = start + length
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def mp3_to_wav(src_path, tar_path):
    """
    Read mp3 file from source path, convert it to wav and write it to target path. 
    Necessary libraries: ffmpeg, libav.

    :param src_path: source mp3 file path
    :param tar_path: target wav file path
    """
    basepath, filename = os.path.split(src_path)
    os.chdir(basepath)
    AudioSegment.from_mp3(src_path).export(tar_path, format='wav')


def prepro_audio(source_path, target_path, format=None, sr=None, db=None):
    """
    Read a wav, change sample rate, format, and average decibel and write to target path.
    :param source_path: source wav file path
    :param target_path: target wav file path
    :param sr: sample rate.
    :param format: output audio format.
    :param db: decibel.
    """
    sound = AudioSegment.from_file(source_path, format)
    if sr:
        sound = sound.set_frame_rate(sr)
    if db:
        change_dBFS = db - sound.dBFS
        sound = sound.apply_gain(change_dBFS)
    sound.export(target_path, 'wav')


def _split_path(path):
    """
    Split path to basename, filename and extension. For example, 'a/b/c.wav' => ('a/b', 'c', 'wav')
    :param path: file path
    :return: basename, filename, and extension
    """
    basepath, filename = os.path.split(path)
    filename, extension = os.path.splitext(filename)
    return basepath, filename, extension


def wav2spec(wav, n_fft, win_length, hop_length, time_first=True):
    """
    Get magnitude and phase spectrogram from waveforms.

    Parameters
    ----------
    wav : np.ndarray [shape=(n,)]
        The real-valued waveform.

    n_fft : int > 0 [scalar]
        FFT window size.

    win_length  : int <= n_fft [scalar]
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

    hop_length : int > 0 [scalar]
        Number audio of frames between STFT columns.

    time_first : boolean. optional.
        if True, time axis is followed by bin axis. In this case, shape of returns is (t, 1 + n_fft/2)

    Returns
    -------
    mag : np.ndarray [shape=(t, 1 + n_fft/2) or (1 + n_fft/2, t)]
        Magnitude spectrogram.

    phase : np.ndarray [shape=(t, 1 + n_fft/2) or (1 + n_fft/2, t)]
        Phase spectrogram.

    """
    stft = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(stft)
    phase = np.angle(stft)

    if time_first:
        mag = mag.T
        phase = phase.T

    return mag, phase


def spec2wav(mag, n_fft, win_length, hop_length, num_iters=30, phase=None):
    """
    Get a waveform from the magnitude spectrogram by Griffin-Lim Algorithm.

    Parameters
    ----------
    mag : np.ndarray [shape=(1 + n_fft/2, t)]
        Magnitude spectrogram.

    n_fft : int > 0 [scalar]
        FFT window size.

    win_length  : int <= n_fft [scalar]
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

    hop_length : int > 0 [scalar]
        Number audio of frames between STFT columns.

    num_iters: int > 0 [scalar]
        Number of iterations of Griffin-Lim Algorithm.

    phase : np.ndarray [shape=(1 + n_fft/2, t)]
        Initial phase spectrogram.

    Returns
    -------
    wav : np.ndarray [shape=(n,)]
        The real-valued waveform.

    """
    assert (num_iters > 0)
    if phase is None:
        phase = np.pi * np.random.rand(*mag.shape)
    stft = mag * np.exp(1.j * phase)
    wav = None
    for i in range(num_iters):
        wav = librosa.istft(stft, win_length=win_length, hop_length=hop_length)
        if i != num_iters - 1:
            stft = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            _, phase = librosa.magphase(stft)
            phase = np.angle(phase)
            stft = mag * np.exp(1.j * phase)
    return wav


def preemphasis(wav, coeff=0.97):
    """
    Emphasize high frequency range of the waveform by increasing power(squared amplitude).

    Parameters
    ----------
    wav : np.ndarray [shape=(n,)]
        Real-valued the waveform.

    coeff: float <= 1 [scalar]
        Coefficient of pre-emphasis.

    Returns
    -------
    preem_wav : np.ndarray [shape=(n,)]
        The pre-emphasized waveform.
    """
    preem_wav = signal.lfilter([1, -coeff], [1], wav)
    return preem_wav


def inv_preemphasis(preem_wav, coeff=0.97):
    """
    Invert the pre-emphasized waveform to the original waveform.

    Parameters
    ----------
    preem_wav : np.ndarray [shape=(n,)]
        The pre-emphasized waveform.

    coeff: float <= 1 [scalar]
        Coefficient of pre-emphasis.

    Returns
    -------
    wav : np.ndarray [shape=(n,)]
        Real-valued the waveform.
    """
    wav = signal.lfilter([1], [1, -coeff], preem_wav)
    return wav


def linear_to_mel(linear, sr, n_fft, n_mels, **kwargs):
    """
    Convert a linear-spectrogram to mel-spectrogram.
    :param linear: Linear-spectrogram.
    :param sr: Sample rate.
    :param n_fft: FFT window size.
    :param n_mels: Number of mel filters.
    :return: Mel-spectrogram.
    """
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels, **kwargs)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, linear)  # (n_mels, t) # mel spectrogram
    return mel


def amp2db(amp):
    return librosa.amplitude_to_db(amp)


def db2amp(db):
    return librosa.db_to_amplitude(db)


def normalize_db(db, max_db, min_db):
    """
    Normalize dB-scaled spectrogram values to be in range of 0~1.
    :param db: Decibel-scaled spectrogram.
    :param max_db: Maximum dB.
    :param min_db: Minimum dB.
    :return: Normalized spectrogram.
    """
    norm_db = np.clip((db - min_db) / (max_db - min_db), 0, 1)
    return norm_db


def denormalize_db(norm_db, max_db, min_db):
    """
    Denormalize the normalized values to be original dB-scaled value.
    :param norm_db: Normalized spectrogram.
    :param max_db: Maximum dB.
    :param min_db: Minimum dB.
    :return: Decibel-scaled spectrogram.
    """
    db = np.clip(norm_db, 0, 1) * (max_db - min_db) + min_db
    return db


def dynamic_range_compression(db, threshold, ratio, method='downward'):
    """
    Execute dynamic range compression(https://en.wikipedia.org/wiki/Dynamic_range_compression) to dB.
    :param db: Decibel-scaled magnitudes
    :param threshold: Threshold dB
    :param ratio: Compression ratio.
    :param method: Downward or upward.
    :return: Range compressed dB-scaled magnitudes
    """
    if method is 'downward':
        db[db > threshold] = (db[db > threshold] - threshold) / ratio + threshold
    elif method is 'upward':
        db[db < threshold] = threshold - ((threshold - db[db < threshold]) / ratio)
    return db


def emphasize_magnitude(mag, power=1.2):
    """
    Emphasize a magnitude spectrogram by applying power function. This is used for removing noise.
    :param mag: magnitude spectrogram.
    :param power: exponent.
    :return: emphasized magnitude spectrogram.
    """
    emphasized_mag = np.power(mag, power)
    return emphasized_mag


def wav2melspec(wav, sr, n_fft, win_length, hop_length, n_mels, time_first=True, **kwargs):
    # Linear spectrogram
    mag_spec, phase_spec = wav2spec(wav, n_fft, win_length, hop_length, time_first=False)

    # Mel-spectrogram
    mel_spec = linear_to_mel(mag_spec, sr, n_fft, n_mels, **kwargs)

    # Time-axis first
    if time_first:
        mel_spec = mel_spec.T  # (t, n_mels)

    return mel_spec


def wav2melspec_db(wav, sr, n_fft, win_length, hop_length, n_mels, normalize=False, max_db=None, min_db=None,
                   time_first=True, **kwargs):
    # Mel-spectrogram
    mel_spec = wav2melspec(wav, sr, n_fft, win_length, hop_length, n_mels, time_first=False, **kwargs)

    # Decibel
    mel_db = librosa.amplitude_to_db(mel_spec)

    # Normalization
    mel_db = normalize_db(mel_db, max_db, min_db) if normalize else mel_db

    # Time-axis first
    if time_first:
        mel_db = mel_db.T  # (t, n_mels)

    return mel_db


def wav2mfcc(wav, sr, n_fft, win_length, hop_length, n_mels, n_mfccs, preemphasis_coeff=0.97, time_first=True,
             **kwargs):
    # Pre-emphasis
    wav_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Decibel-scaled mel-spectrogram
    mel_db = wav2melspec_db(wav_preem, sr, n_fft, win_length, hop_length, n_mels, time_first=False, **kwargs)

    # MFCCs
    mfccs = np.dot(librosa.filters.dct(n_mfccs, mel_db.shape[0]), mel_db)

    # Time-axis first
    if time_first:
        mfccs = mfccs.T  # (t, n_mfccs)

    return mfccs