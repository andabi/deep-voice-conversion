# -*- coding: utf-8 -*-
#/usr/bin/python2


# path
## remote
# data_path = '/data/private/vc/datasets'
# logdir_path = '/data/private/vc'

## local
data_path_base = './datasets'
logdir_path = '.'


class Hyperparams:
    # signal processing
    sr = 16000 # Sampling rate.
    frame_shift = 0.005  # seconds
    frame_length = 0.025  # seconds
    n_fft = 512

    # sr = 20000 # Sampling rate.
    # frame_shift = 0.0125  # seconds
    # frame_length = 0.05  # seconds
    # n_fft = 1024

    hop_length = int(sr*frame_shift) # samples.  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples. This is dependent on the frame_length.

    preemphasis = 0.97
    n_mfcc = 40
    n_iter = 30 # Number of inversion iterations
    n_mels = 128
    duration = 1
    emphasis_magnitude = 1.2

    # model
    hidden_units = 256 # alias = E
    num_banks = 16
    num_highwaynet_blocks = 4
    norm_type = 'ins'  # a normalizer function. value: bn, ln, ins, or None
    dropout_rate = 0.2
    t = 1.0  # temperature

    # default
    batch_size = 32

    class train1:
        data_path = '{}/timit/TIMIT/TRAIN/*/*/*.wav'.format(data_path_base)
        batch_size = 32
        lr = 0.0005
        num_epochs = 10000
        save_per_epoch = 2

    class test1:
        data_path = '{}/timit/TIMIT/TEST/*/*/*.wav'.format(data_path_base)
        batch_size = 32

    class train2:
        data_path = '{}/kate/sense_and_sensibility_split/*.wav'.format(data_path_base)
        # data_path = '{}/kate/therese_raquin_split/*.wav'.format(data_path_base)
        # data_path = '{}/arctic/slt/*.wav'.format(data_path_base)
        batch_size = 32
        lr = 0.0005
        num_epochs = 10000
        save_per_epoch = 50

    class test2:
        data_path = '{}/arctic/bdl/*.wav'.format(data_path_base)
        # data_path = '{}/kate/sense_and_sensibility_split/*.wav'.format(data_path_base)
        # wav_files = glob.glob('datasets/arctic/slt/*.wav'.format(hp.data_path))[-10:]
        batch_size = 32

    class convert:
        data_path = '{}/arctic/bdl/*.wav'.format(data_path_base)
        # wav_files = glob.glob('{}/arctic/slt/*.wav'.format(hp.data_path))
        # wav_files = glob.glob('{}/iKala/Wavfile/*.wav'.format(hp.data_path))
        # wav_files = glob.glob('{}/kate/sense_and_sensibility_split/*.wav'.format(hp.data_path))[-100:]
        batch_size = 3