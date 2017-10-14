# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import sys

import tensorflow as tf

from data_load import get_batch_queue, load_vocab
from hyperparams import Hyperparams as hp
from modules import prenet, conv1d, conv1d_banks, normalize, gru, highwaynet


class Model:
    def __init__(self, mode=None, batch_size=hp.batch_size, queue=True, log_mag=True):
        self.mode = mode
        self.batch_size = batch_size
        self.queue = queue
        self.log_mag = log_mag
        self.is_training = self.get_is_training(mode)

        # Input
        self.x_mfcc, self.y_ppgs, self.y_spec, self.num_batch = self.get_input(mode, batch_size, queue)

        # Convert to log of magnitude
        if log_mag:
            self.y_log_spec = tf.log(self.y_spec + sys.float_info.epsilon)
        else:
            self.y_log_spec = self.y_spec

        # Networks
        self.net_template = tf.make_template('net', self._net2)
        self.ppgs, self.preds_ppg, self.logits_ppg, self.preds_spec = self.net_template()

    def __call__(self):
        return self.preds_spec

    def get_input(self, mode, batch_size, queue):
        '''
        mode: A string. One of the phases below:
          `train1`: TIMIT TRAIN waveform -> mfccs (inputs) -> PGGs -> phones (target) (ce loss)
          `test1`: TIMIT TEST waveform -> mfccs (inputs) -> PGGs -> phones (target) (accuracy)
          `train2`: ARCTIC SLT waveform -> mfccs -> PGGs (inputs) -> spectrogram (target)(l2 loss)
          `test2`: ARCTIC SLT waveform -> mfccs -> PGGs (inputs) -> spectrogram (target)(accuracy)
          `convert`: ARCTIC BDL waveform -> mfccs (inputs) -> PGGs -> spectrogram -> waveform (output)
        '''
        if mode not in ('train1', 'test1', 'train2', 'test2', 'convert'):
            raise Exception("invalid mode={}".format(mode))

        x_mfcc = tf.placeholder(tf.float32, shape=(batch_size, None, hp.n_mfcc))
        y_ppgs = tf.placeholder(tf.int32, shape=(batch_size, None,))
        y_spec = tf.placeholder(tf.float32, shape=(batch_size, None, 1 + hp.n_fft // 2))
        num_batch = 1

        if queue:
            if mode in ("train1", "test1"):  # x: mfccs (N, T, n_mfccs), y: Phones (N, T)
                x_mfcc, y_ppgs, num_batch = get_batch_queue(mode=mode, batch_size=batch_size)
            elif mode in ("train2", "test2", "convert"): # x: mfccs (N, T, n_mfccs), y: spectrogram (N, T, 1+n_fft//2)
                x_mfcc, y_spec, num_batch = get_batch_queue(mode=mode, batch_size=batch_size)
        return x_mfcc, y_ppgs, y_spec, num_batch

    def get_is_training(self, mode):
        if mode in ('train1', 'train2'):
            is_training = True
        else:
            is_training = False
        return is_training

    def _net1(self):
        with tf.variable_scope('net1'):
            # Load vocabulary
            phn2idx, idx2phn = load_vocab()

            # Pre-net
            prenet_out = prenet(self.x_mfcc,
                                num_units=[hp.hidden_units, hp.hidden_units // 2],
                                dropout_rate=hp.dropout_rate,
                                is_training=self.is_training)  # (N, T, E/2)

            # CBHG
            ## Conv1D banks
            enc = conv1d_banks(prenet_out,
                               K=hp.encoder_num_banks,
                               num_units=hp.hidden_units // 2,
                               norm_type=hp.norm_type,
                               is_training=self.is_training)  # (N, T, K * E / 2)

            ## Max pooling
            enc = tf.layers.max_pooling1d(enc, 2, 1, padding="same")  # (N, T, K * E / 2)

            ## Conv1D projections
            enc = conv1d(enc, hp.hidden_units // 2, 3, scope="conv1d_1")  # (N, T, E/2)
            enc = normalize(enc, type=hp.norm_type, is_training=self.is_training, activation_fn=tf.nn.relu)
            enc = conv1d(enc, hp.hidden_units // 2, 3, scope="conv1d_2")  # (N, T, E/2)
            enc += prenet_out  # (N, T, E/2) # residual connections

            ## Highway Nets
            for i in range(hp.num_highwaynet_blocks):
                enc = highwaynet(enc,
                                 num_units=hp.hidden_units // 2,
                                 scope='highwaynet_{}'.format(i))  # (N, T, E/2)

            ## Bidirectional GRU
            enc = gru(enc, hp.hidden_units // 2, True)  # (N, T, E)

            # Final linear projection
            logits = tf.layers.dense(enc, len(phn2idx))  # (N, T, V)
            ppgs = tf.nn.softmax(logits) # (N, T, V)
            preds = tf.to_int32(tf.arg_max(logits, dimension=-1))  # (N, T)

        return ppgs, preds, logits

    def loss_net1(self):
        istarget = tf.sign(tf.abs(tf.reduce_sum(self.x_mfcc, -1)))  # indicator: (N, T)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_ppg, labels=self.y_ppgs)
        loss *= istarget
        loss = tf.reduce_mean(loss)
        return loss

    def acc_net1(self):
        istarget = tf.sign(tf.abs(tf.reduce_sum(self.x_mfcc, -1)))  # indicator: (N, T)
        num_hits = tf.reduce_sum(tf.to_float(tf.equal(self.preds_ppg, self.y_ppgs)) * istarget)
        num_targets = tf.reduce_sum(istarget)
        acc = num_hits / num_targets
        return acc

    def _net2(self):
        # PPGs from net1
        ppgs, preds_ppg, logits_ppg = self._net1()

        with tf.variable_scope('net2'):
            # Encoder
            # Encoder pre-net
            prenet_out = prenet(ppgs,
                                num_units=[hp.hidden_units, hp.hidden_units // 2],
                                dropout_rate=hp.dropout_rate,
                                is_training=self.is_training)  # (N, T, E/2)

            # Encoder CBHG
            ## Conv1D bank
            enc = conv1d_banks(prenet_out,
                               K=hp.encoder_num_banks,
                               num_units=hp.hidden_units // 2,
                               norm_type=hp.norm_type,
                               is_training=self.is_training)  # (N, T, K * E / 2)

            ### Max pooling
            enc = tf.layers.max_pooling1d(enc, 2, 1, padding="same")  # (N, T, K * E / 2)

            ### Conv1D projections
            enc = conv1d(enc, hp.hidden_units // 2, 3, scope="conv1d_1")  # (N, T, E/2)
            enc = normalize(enc, type="ins", is_training=self.is_training, activation_fn=tf.nn.relu)
            enc = conv1d(enc, hp.hidden_units // 2, 3, scope="conv1d_2")  # (N, T, E/2)
            enc += prenet_out  # (N, T, E/2) # residual connections

            ### Highway Nets
            for i in range(hp.num_highwaynet_blocks):
                enc = highwaynet(enc, num_units=hp.hidden_units // 2,
                                 scope='highwaynet_{}'.format(i))  # (N, T, E/2)

            ### Bidirectional GRU
            enc = gru(enc, hp.hidden_units // 2, True)  # (N, T, E)

            # Final projection
            preds_spec = tf.layers.dense(enc, self.y_log_spec.shape[-1])  # log magnitude: (N, T, 1+hp.n_fft//2)

        return ppgs, preds_ppg, logits_ppg, preds_spec

    def loss_net2(self):  # Loss
        loss = tf.squared_difference(self.preds_spec, self.y_log_spec)

        # Mean loss
        loss = tf.reduce_mean(loss)

        return loss

    @staticmethod
    def load(sess, mode, logdir, logdir2=None):

        def print_model_loaded(mode, logdir):
            model_name = Model.get_model_name(logdir)
            print('Model loaded. mode: {}, model_name: {}'.format(mode, model_name))

        if mode in ['train1', 'test1']:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1')
            if Model._load_variables(sess, logdir, var_list=var_list):
                print_model_loaded(mode, logdir)

        elif mode == 'train2':
            var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1')
            if Model._load_variables(sess, logdir, var_list=var_list1):
                print_model_loaded(mode, logdir)

            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net2')
            if Model._load_variables(sess, logdir2, var_list=var_list2):
                print_model_loaded(mode, logdir2)

        elif mode in ['test2', 'convert']:
            if Model._load_variables(sess, logdir, var_list=None):  # Load all variables
                print_model_loaded(mode, logdir)

    @staticmethod
    def _load_variables(sess, logdir, var_list):
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt:
            tf.train.Saver(var_list=var_list).restore(sess, ckpt)
            return True
        else:
            return False

    @staticmethod
    def get_model_name(logdir):
        path = '{}/checkpoint'.format(logdir)
        if os.path.exists(path):
            ckpt_path = open(path, 'r').read().split('"')[1]
            _, model_name = os.path.split(ckpt_path)
        else:
            model_name = None
        return model_name

    @staticmethod
    def get_global_step(logdir):
        model_name = Model.get_model_name(logdir)
        if model_name:
            gs = int(model_name.split('_')[3])
        else:
            gs = 0
        return gs

    @staticmethod
    def all_model_names(logdir):
        import glob, os
        path = '{}/*.meta'.format(logdir)
        model_names = map(lambda f: os.path.basename(f).replace('.meta', ''), glob.glob(path))
        return model_names
