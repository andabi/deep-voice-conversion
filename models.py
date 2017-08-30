# -*- coding: utf-8 -*-
#!/usr/bin/env python

import tensorflow as tf
from data_load import get_batch, load_vocab
from modules import prenet, conv1d, conv1d_banks, normalize, gru, highwaynet
from hyperparams import Hyperparams as hp


class Model:
    def __init__(self, mode=None):
        '''
        mode: A string. One of the phases below:
          `train1`: TIMIT TRAIN waveform -> mfccs (inputs) -> PGGs -> phones (target) (ce loss)
          `test1`: TIMIT TEST waveform -> mfccs (inputs) -> PGGs -> phones (target) (accuracy)
          `train2`: ARCTIC SLT waveform -> mfccs -> PGGs (inputs) -> spectrogram (target)(l2 loss)
          `test2`: ARCTIC SLT waveform -> mfccs -> PGGs (inputs) -> spectrogram (target)(accuracy)
          `convert`: ARCTIC BDL waveform -> mfccs (inputs) -> PGGs -> spectrogram -> waveform (output)
        '''

        self.mode = mode
        self.is_training = False

        self.x_mfcc = tf.placeholder(tf.float32, shape=(None, hp.n_mfcc))
        self.y_ppgs = tf.placeholder(tf.int32, shape=(None, None))
        self.y_spec = tf.placeholder(tf.float32, shape=(None, 1 + hp.n_fft // 2))

        # TODO refactoring
        # Inputs
        if mode == "train1": # x: mfccs (N, T, n_mfccs), y: Phones (N, T)
            self.x_mfcc, self.y_ppgs, self.num_batch = get_batch(mode=mode)
            self.is_training = True
        elif mode == "train2": # x: mfccs (N, T, n_mfccs), y: spectrogram (N, T, 1+n_fft//2)
            self.x_mfcc, self.y_spec, self.num_batch = get_batch(mode=mode)
            self.is_training = True
        elif mode == "test1":  # x: mfccs (N, T, n_mfccs), y: Phones (N, T)
            self.x_mfcc, self.y_ppgs, self.num_batch = get_batch(mode=mode)
        elif mode == "test2":
            self.x_mfcc, self.y_spec, self.num_batch = get_batch(mode=mode)
        else: # `convert`
            self.x_mfcc, self.y_spec, self.num_batch = get_batch(mode=mode)

        # Networks
        self.net_template = tf.make_template('net', self._net2)
        self.ppgs, self.preds_ppgs, self.logits_ppgs, self.preds_spec = self.net_template()

    def __call__(self):
        return self.preds_spec

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
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_ppgs, labels=self.y_ppgs)
        loss *= istarget
        loss = tf.reduce_mean(loss)
        return loss

    def acc_net1(self):
        istarget = tf.sign(tf.abs(tf.reduce_sum(self.x_mfcc, -1)))  # indicator: (N, T)
        num_hits = tf.reduce_sum(tf.to_float(tf.equal(self.preds_ppgs, self.y_ppgs)) * istarget)
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

            # Final linear projection
            # logits_spec = tf.layers.dense(enc, self.y_spec.shape[-1])  # log magnitude: (N, T, 1+hp.n_fft//2)
            # istarget = tf.sign(tf.abs(tf.reduce_sum(ppgs, -1)))  # (N, T)
            # preds_spec = tf.to_int32(tf.arg_max(logits_spec, dimension=-1)) # (N, T)
            # preds_spec *= tf.to_int32(istarget)  # (N, T)
            preds_spec = tf.layers.dense(enc, self.y_spec.shape[-1])  # log magnitude: (N, T, 1+hp.n_fft//2)

        return ppgs, preds_ppg, logits_ppg, preds_spec

    def loss_net2(self):  # Loss
        loss = tf.squared_difference(self.preds_spec, self.y_spec)

        # Mean loss
        loss = tf.reduce_mean(loss)

        return loss