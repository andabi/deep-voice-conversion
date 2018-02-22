# -*- coding: utf-8 -*-
#!/usr/bin/env python

import tensorflow as tf

from data_load import get_batch_queue, phns
from modules import prenet, cbhg
import glob, os
from utils import split_path


class Model:
    def __init__(self, mode, batch_size, hp, queue=True):
        self.mode = mode
        self.batch_size = batch_size
        self.hp = hp
        self.queue = queue
        self.is_training = self.get_is_training(mode)

        # Input
        self.x_mfcc, self.y_ppg, self.y_spec, self.y_mel, self.num_batch = self.get_input(mode, batch_size, queue)

        # Networks
        self.net_template = tf.make_template('net', self._net2)
        self.ppgs, self.pred_ppg, self.logits_ppg, self.pred_spec, self.pred_mel = self.net_template()

    def __call__(self):
        return self.pred_spec

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

        x_mfcc = tf.placeholder(tf.float32, shape=(batch_size, None, self.hp.default.n_mfcc))
        y_ppgs = tf.placeholder(tf.int32, shape=(batch_size, None,))
        y_spec = tf.placeholder(tf.float32, shape=(batch_size, None, 1 + self.hp.default.n_fft // 2))
        y_mel = tf.placeholder(tf.float32, shape=(batch_size, None, self.hp.default.n_mels))
        num_batch = 1

        if queue:
            if mode in ("train1", "test1"):  # x: mfccs (N, T, n_mfccs), y: Phones (N, T)
                x_mfcc, y_ppgs, num_batch = get_batch_queue(mode=mode, batch_size=batch_size)
            elif mode in ("train2", "test2", "convert"): # x: mfccs (N, T, n_mfccs), y: spectrogram (N, T, 1+n_fft//2)
                x_mfcc, y_spec, y_mel, num_batch = get_batch_queue(mode=mode, batch_size=batch_size)
        return x_mfcc, y_ppgs, y_spec, y_mel, num_batch

    def get_is_training(self, mode):
        if mode in ('train1', 'train2'):
            is_training = True
        else:
            is_training = False
        return is_training

    def _net1(self):
        with tf.variable_scope('net1'):
            # Pre-net
            prenet_out = prenet(self.x_mfcc,
                                num_units=[self.hp.train1.hidden_units, self.hp.train1.hidden_units // 2],
                                dropout_rate=self.hp.train1.dropout_rate,
                                is_training=self.is_training)  # (N, T, E/2)

            # CBHG
            out = cbhg(prenet_out, self.hp.train1.num_banks, self.hp.train1.hidden_units // 2, self.hp.train1.num_highway_blocks, self.hp.train1.norm_type, self.is_training)

            # Final linear projection
            logits = tf.layers.dense(out, len(phns))  # (N, T, V)
            ppgs = tf.nn.softmax(logits / self.hp.train1.t)  # (N, T, V)
            preds = tf.to_int32(tf.arg_max(logits, dimension=-1))  # (N, T)

        return ppgs, preds, logits

    def loss_net1(self):
        istarget = tf.sign(tf.abs(tf.reduce_sum(self.x_mfcc, -1)))  # indicator: (N, T)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_ppg / self.hp.train1.t, labels=self.y_ppg)
        loss *= istarget
        loss = tf.reduce_mean(loss)
        return loss

    def acc_net1(self):
        istarget = tf.sign(tf.abs(tf.reduce_sum(self.x_mfcc, -1)))  # indicator: (N, T)
        num_hits = tf.reduce_sum(tf.to_float(tf.equal(self.pred_ppg, self.y_ppg)) * istarget)
        num_targets = tf.reduce_sum(istarget)
        acc = num_hits / num_targets
        return acc

    def _net2(self):
        # PPGs from net1
        ppgs, preds_ppg, logits_ppg = self._net1()

        with tf.variable_scope('net2'):
            # Pre-net
            prenet_out = prenet(ppgs,
                                num_units=[self.hp.train2.hidden_units, self.hp.train2.hidden_units // 2],
                                dropout_rate=self.hp.train2.dropout_rate,
                                is_training=self.is_training)  # (N, T, E/2)

            # CBHG1: mel-scale
            pred_mel = cbhg(prenet_out, self.hp.train2.num_banks, self.hp.train2.hidden_units // 2, self.hp.train2.num_highway_blocks, self.hp.train2.norm_type, self.is_training, scope="cbhg1")
            pred_mel = tf.layers.dense(pred_mel, self.y_mel.shape[-1])  # log magnitude: (N, T, n_mels)
            # pred_mel = prenet_out

            # CBHG2: linear-scale
            pred_spec = tf.layers.dense(pred_mel, self.hp.train2.hidden_units // 2)  # log magnitude: (N, T, n_mels)
            pred_spec = cbhg(pred_spec, self.hp.train2.num_banks, self.hp.train2.hidden_units // 2, self.hp.train2.num_highway_blocks, self.hp.train2.norm_type, self.is_training, scope="cbhg2")
            pred_spec = tf.layers.dense(pred_spec, self.y_spec.shape[-1])  # log magnitude: (N, T, 1+hp.n_fft//2)

        return ppgs, preds_ppg, logits_ppg, pred_spec, pred_mel

    def loss_net2(self):
        loss_spec = tf.reduce_mean(tf.squared_difference(self.pred_spec, self.y_spec))
        loss_mel = tf.reduce_mean(tf.squared_difference(self.pred_mel, self.y_mel))
        # loss_mel = 0
        loss = loss_spec + loss_mel
        return loss

    @staticmethod
    def load(sess, mode, logdir, logdir2=None, step=None):
        def print_model_loaded(mode, logdir, step):
            model_name = Model.get_model_name(logdir, step=step)
            print('Model loaded. mode: {}, model_name: {}'.format(mode, model_name))

        if mode in ['train1', 'test1']:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1')
            if Model._load_variables(sess, logdir, var_list=var_list, step=step):
                print_model_loaded(mode, logdir, step)

        elif mode == 'train2':
            var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1')
            if Model._load_variables(sess, logdir, var_list=var_list1, step=step):
                print_model_loaded(mode, logdir, step)

            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net2')
            if Model._load_variables(sess, logdir2, var_list=var_list2, step=step):
                print_model_loaded(mode, logdir2, step)

        elif mode in ['test2', 'convert']:
            if Model._load_variables(sess, logdir, var_list=None, step=step):  # Load all variables
                print_model_loaded(mode, logdir, step)

    @staticmethod
    def _load_variables(sess, logdir, var_list, step=None):
        model_name = Model.get_model_name(logdir, step)
        if model_name:
            ckpt = os.path.join(logdir, model_name)
            tf.train.Saver(var_list=var_list).restore(sess, ckpt)
            return True
        else:
            return False

    @staticmethod
    def get_model_name(logdir, step=None):
        model_name = None
        if step:
            paths = glob.glob('{}/*step_{}.index*'.format(logdir, step))
            if paths:
                _, model_name, _ = split_path(paths[0])
        else:
            ckpt = tf.train.latest_checkpoint(logdir)
            if ckpt:
                _, model_name = os.path.split(ckpt)
        return model_name

    @staticmethod
    def get_epoch_and_global_step(logdir, step=None):
        model_name = Model.get_model_name(logdir, step)
        if model_name:
            tokens = model_name.split('_')
            epoch, gs = int(tokens[1]), int(tokens[3])
        else:
            epoch = gs = 0
        return epoch, gs

    @staticmethod
    def all_model_names(logdir):
        path = '{}/*.meta'.format(logdir)
        model_names = map(lambda f: os.path.basename(f).replace('.meta', ''), glob.glob(path))
        return model_names
