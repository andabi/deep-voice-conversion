# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from tensorpack.tfutils import (
    get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

import tensorpack_extension
from data_load import phns
from hparam import hparam as hp
from modules import prenet, cbhg, normalize


class Net1(ModelDesc):
    def __init__(self):
        pass

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, None, hp.default.n_mfcc), 'x_mfccs'),
                InputDesc(tf.int32, (None, None,), 'y_ppgs')]

    def _build_graph(self, inputs):
        self.x_mfccs, self.y_ppgs = inputs
        is_training = get_current_tower_context().is_training
        with tf.variable_scope('net1'):
            self.ppgs, self.preds, self.logits = self.network(self.x_mfccs, is_training)
        self.cost = self.loss()
        acc = self.acc()

        # summaries
        tf.summary.scalar('net1/train/loss', self.cost)
        tf.summary.scalar('net1/train/acc', acc)

        if not is_training:
            # summaries
            tf.summary.scalar('net1/eval/summ_loss', self.cost)
            tf.summary.scalar('net1/eval/summ_acc', acc)

            # for confusion matrix
            tf.reshape(self.y_ppgs, shape=(tf.size(self.y_ppgs),), name='net1/eval/y_ppg_1d')
            tf.reshape(self.preds, shape=(tf.size(self.preds),), name='net1/eval/pred_ppg_1d')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=hp.train1.lr, trainable=False)
        return tf.train.AdamOptimizer(lr)

    @auto_reuse_variable_scope
    def network(self, x_mfcc, is_training):
        # Pre-net
        prenet_out = prenet(x_mfcc,
                            num_units=[hp.train1.hidden_units, hp.train1.hidden_units // 2],
                            dropout_rate=hp.train1.dropout_rate,
                            is_training=is_training)  # (N, T, E/2)

        # CBHG
        out = cbhg(prenet_out, hp.train1.num_banks, hp.train1.hidden_units // 2,
                   hp.train1.num_highway_blocks, hp.train1.norm_type, is_training)

        # Final linear projection
        logits = tf.layers.dense(out, len(phns))  # (N, T, V)
        ppgs = tf.nn.softmax(logits / hp.train1.t, name='ppgs')  # (N, T, V)
        preds = tf.to_int32(tf.argmax(logits, axis=-1))  # (N, T)

        return ppgs, preds, logits

    def loss(self):
        istarget = tf.sign(tf.abs(tf.reduce_sum(self.x_mfccs, -1)))  # indicator: (N, T)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits / hp.train1.t,
                                                              labels=self.y_ppgs)
        loss *= istarget
        loss = tf.reduce_mean(loss)
        return loss

    def acc(self):
        istarget = tf.sign(tf.abs(tf.reduce_sum(self.x_mfccs, -1)))  # indicator: (N, T)
        num_hits = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y_ppgs)) * istarget)
        num_targets = tf.reduce_sum(istarget)
        acc = num_hits / num_targets
        return acc


class Net2(ModelDesc):

    def _get_inputs(self):
        n_timesteps = (hp.default.duration * hp.default.sr) // hp.default.hop_length + 1

        return [InputDesc(tf.float32, (None, n_timesteps, hp.default.n_mfcc), 'x_mfccs'),
                InputDesc(tf.float32, (None, n_timesteps, hp.default.n_fft // 2 + 1), 'y_spec'),
                InputDesc(tf.float32, (None, n_timesteps, hp.default.n_mels), 'y_mel'), ]

    def _build_graph(self, inputs):
        self.x_mfcc, self.y_spec, self.y_mel = inputs

        is_training = get_current_tower_context().is_training

        # build net1
        self.net1 = Net1()
        with tf.variable_scope('net1'):
            self.ppgs, _, _ = self.net1.network(self.x_mfcc, is_training)
        self.ppgs = tf.identity(self.ppgs, name='ppgs')

        # build net2
        with tf.variable_scope('net2'):
            self.pred_spec, self.pred_mel = self.network(self.ppgs, is_training)
        self.pred_spec = tf.identity(self.pred_spec, name='pred_spec')

        self.cost = self.loss()

        # summaries
        tf.summary.scalar('net2/train/loss', self.cost)

        if not is_training:
            tf.summary.scalar('net2/eval/summ_loss', self.cost)

    def _get_optimizer(self):
        gradprocs = [
            tensorpack_extension.FilterGradientVariables('.*net2.*', verbose=False),
            gradproc.MapGradient(
                lambda grad: tf.clip_by_value(grad, hp.train2.clip_value_min, hp.train2.clip_value_max)),
            gradproc.GlobalNormClip(hp.train2.clip_norm),
            # gradproc.PrintGradient(),
            # gradproc.CheckGradient(),
        ]
        lr = tf.get_variable('learning_rate', initializer=hp.train2.lr, trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        return optimizer.apply_grad_processors(opt, gradprocs)

    @auto_reuse_variable_scope
    def network(self, ppgs, is_training):
        # Pre-net
        prenet_out = prenet(ppgs,
                            num_units=[hp.train2.hidden_units, hp.train2.hidden_units // 2],
                            dropout_rate=hp.train2.dropout_rate,
                            is_training=is_training)  # (N, T, E/2)

        # CBHG1: mel-scale
        pred_mel = cbhg(prenet_out, hp.train2.num_banks, hp.train2.hidden_units // 2,
                        hp.train2.num_highway_blocks, hp.train2.norm_type, is_training,
                        scope="cbhg_mel")
        pred_mel = tf.layers.dense(pred_mel, self.y_mel.shape[-1], name='pred_mel')  # (N, T, n_mels)

        # CBHG2: linear-scale
        pred_spec = tf.layers.dense(pred_mel, hp.train2.hidden_units // 2)  # (N, T, n_mels)
        pred_spec = cbhg(pred_spec, hp.train2.num_banks, hp.train2.hidden_units // 2,
                   hp.train2.num_highway_blocks, hp.train2.norm_type, is_training, scope="cbhg_linear")
        pred_spec = tf.layers.dense(pred_spec, self.y_spec.shape[-1], name='pred_spec')  # log magnitude: (N, T, 1+n_fft//2)

        return pred_spec, pred_mel

    def loss(self):
        loss_spec = tf.reduce_mean(tf.squared_difference(self.pred_spec, self.y_spec))
        loss_mel = tf.reduce_mean(tf.squared_difference(self.pred_mel, self.y_mel))
        loss = loss_spec + loss_mel
        return loss