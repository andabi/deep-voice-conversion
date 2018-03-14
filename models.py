# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib import distributions
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from tensorpack.train.tower import get_current_tower_context

from data_load import phns
from hparam import hparam as hp
from modules import prenet, cbhg
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils import (
    summary, get_current_tower_context, optimizer, gradproc)


class Net1(ModelDesc):
    def __init__(self):
        pass

    def _get_inputs(self):
        return [InputDesc(tf.float32, (hp.train1.batch_size, None, hp.default.n_mfcc), 'x_mfccs'),
                InputDesc(tf.int32, (hp.train1.batch_size, None,), 'y_ppgs')]

    def _build_graph(self, inputs):
        self.x_mfccs, self.y_ppgs = inputs
        with tf.variable_scope('net1'):
            self.ppgs, self.preds, self.logits = self.network(self.x_mfccs, get_current_tower_context().is_training)
        self.cost = self.loss()

        # summaries
        tf.summary.scalar('net1/train/loss', self.cost)
        tf.summary.scalar('net1/train/acc', self.acc())

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
        ppgs = tf.nn.softmax(logits / hp.train1.t)  # (N, T, V)
        preds = tf.to_int32(tf.arg_max(logits, dimension=-1))  # (N, T)

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
    def __init__(self):
        self.net1 = Net1()

    def _get_inputs(self):
        return [InputDesc(tf.float32, (hp.train2.batch_size, None, hp.default.n_mfcc), 'x_mfccs'),
                InputDesc(tf.float32, (hp.train2.batch_size, None, hp.default.n_fft // 2 + 1), 'y_spec'),
                InputDesc(tf.float32, (hp.train2.batch_size, None, hp.default.n_mels), 'y_mel'), ]

    def _build_graph(self, inputs):
        self.x_mfcc, self.y_spec, self.y_mel = inputs

        is_training = get_current_tower_context().is_training

        # build net1
        with tf.variable_scope('net1'):
            self.ppgs, _, _ = self.net1.network(self.x_mfcc, is_training)

        # build net2
        with tf.variable_scope('net2'):
            self.pred_spec_mu, self.pred_spec_phi = self.network(self.ppgs, is_training)

        self.cost = self.loss()

        # summaries
        tf.summary.scalar('net2/train/loss', self.cost)
        # tf.summary.scalar('net2/train/lr', lr)
        tf.summary.histogram('net2/train/mu', self.pred_spec_mu)
        tf.summary.histogram('net2/train/phi', self.pred_spec_phi)

        # TODO remove
        tf.summary.scalar('net2/prob_min', self.prob_min)

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=hp.train2.lr, trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        gradprocs = [gradproc.MapGradient(lambda grad: grad, regex='.*net2.*'),  # apply only gradients of net2
                     gradproc.GlobalNormClip(hp.train2.clip_norm),
                     # gradproc.PrintGradient()]
                     ]

        return optimizer.apply_grad_processors(opt, gradprocs)

    @auto_reuse_variable_scope
    def network(self, ppgs, is_training):
        # Pre-net
        prenet_out = prenet(ppgs,
                            num_units=[hp.train2.hidden_units, hp.train2.hidden_units // 2],
                            dropout_rate=hp.train2.dropout_rate,
                            is_training=is_training)  # (N, T, E/2)

        # CBHG1: mel-scale
        # pred_mel = cbhg(prenet_out, hp.train2.num_banks, hp.train2.hidden_units // 2,
        #                 hp.train2.num_highway_blocks, hp.train2.norm_type, is_training,
        #                 scope="cbhg_mel")
        # pred_mel = tf.layers.dense(pred_mel, self.y_mel.shape[-1])  # (N, T, n_mels)
        pred_mel = prenet_out

        # CBHG2: linear-scale
        pred_spec = tf.layers.dense(pred_mel, hp.train2.hidden_units // 2)  # (N, T, n_mels)
        pred_spec = cbhg(pred_spec, hp.train2.num_banks, hp.train2.hidden_units // 2,
                         hp.train2.num_highway_blocks, hp.train2.norm_type, is_training,
                         scope="cbhg_linear")
        pred_spec = tf.layers.dense(pred_spec, self.y_spec.shape[-1])  # (N, T, 1+hp.n_fft//2)
        pred_spec = tf.expand_dims(pred_spec, axis=-1)
        pred_spec_mu = tf.layers.dense(pred_spec, hp.train2.n_mixtures)  # (N, T, 1+hp.n_fft//2, n_mixtures)
        pred_spec_phi = tf.nn.softmax(
            tf.layers.dense(pred_spec, hp.train2.n_mixtures))  # (N, T, 1+hp.n_fft//2, n_mixtures)

        return pred_spec_mu, pred_spec_phi

    def loss(self):
        # negative log likelihood
        normal_dists = []
        for i in range(hp.train2.n_mixtures):
            mu = self.pred_spec_mu[..., i]
            normal_dist = distributions.Logistic(mu, tf.ones_like(mu))
            normal_dists.append(normal_dist)
        cat = distributions.Categorical(probs=self.pred_spec_phi)
        mixture_dist = distributions.Mixture(cat=cat, components=normal_dists)
        prob = mixture_dist.cdf(value=self.y_spec + hp.train2.mol_step) - \
               mixture_dist.cdf(value=self.y_spec - hp.train2.mol_step)
        prob /= hp.train2.mol_step * 2
        self.prob_min = tf.reduce_min(prob)
        loss = -tf.reduce_mean(tf.log(prob + 1e-8))
        return loss
