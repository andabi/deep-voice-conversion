# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib import distributions
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from tensorpack.tfutils import (
    get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

import tensorpack_extension
from data_load import phns
from hparam import hparam as hp
from modules import prenet, cbhg, normalize
import sys


class Net1(ModelDesc):
    def __init__(self):
        pass

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, None, hp.default.n_mfcc), 'x_mfccs'),
                InputDesc(tf.int32, (None, None,), 'y_ppgs')]

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
    def __init__(self, batch_size):
        self.net1 = Net1()
        self.batch_size = batch_size

    def _get_inputs(self):
        return [InputDesc(tf.float32, (self.batch_size, None, hp.default.n_mfcc), 'x_mfccs'),
                InputDesc(tf.float32, (self.batch_size, None, hp.default.n_fft // 2 + 1), 'y_spec'),
                InputDesc(tf.float32, (self.batch_size, None, hp.default.n_mels), 'y_mel'), ]

    def _build_graph(self, inputs):
        self.x_mfcc, self.y_spec, self.y_mel = inputs

        is_training = get_current_tower_context().is_training

        # build net1
        with tf.variable_scope('net1'):
            self.ppgs, _, _ = self.net1.network(self.x_mfcc, is_training)

        # build net2
        with tf.variable_scope('net2'):
            self.pred_spec_mu, self.pred_spec_logvar, self.pred_spec_phi = self.network(self.ppgs, is_training)

        self.cost = self.loss()

        # build for conversion phase
        self.convert()

        # summaries
        tf.summary.scalar('net2/train/loss', self.cost)
        # tf.summary.scalar('net2/train/lr', lr)
        tf.summary.histogram('net2/train/mu', self.pred_spec_mu)
        tf.summary.histogram('net2/train/var', tf.exp(self.pred_spec_logvar))
        tf.summary.histogram('net2/train/phi', self.pred_spec_phi)

    def _get_optimizer(self):
        gradprocs = [
            tensorpack_extension.FilterGradientVariables('.*net2.*', verbose=False),
            gradproc.MapGradient(lambda grad: tf.clip_by_value(grad, hp.train2.clip_value_min, hp.train2.clip_value_max)),
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
        # pred_mel = cbhg(prenet_out, hp.train2.num_banks, hp.train2.hidden_units // 2,
        #                 hp.train2.num_highway_blocks, hp.train2.norm_type, is_training,
        #                 scope="cbhg_mel")
        # pred_mel = tf.layers.dense(pred_mel, self.y_mel.shape[-1])  # (N, T, n_mels)
        pred_mel = prenet_out

        # CBHG2: linear-scale
        out = tf.layers.dense(pred_mel, hp.train2.hidden_units // 2)  # (N, T, n_mels)
        out = cbhg(out, hp.train2.num_banks, hp.train2.hidden_units // 2,
                         hp.train2.num_highway_blocks, hp.train2.norm_type, is_training,
                         scope="cbhg_linear")

        batch_size, _, num_bins = self.y_spec.get_shape().as_list()
        num_units = num_bins * hp.train2.n_mixtures
        out = tf.layers.dense(out, num_units * 3, bias_initializer=tf.random_uniform_initializer(minval=-3., maxval=3.))

        mu = tf.nn.sigmoid(out[..., :num_units])
        mu = tf.reshape(mu, shape=(batch_size, -1, num_bins, hp.train2.n_mixtures))  # (N, T, 1+hp.n_fft//2, n_mixtures)

        logvar = tf.clip_by_value(out[..., num_units: 2 * num_units], clip_value_min=-7, clip_value_max=7)
        logvar = tf.reshape(logvar, shape=(batch_size, -1, num_bins, hp.train2.n_mixtures))  # (N, T, 1+hp.n_fft//2, n_mixtures)

        # normalize to prevent softmax output to be NaN.
        pi = tf.reshape(out[..., 2 * num_units: 3 * num_units], shape=(batch_size, -1, num_bins, hp.train2.n_mixtures))  # (N, T, 1+hp.n_fft//2, n_mixtures)
        pi = normalize(pi, type='ins', is_training=get_current_tower_context().is_training, scope='normalize_phi')
        pi = tf.nn.softmax(pi)

        return mu, logvar, pi

    def loss(self):
        # negative log likelihood
        logistic_dists = []
        for i in range(hp.train2.n_mixtures):
            mu = self.pred_spec_mu[..., i]
            logvar = self.pred_spec_logvar[..., i]
            logistic_dist = distributions.Logistic(mu, tf.exp(logvar))
            logistic_dists.append(logistic_dist)
        cat = distributions.Categorical(probs=self.pred_spec_phi)
        mixture_dist = distributions.Mixture(cat=cat, components=logistic_dists)
        cdf_pos = mixture_dist.cdf(value=self.y_spec + hp.train2.mol_step)
        cdf_neg = mixture_dist.cdf(value=self.y_spec - hp.train2.mol_step)
        # FIXME: why minus prob?
        prob = cdf_pos - cdf_neg
        prob /= hp.train2.mol_step * 2
        prob = tf.maximum(prob, sys.float_info.epsilon)

        tf.summary.histogram('net2/train/cdf_pos', cdf_pos)
        tf.summary.histogram('net2/train/cdf_neg', cdf_neg)
        tf.summary.scalar('net2/train/min_cdf_pos', tf.reduce_min(cdf_pos))
        tf.summary.scalar('net2/train/min_cdf_neg', tf.reduce_min(cdf_neg))
        tf.summary.histogram('net2/train/prob', prob)
        tf.summary.scalar('net2/prob_min', tf.reduce_min(prob))

        loss_mle = -tf.reduce_mean(tf.log(prob))

        mean = tf.reduce_sum(self.pred_spec_mu * self.pred_spec_phi, axis=-1, keep_dims=True)
        loss_mix = tf.reduce_sum(self.pred_spec_phi * tf.squared_difference(self.pred_spec_mu, mean), axis=-1)
        loss_mix = -tf.reduce_mean(loss_mix)

        lamb = 0
        loss = loss_mle + lamb * loss_mix

        tf.summary.scalar('net2/train/loss_mle', loss_mle)
        tf.summary.scalar('net2/train/loss_mix', loss_mix)

        return loss

    def convert(self):
        argmax = tf.one_hot(tf.argmax(self.pred_spec_phi, axis=-1), hp.train2.n_mixtures)
        pred_spec = tf.reduce_sum(self.pred_spec_mu * argmax, axis=-1, name='pred_spec')
        return pred_spec