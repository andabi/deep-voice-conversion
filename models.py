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
            self.mu, self.log_var, self.log_pi = self.network(self.ppgs, is_training)

        self.cost = self.loss()

        # summaries
        tf.summary.scalar('net2/train/loss', self.cost)
        tf.summary.histogram('net2/train/mu', self.mu)
        tf.summary.histogram('net2/train/var', tf.exp(self.log_var))
        tf.summary.histogram('net2/train/pi', tf.exp(self.log_pi))

        if not is_training:
            tf.summary.scalar('net2/eval/summ_loss', self.cost)

            # build for conversion phase
            self.convert()

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
        # pred_mel = cbhg(prenet_out, hp.train2.num_banks, hp.train2.hidden_units // 2,
        #                 hp.train2.num_highway_blocks, hp.train2.norm_type, is_training,
        #                 scope="cbhg_mel")
        # pred_mel = tf.layers.dense(pred_mel, self.y_mel.shape[-1])  # (N, T, n_mels)
        pred_mel = prenet_out

        # CBHG2: linear-scale
        out = tf.layers.dense(pred_mel, hp.train2.hidden_units // 2)  # (N, T, n_mels)
        out = cbhg(out, hp.train2.num_banks, hp.train2.hidden_units // 2,
                   hp.train2.num_highway_blocks, hp.train2.norm_type, is_training, scope="cbhg_linear")

        _, n_timesteps, n_bins = self.y_spec.get_shape().as_list()
        n_units = n_bins * hp.train2.n_mixtures
        out = tf.layers.dense(out, n_units * 3, bias_initializer=tf.random_uniform_initializer(minval=-3., maxval=3.))

        mu = tf.nn.sigmoid(out[..., :n_units])
        mu = tf.reshape(mu, shape=(-1, n_timesteps, n_bins, hp.train2.n_mixtures))  # (N, T, 1+hp.n_fft//2, n_mixtures)

        log_var = tf.maximum(out[..., n_units: 2 * n_units], -7.0)
        log_var = tf.reshape(log_var,
                            shape=(-1, n_timesteps, n_bins, hp.train2.n_mixtures))  # (N, T, 1+hp.n_fft//2, n_mixtures)

        log_pi = tf.reshape(out[..., 2 * n_units: 3 * n_units],
                        shape=(-1, n_timesteps, n_bins, hp.train2.n_mixtures))  # (N, T, 1+hp.n_fft//2, n_mixtures)
        log_pi = normalize(log_pi, type='ins', is_training=get_current_tower_context().is_training, scope='normalize_pi')
        log_pi = tf.nn.log_softmax(log_pi)

        return mu, log_var, log_pi

    def loss(self):
        y = tf.expand_dims(self.y_spec, axis=-1)
        y = tf.concat([y]*hp.train2.n_mixtures, axis=-1)

        centered_x = y - self.mu
        inv_stdv = tf.exp(-self.log_var)
        plus_in = inv_stdv * (centered_x + hp.train2.mol_step)
        min_in = inv_stdv * (centered_x - hp.train2.mol_step)
        cdf_plus = tf.sigmoid(plus_in)
        cdf_min = tf.sigmoid(min_in)

        # log probability for edge case
        log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
        log_one_minus_cdf_min = -tf.nn.softplus(min_in)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        log_prob = tf.where(y < 0.001, log_cdf_plus, tf.where(y > 0.999, log_one_minus_cdf_min, tf.log(tf.maximum(cdf_delta, 1e-12))))

        tf.summary.histogram('net2/train/prob', tf.exp(log_prob))

        log_prob = log_prob + self.log_pi

        tf.summary.histogram('net2/prob_max', tf.reduce_max(tf.exp(log_prob), axis=-1))

        log_prob = tf.reduce_logsumexp(log_prob, axis=-1)

        loss_mle = -tf.reduce_mean(log_prob)

        mean = tf.reduce_sum(self.mu * self.log_pi, axis=-1, keep_dims=True)
        loss_mix = tf.reduce_sum(self.log_pi * tf.squared_difference(self.mu, mean), axis=-1)
        loss_mix = -tf.reduce_mean(loss_mix)

        lamb = 0
        loss = loss_mle + lamb * loss_mix

        tf.summary.scalar('net2/train/loss_mle', loss_mle)
        tf.summary.scalar('net2/train/loss_mix', loss_mix)

        return loss

    def convert(self):
        argmax = tf.one_hot(tf.argmax(self.log_pi, axis=-1), hp.train2.n_mixtures)
        pred_spec = tf.reduce_sum(self.mu * argmax, axis=-1, name='pred_spec')
        return pred_spec
