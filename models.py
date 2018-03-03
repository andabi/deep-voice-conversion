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


class Net1(ModelDesc):
    def __init__(self):
        pass

    def _get_inputs(self):
        return [InputDesc(tf.float32, (hp.train1.batch_size, None, hp.default.n_mfcc), 'x_mfccs'),
                InputDesc(tf.int32, (hp.train1.batch_size, None, ), 'y_ppgs')]

    def _build_graph(self, inputs):
        self.x_mfccs, self.y_ppgs = inputs
        self.is_training = get_current_tower_context().is_training
        with tf.variable_scope('net1'):
            self.ppgs, self.preds, self.logits = self.network(self.x_mfccs)
        self.cost = self.loss()

        # summaries
        tf.summary.scalar('net1/train/loss', self.cost)
        tf.summary.scalar('net1/train/acc', self.acc())

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=hp.train1.lr, trainable=False)
        return tf.train.AdamOptimizer(lr)

    @auto_reuse_variable_scope
    def network(self, x_mfcc):
        # Pre-net
        prenet_out = prenet(x_mfcc,
                            num_units=[hp.train1.hidden_units, hp.train1.hidden_units // 2],
                            dropout_rate=hp.train1.dropout_rate,
                            is_training=self.is_training)  # (N, T, E/2)

        # CBHG
        out = cbhg(prenet_out, hp.train1.num_banks, hp.train1.hidden_units // 2,
                   hp.train1.num_highway_blocks, hp.train1.norm_type, self.is_training)

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
                InputDesc(tf.float32, (hp.train2.batch_size, None, hp.default.n_mels), 'y_mel'),
                InputDesc(tf.float32, (hp.train2.batch_size, None, hp.default.n_fft//2 + 1), 'y_spec')]

    def _build_graph(self, inputs):
        self.x_mfcc, self.y_mel, self.y_spec = inputs

        # build net1
        with tf.variable_scope('net1'):
            self.ppgs, _, _ = self.net1.network(self.x_mfcc)

        # build net2
        with tf.variable_scope('net2'):
            self.pred_spec_mu, self.pred_spec_phi = self.network(self.ppgs)

        self.cost = self.loss()

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=hp.train2.lr, trainable=False)
        return tf.train.AdamOptimizer(lr)

    @auto_reuse_variable_scope
    def network(self, ppgs):
        # Pre-net
        prenet_out = prenet(ppgs,
                            num_units=[hp.train2.hidden_units, hp.train2.hidden_units // 2],
                            dropout_rate=hp.train2.dropout_rate,
                            is_training=self.is_training)  # (N, T, E/2)

        # CBHG1: mel-scale
        # pred_mel = cbhg(prenet_out, hp.train2.num_banks, hp.train2.hidden_units // 2,
        #                 hp.train2.num_highway_blocks, hp.train2.norm_type, self.is_training,
        #                 scope="cbhg_mel")
        # pred_mel = tf.layers.dense(pred_mel, self.y_mel.shape[-1])  # (N, T, n_mels)
        pred_mel = prenet_out

        # CBHG2: linear-scale
        pred_spec = tf.layers.dense(pred_mel, hp.train2.hidden_units // 2)  # (N, T, n_mels)
        pred_spec = cbhg(pred_spec, hp.train2.num_banks, hp.train2.hidden_units // 2,
                         hp.train2.num_highway_blocks, hp.train2.norm_type, self.is_training,
                         scope="cbhg_linear")
        pred_spec = tf.layers.dense(pred_spec, self.y_spec.shape[-1])  # (N, T, 1+hp.n_fft//2)
        pred_spec = tf.expand_dims(pred_spec, axis=-1)
        pred_spec_mu = tf.layers.dense(pred_spec, hp.train2.n_mixtures)  # (N, T, 1+hp.n_fft//2, n_mixtures)
        pred_spec_phi = tf.nn.softmax(tf.layers.dense(pred_spec, hp.train2.n_mixtures))  # (N, T, 1+hp.n_fft//2, n_mixtures)

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


# class Model:
#     def __init__(self, mode, batch_size):
#         self.mode = mode
#         self.batch_size = batch_size
#         self.is_training = self.get_is_training(mode)
#
#         # Networks
#         self.net_template = tf.make_template('net', self._net2)
#         self.ppgs, self.pred_ppg, self.logits_ppg, self.pred_spec_mu, self.pred_spec_phi = self.net_template()
#
#     def __call__(self):
#         return self.pred_spec_mu
#
#     def _get_inputs(self):
#         length = hp.signal.duration * hp.signal.sr
#         length_spec = length // hp.signal.hop_length + 1
#         return [InputDesc(tf.float32, (None, length), 'wav'),
#                 InputDesc(tf.float32, (None, length_spec, hp.signal.n_mels), 'x'),
#                 InputDesc(tf.int32, (None,), 'speaker_id')]
#
#     def get_input(self, mode, batch_size, queue):
#         '''
#         mode: A string. One of the phases below:
#           `train1`: TIMIT TRAIN waveform -> mfccs (inputs) -> PGGs -> phones (target) (ce loss)
#           `test1`: TIMIT TEST waveform -> mfccs (inputs) -> PGGs -> phones (target) (accuracy)
#           `train2`: ARCTIC SLT waveform -> mfccs -> PGGs (inputs) -> spectrogram (target)(l2 loss)
#           `test2`: ARCTIC SLT waveform -> mfccs -> PGGs (inputs) -> spectrogram (target)(accuracy)
#           `convert`: ARCTIC BDL waveform -> mfccs (inputs) -> PGGs -> spectrogram -> waveform (output)
#         '''
#         if mode not in ('train1', 'test1', 'train2', 'test2', 'convert'):
#             raise Exception("invalid mode={}".format(mode))
#
#         x_mfcc = tf.placeholder(tf.float32, shape=(batch_size, None, hp.default.n_mfcc))
#         y_ppgs = tf.placeholder(tf.int32, shape=(batch_size, None,))
#         y_spec = tf.placeholder(tf.float32, shape=(batch_size, None, 1 + hp.default.n_fft // 2))
#         y_mel = tf.placeholder(tf.float32, shape=(batch_size, None, hp.default.n_mels))
#         num_batch = 1
#
#         if queue:
#             if mode in ("train1", "test1"):  # x: mfccs (N, T, n_mfccs), y: Phones (N, T)
#                 x_mfcc, y_ppgs, num_batch = get_batch_queue(mode=mode, batch_size=batch_size)
#             elif mode in ("train2", "test2", "convert"):  # x: mfccs (N, T, n_mfccs), y: spectrogram (N, T, 1+n_fft//2)
#                 x_mfcc, y_spec, y_mel, num_batch = get_batch_queue(mode=mode, batch_size=batch_size)
#         return x_mfcc, y_ppgs, y_spec, y_mel, num_batch
#
#     def get_is_training(self, mode):
#         if mode in ('train1', 'train2'):
#             is_training = True
#         else:
#             is_training = False
#         return is_training
#
#     def _net1(self):
#         with tf.variable_scope('net1'):
#             # Pre-net
#             prenet_out = prenet(self.x_mfcc,
#                                 num_units=[hp.train1.hidden_units, hp.train1.hidden_units // 2],
#                                 dropout_rate=hp.train1.dropout_rate,
#                                 is_training=self.is_training)  # (N, T, E/2)
#
#             # CBHG
#             out = cbhg(prenet_out, hp.train1.num_banks, hp.train1.hidden_units // 2,
#                        hp.train1.num_highway_blocks, hp.train1.norm_type, self.is_training)
#
#             # Final linear projection
#             logits = tf.layers.dense(out, len(phns))  # (N, T, V)
#             ppgs = tf.nn.softmax(logits / hp.train1.t)  # (N, T, V)
#             preds = tf.to_int32(tf.arg_max(logits, dimension=-1))  # (N, T)
#
#         return ppgs, preds, logits
#
#     def loss_net1(self):
#         istarget = tf.sign(tf.abs(tf.reduce_sum(self.x_mfcc, -1)))  # indicator: (N, T)
#         loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_ppg / hp.train1.t,
#                                                               labels=self.y_ppg)
#         loss *= istarget
#         loss = tf.reduce_mean(loss)
#         return loss
#
#     def acc_net1(self):
#         istarget = tf.sign(tf.abs(tf.reduce_sum(self.x_mfcc, -1)))  # indicator: (N, T)
#         num_hits = tf.reduce_sum(tf.to_float(tf.equal(self.pred_ppg, self.y_ppg)) * istarget)
#         num_targets = tf.reduce_sum(istarget)
#         acc = num_hits / num_targets
#         return acc
#
#     def _net2(self):
#         # PPGs from net1
#         ppgs, preds_ppg, logits_ppg = self._net1()
#
#         with tf.variable_scope('net2'):
#             # Pre-net
#             prenet_out = prenet(ppgs,
#                                 num_units=[hp.train2.hidden_units, hp.train2.hidden_units // 2],
#                                 dropout_rate=hp.train2.dropout_rate,
#                                 is_training=self.is_training)  # (N, T, E/2)
#
#             # CBHG1: mel-scale
#             # pred_mel = cbhg(prenet_out, hp.train2.num_banks, hp.train2.hidden_units // 2,
#             #                 hp.train2.num_highway_blocks, hp.train2.norm_type, self.is_training,
#             #                 scope="cbhg_mel")
#             # pred_mel = tf.layers.dense(pred_mel, self.y_mel.shape[-1])  # (N, T, n_mels)
#             pred_mel = prenet_out
#
#             # CBHG2: linear-scale
#             pred_spec = tf.layers.dense(pred_mel, hp.train2.hidden_units // 2)  # (N, T, n_mels)
#             pred_spec = cbhg(pred_spec, hp.train2.num_banks, hp.train2.hidden_units // 2,
#                              hp.train2.num_highway_blocks, hp.train2.norm_type, self.is_training,
#                              scope="cbhg_linear")
#             pred_spec = tf.layers.dense(pred_spec, self.y_spec.shape[-1])  # (N, T, 1+hp.n_fft//2)
#             pred_spec = tf.expand_dims(pred_spec, axis=-1)
#             pred_spec_mu = tf.layers.dense(pred_spec, hp.train2.n_mixtures)  # (N, T, 1+hp.n_fft//2, n_mixtures)
#             pred_spec_phi = tf.nn.softmax(tf.layers.dense(pred_spec, hp.train2.n_mixtures))  # (N, T, 1+hp.n_fft//2, n_mixtures)
#
#         return ppgs, preds_ppg, logits_ppg, pred_spec_mu, pred_spec_phi
#
#     def loss_net2(self):
#         # negative log likelihood
#         normal_dists = []
#         for i in range(hp.train2.n_mixtures):
#             mu = self.pred_spec_mu[..., i]
#             normal_dist = distributions.Logistic(mu, tf.ones_like(mu))
#             normal_dists.append(normal_dist)
#         cat = distributions.Categorical(probs=self.pred_spec_phi)
#         mixture_dist = distributions.Mixture(cat=cat, components=normal_dists)
#         prob = mixture_dist.cdf(value=self.y_spec + hp.train2.mol_step) - \
#                mixture_dist.cdf(value=self.y_spec - hp.train2.mol_step)
#         prob /= hp.train2.mol_step * 2
#         self.prob_min = tf.reduce_min(prob)
#         loss = -tf.reduce_mean(tf.log(prob + 1e-8))
#         return loss
#
#     # def loss_net2(self):
#     #     mol_step = 1e-3
#     #     # negative log likelihood
#     #     # normal_dists = []
#     #     # for i in range(hp.train2.n_mixtures):
#     #     # mu = self.pred_spec_mu[..., i]
#     #     mu = self.pred_spec_mu
#     #     normal_dist = distributions.Logistic(mu, tf.ones_like(mu))
#     #     # normal_dists.append(normal_dist)
#     #     prob = normal_dist.cdf(value=self.y_spec + mol_step) - normal_dist.cdf(value=self.y_spec - mol_step)
#     #     prob /= mol_step * 2
#     #     prob = tf.reduce_sum(prob * self.pred_spec_phi, axis=-1)
#     #     # cat = distributions.Categorical(probs=self.pred_spec_phi)
#     #     # mixture_dist = distributions.Mixture(cat=cat, components=normal_dists)
#     #     # log_likelihood = tf.reduce_sum(mixture_dist.log_prob(value=self.y_spec))
#     #     loss = -tf.reduce_mean(tf.log(prob))
#     #     return loss
#
#     @staticmethod
#     def load(sess, mode, logdir, logdir2=None, step=None):
#         def print_model_loaded(mode, logdir, step):
#             model_name = Model.get_model_name(logdir, step=step)
#             print('Model loaded. mode: {}, model_name: {}'.format(mode, model_name))
#
#         if mode in ['train1', 'test1']:
#             var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1')
#             if Model._load_variables(sess, logdir, var_list=var_list, step=step):
#                 print_model_loaded(mode, logdir, step)
#
#         elif mode == 'train2':
#             var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1')
#             if Model._load_variables(sess, logdir, var_list=var_list1, step=step):
#                 print_model_loaded(mode, logdir, step)
#
#             var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net2')
#             if Model._load_variables(sess, logdir2, var_list=var_list2, step=step):
#                 print_model_loaded(mode, logdir2, step)
#
#         elif mode in ['test2', 'convert']:
#             if Model._load_variables(sess, logdir, var_list=None, step=step):  # Load all variables
#                 print_model_loaded(mode, logdir, step)
#
#     @staticmethod
#     def _load_variables(sess, logdir, var_list, step=None):
#         model_name = Model.get_model_name(logdir, step)
#         if model_name:
#             ckpt = os.path.join(logdir, model_name)
#             tf.train.Saver(var_list=var_list).restore(sess, ckpt)
#             return True
#         else:
#             return False
#
#     @staticmethod
#     def get_model_name(logdir, step=None):
#         model_name = None
#         if step:
#             paths = glob.glob('{}/*step_{}.index*'.format(logdir, step))
#             if paths:
#                 _, model_name, _ = split_path(paths[0])
#         else:
#             ckpt = tf.train.latest_checkpoint(logdir)
#             if ckpt:
#                 _, model_name = os.path.split(ckpt)
#         return model_name
#
#     @staticmethod
#     def get_epoch_and_global_step(logdir, step=None):
#         model_name = Model.get_model_name(logdir, step)
#         if model_name:
#             tokens = model_name.split('_')
#             epoch, gs = int(tokens[1]), int(tokens[3])
#         else:
#             epoch = gs = 0
#         return epoch, gs
#
#     @staticmethod
#     def all_model_names(logdir):
#         path = '{}/*.meta'.format(logdir)
#         model_names = map(lambda f: os.path.basename(f).replace('.meta', ''), glob.glob(path))
#         return model_names
