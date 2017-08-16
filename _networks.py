# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
from modules import *
from prepro import load_vocab


def encode(inputs, is_training=True, scope="encoder", reuse=None):
    '''
    Args:
      inputs: A 2d tensor with shape of [N, T], dtype of int32.
      seqlens: A 1d tensor with shape of [N,], dtype of int32.
      masks: A 3d tensor with shape of [N, T, 1], dtype of float32.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A collection of Hidden vectors, whose shape is (N, T, E).
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Load vocabulary 
        char2idx, idx2char = load_vocab()
        
        # Character Embedding
        inputs = embed(inputs, len(char2idx), hp.embed_size) # (N, T, E)  
        
        # Encoder pre-net
        prenet_out = prenet(inputs, is_training=is_training) # (N, T, E/2)
        
        # Encoder CBHG 
        ## Conv1D bank 
        enc = conv1d_banks(prenet_out, K=hp.encoder_num_banks, is_training=is_training) # (N, T, K * E / 2)
        
        ### Max pooling
        enc = tf.layers.max_pooling1d(enc, 2, 1, padding="same")  # (N, T, K * E / 2)
          
        ### Conv1D projections
        enc = conv1d(enc, hp.embed_size//2, 3, scope="conv1d_1") # (N, T, E/2)
        enc = normalize(enc, type=hp.norm_type, is_training=is_training, 
                            activation_fn=tf.nn.relu, scope="norm1")
        enc = conv1d(enc, hp.embed_size//2, 3, scope="conv1d_2") # (N, T, E/2)
        enc = normalize(enc, type=hp.norm_type, is_training=is_training, 
                            activation_fn=None, scope="norm2")
        enc += prenet_out # (N, T, E/2) # residual connections
          
        ### Highway Nets
        for i in range(hp.num_highwaynet_blocks):
            enc = highwaynet(enc, num_units=hp.embed_size//2, 
                                 scope='highwaynet_{}'.format(i)) # (N, T, E/2)

        ### Bidirectional GRU
        memory = gru(enc, hp.embed_size//2, True, seqlens=None) # (N, T, E)
    
    return memory
        
def decode1(decoder_inputs, memory, seqlens=None, is_training=True, scope="decoder1", reuse=None):
    '''
    Args:
      decoder_inputs: A 3d tensor with shape of [N, T', C'], where C'=hp.n_mels*hp.r, 
        dtype of float32. Shifted melspectrogram of sound files. 
      memory: A 3d tensor with shape of [N, T, C], where C=hp.embed_size.
      seqlens: A 1d tensor with shape of [N,], dtype of int32.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted melspectrogram tensor with shape of [N, T', C'].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        dec = prenet(decoder_inputs, is_training=is_training) # (N, T', E/2)
        
        # Attention RNN
        dec = attention_decoder(dec, memory, seqlens=None, num_units=hp.embed_size) # (N, T', E)

        # Decoder RNNs
        dec += gru(dec, hp.embed_size, False, scope="decoder_gru1") # (N, T', E)
        dec += gru(dec, hp.embed_size, False, scope="decoder_gru2") # (N, T', E)
          
        # Outputs => (N, T', hp.n_mels*hp.r)
        out_dim = decoder_inputs.get_shape().as_list()[-1]
        outputs = tf.layers.dense(dec, out_dim) 
    
    return outputs

def decode2(inputs, is_training=True, scope="decoder2", reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, T'', C''], where C''=hp.n_mels, 
        dtype of float32. Log magnitude spectrogram of sound files.
      is_training: Whether or not the layer is in training mode.  
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted magnitude spectrogram tensor with shape of [N, T'', C_], 
        where C_ = 1+hp.n_fft//2.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        prenet_out = prenet(inputs, is_training=is_training) # (N, T'', E/2)
        
        # Decoder Post-processing net = CBHG
        ## Conv1D bank
        dec = conv1d_banks(prenet_out, K=hp.decoder_num_banks, is_training=is_training) # (N, T', E*K/2)
         
        ## Max pooling
        dec = tf.layers.max_pooling1d(dec, 2, 1, padding="same") # (N, T', E*K/2)
         
        ## Conv1D projections
        dec = conv1d(dec, hp.embed_size, 3, scope="conv1d_1") # (N, T', E)
        dec = normalize(dec, type=hp.norm_type, is_training=is_training, 
                            activation_fn=tf.nn.relu, scope="norm1")
        dec = conv1d(dec, hp.embed_size//2, 3, scope="conv1d_2") # (N, T', E/2)
        dec = normalize(dec, type=hp.norm_type, is_training=is_training, 
                            activation_fn=None, scope="norm2")
        dec += prenet_out
         
        ## Highway Nets
        for i in range(4):
            dec = highwaynet(dec, num_units=hp.embed_size//2, 
                                 scope='highwaynet_{}'.format(i)) # (N, T, E/2)
         
        ## Bidirectional GRU    
        dec = gru(dec, hp.embed_size//2, True) # (N, T', E)  
        
        # Outputs => (N, T', (1+hp.n_fft//2)*hp.r)
        out_dim = (1+hp.n_fft//2)*hp.r
        outputs = tf.layers.dense(dec, out_dim)
    
    return outputs
