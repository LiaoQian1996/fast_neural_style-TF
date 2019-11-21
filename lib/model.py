from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from lib.ops import *
from functools import partial
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
import collections
    
# Definition of the generator and discriminator
conv = partial(slim.conv2d, activation_fn = None, \
               weights_initializer = tf.truncated_normal_initializer(stddev=0.02)
              )
deconv = partial(slim.conv2d_transpose, activation_fn=None, \
                weights_initializer=tf.random_normal_initializer(stddev=0.02)
              ) 

relu = tf.nn.relu
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, \
                     updates_collections=tf.GraphKeys.UPDATE_OPS)
instance_norm = partial(layers.instance_norm, center = True, scale = True, epsilon = 1e-6)
crop = partial(tf.image.crop_to_bounding_box, offset_height = 0, offset_width = 0)
'''
# CHECK LOSS BY OPMIZATION METHOD
def transfer_net(content, FLAGS, reuse = False, training = True):
    shape = [FLAGS.batch_size, 256,256,3]
    var = tf.get_variable('gen_img',shape = shape, \
                  initializer = tf.random_normal_initializer(0, 0.5), \
                               dtype=tf.float32,trainable=True, collections=None)   
    return tf.tanh(var)    
'''    
def transfer_net(content, FLAGS, reuse = False, training = True):
    
    bn = partial(batch_norm, is_training = training) 
    insn = partial(instance_norm)  
    if FLAGS.normalizer == 'in':
        normalizer = insn
    elif FLAGS.normalizer == 'bn':
        normalizer = bn
        
    conv_norm_relu = partial(conv, normalizer_fn = normalizer, \
                                   activation_fn = relu, biases_initializer = None)
    deconv_norm_relu = partial(deconv, normalizer_fn = normalizer, \
                              activation_fn = relu, biases_initializer = None)
    
    def residual_block(input, name_scope):
        with tf.variable_scope(name_scope):
            net = conv_norm_relu(input, 128, 3, 1, 'VALID')
            net = conv(net, 128, 3, 1, 'VALID')
            net = normalizer(net)
            _, h, w, _ = input.get_shape().as_list() 
            # we should crop the input tensor to conduct element-wise addition
            return input[:, 2:-2, 2:-2, :] + net
            
    with tf.variable_scope('conv_stage'):
        net = tf.pad(content, 
                     [[0,0],[40,40],[40,40],[0,0]], # [n, 256, 256, 3] -> [n, 336, 336, 3]
                     "REFLECT")  
        
        net = conv_norm_relu(net, 32, 9, 1)       # [n, 336, 336, 3] -> [n, 336, 336, 32]
        net = conv_norm_relu(net, 64, 3, 2)       # [n, 336, 336, 32] -> [n, 168, 168, 32]
        net = conv_norm_relu(net, 128, 3, 2)      # [n, 168, 168, 32] -> [n, 84, 84, 64]
    
    with tf.variable_scope('residual_stage'):
        
        for i in range(1, 5 + 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, name_scope)
    
    with tf.variable_scope('deconv_stage'):
        net = deconv_norm_relu(net, 64, 3, 2)   # [n, 64, 64, 128] -> [n, 128, 128, 64]
        net = deconv_norm_relu(net, 32, 3, 2)   # [n, 128, 128, 64] -> [n, 256, 256, 32]
        #net = deconv_norm_relu(net, 3, 9, 1)    # [n, 256, 256 ,32] -> [n, 256, 256, 3]
        net = tf.tanh(conv(net, 3, 9, 1))
        return net
        
def TransferNet(content, style, FLAGS):
    Network = collections.namedtuple('Network', 'content_loss, style_loss, gen_loss, tv_loss, \
                                     train_op, outputs, \
                                     contents, learning_rate')         
    
    with tf.variable_scope("transfer_net"):
        output = transfer_net(content, FLAGS)
        
    with tf.name_scope('tv_loss'):
        tv_loss = total_variation_loss(output)        
    
    with tf.name_scope("content_loss"):
        _, output_feature_maps = vgg_19(output, is_training = False, reuse = False)
        _, content_feature_maps = vgg_19(content, is_training = False, reuse = True)
        content_layer = get_layer_scope(FLAGS.content_layer) 
        content_feature_map  = content_feature_maps[content_layer]
        output_feature_map = output_feature_maps[content_layer]
        content_loss = tf.reduce_mean(tf.square( content_feature_map - output_feature_map ))
    
    with tf.name_scope('style_loss'):
        _, style_feature_maps = vgg_19(style, is_training=False, reuse=True)
        style_layer_list = ['conv11','conv21','conv31','conv41','conv51','conv54'] 
        sl = tf.zeros([])
        ratio_list = [1.0, 1.0, 1.0, 0.001, 1.0, 10.0]
        
        def subtract_feature_mean(feature):
            mean = tf.reduce_mean(tf.reduce_mean(feature, axis = 1, keep_dims = True),\
                                  keep_dims = True, axis=1)
            return feature - mean
        
        for i in range(len(style_layer_list)):
            tar_layer = style_layer_list[i]
            target_layer = get_layer_scope(tar_layer)
            output_feature_map = output_feature_maps[target_layer]
            output_feature_map = subtract_feature_mean(output_feature_map)
            style_feature_map = style_feature_maps[target_layer]
            style_feature_map = subtract_feature_mean(style_feature_map) 
            sl = sl + compute_style_loss(output_feature_map, style_feature_map) * ratio_list[i]
        style_loss = sl

    gen_loss = 1e-4 * content_loss + style_loss + 1e-3 * tv_loss
    
    with tf.variable_scope('get_learning_rate_and_global_step'):  
        global_step = tf.train.get_or_create_global_step()
        le_rate = tf.train.exponential_decay(FLAGS.lr,global_step, FLAGS.decay_step, \
                                             FLAGS.decay_rate,True)
        incr_global_step = tf.assign(global_step, global_step + 1)     
        
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='transfer_net')
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(le_rate).minimize(gen_loss, global_step = global_step,\
                                                     var_list = var_list)
    
    return Network(
        learning_rate = le_rate,
        gen_loss = gen_loss,
        content_loss = content_loss,
        style_loss = style_loss,
        tv_loss = tv_loss,
        train_op = train_op,
        contents = content,
        outputs = output
    )    