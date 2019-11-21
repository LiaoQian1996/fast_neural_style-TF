import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow.contrib.slim as slim
import scipy.misc as sic
import numpy as np
import collections 
import math
import os

def total_variation_loss(image):
    tv_y_size = tf.size(image[:,1:,:,:],out_type=tf.float32)
    tv_x_size = tf.size(image[:,:,1:,:],out_type=tf.float32)
    tv_loss =   (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:-1,:]) /
                    tv_x_size))
    return tv_loss

def get_layer_scope(layer):
    target_layer = 'vgg_19/conv' + layer[-2] + '/conv' + layer[-2] + '_' + layer[-1]  
    return target_layer

def gram(features):
    # _, h, w, c = map(lambda i: i.value, feature.get_shape())
    features = tf.reshape(features,[-1,features.shape[3]])
    return tf.matmul(features,features,transpose_a=True) / \
           tf.cast(features.shape[1]*features.shape[0], dtype=tf.float32)

def compute_style_loss(feature1,feature2):
    return tf.reduce_mean(tf.reduce_sum(tf.square(gram(feature1)-gram(feature2)), axis=0))

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1
    
def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2
    
def compute_psnr(ref, content):
    ref = tf.cast(ref, tf.float32)
    content = tf.cast(content, tf.float32)
    diff = content - ref
    sqr = tf.multiply(diff, diff)
    err = tf.reduce_sum(sqr)
    v = tf.shape(diff)[0] * tf.shape(diff)[1] * tf.shape(diff)[2] * tf.shape(diff)[3]
    mse = err / tf.cast(v, tf.float32)
    psnr = 10. * (tf.log(255. * 255. / mse) / tf.log(10.))
    return psnr

def get_variable(name,shape):  # 给全连接层用的
    xavier = tf.contrib.layers.xavier_initializer()
    zeros = tf.zeros_initializer()
    if name == 'w':
        return tf.get_variable(name,shape,initializer=xavier)
    else:
        return tf.get_variable(name,shape,initializer=zeros)
    
def data_loader(FLAGS):
    with tf.device('/cpu:0'):
        Data = collections.namedtuple('Data', 'contents, style, image_count')
        with tf.name_scope('load_style_image'):
            image_raw = Image.open(FLAGS.style_dir)
            if image_raw.mode is not 'RGB':
                image_raw = image_raw.convert('RGB')
            image_raw = np.asarray(image_raw)/255
            targets = tf.constant(image_raw)  
            targets = tf.image.convert_image_dtype(targets, dtype = tf.float32, saturate = True)
            targets = preprocess(targets)  
            style_image = tf.expand_dims(targets, axis=0) 

        image_list_tar = os.listdir(FLAGS.content_dir)    
        image_list_tar = [_ for _ in image_list_tar if _.endswith('.jpg')]
        if len(image_list_tar)==0:
            raise Exception('No png files in the input directory !')
        image_list_tar = [os.path.join(FLAGS.content_dir, _) for _ in image_list_tar]
        
        with tf.variable_scope('load_content_image'):
            filename_queue = tf.train.string_input_producer(image_list_tar,
                                                   shuffle=False, capacity=FLAGS.name_queue_capacity)
            print('filename_queue : ',filename_queue)
            reader = tf.WholeFileReader()
            key,value = reader.read(filename_queue)
            input_image_tar = tf.image.decode_png(value, channels=3)
            input_image_tar = tf.image.convert_image_dtype(input_image_tar, dtype=tf.float32)
            assertion = tf.assert_equal(tf.shape(input_image_tar)[2], 3, message="image does not have 3 channels")
            with tf.control_dependencies([assertion]):
                input_image_tar = tf.identity(input_image_tar)

            # Normalize the low resolution image to [0, 1], high resolution to [-1, 1]
            contents = tf.identity(input_image_tar) 
        # The data augmentation part
        with tf.name_scope('data_preprocessing'):
            with tf.name_scope('random_crop'):
                print('[Config] Use random crop')   
                input_size = tf.shape(contents)
                h, w = tf.cast(input_size[0], tf.float32),\
                tf.cast(input_size[1], tf.float32)
                offset_w = tf.cast(tf.floor(tf.random_uniform([], 0, w - FLAGS.crop_size)),
                                   dtype=tf.int32)
                offset_h = tf.cast(tf.floor(tf.random_uniform([], 0, h - FLAGS.crop_size)),
                                   dtype=tf.int32)
                contents = tf.image.crop_to_bounding_box(contents, offset_h, offset_w, FLAGS.crop_size,FLAGS.crop_size)  
                
            with tf.variable_scope('random_flip'):
                # Check for random flip:
                if (FLAGS.flip is True) and (FLAGS.mode == 'train'):
                    print('[Config] Use random flip')
                    # Produce the decision of random flip
                    decision = tf.random_uniform([], 0, 1, dtype=tf.float32)
                    content_images = random_flip(contents, decision)
                else:
                    content_images = tf.identity(contents)
            
            content_images = preprocess(contents)
                
        contents_batch = tf.train.batch([content_images],\
                        batch_size = FLAGS.batch_size,\
                        capacity = FLAGS.image_queue_capacity + 4 * FLAGS.batch_size,\
                        num_threads=FLAGS.queue_thread) 
        #steps_per_epoch = int(math.ceil(len(image_list_tar) / FLAGS.batch_size))                     

        return Data(
            style = style_image,
            contents = contents_batch,
            image_count=len(image_list_tar),
        ) 

def inference_data_loader(FLAGS):
    Data = collections.namedtuple('Data', 'contents, contents_names, image_count')    
        
    content_list = os.listdir(FLAGS.content_dir)
    content_list = [_ for _ in content_list if (_.split('.')[-1] == 'png') or \
                                             (_.split('.')[-1] == 'jpg') ]

    #filelist.sort()
    imgs = []
    img_names = []
    for _ in content_list:
        img = np.array(Image.open(FLAGS.content_dir + _))
        img = preprocess(img / 255)
        imgs.append(np.expand_dims(img, 0))
        img_name = _.split('/')[-1]
        # img_name = _.split('.')[0]
        img_names.append(img_name)

    return Data(
        contents = imgs,
        contents_names = img_names,
        image_count = len(imgs)
    )
    
def random_flip(input, decision):
    f1 = tf.identity(input)
    f2 = tf.image.flip_left_right(input)
    output = tf.cond(tf.less(decision, 0.5), lambda: f2, lambda: f1)
    return output

def plot(samples,n,FLAGS):
    fig = plt.figure(figsize=(n, n))
    gs = gridspec.GridSpec(n, n)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(FLAGS.crop_size, FLAGS.crop_size, 3))

    return fig
    
def print_configuration_op(FLAGS):
    print('[Configurations]:')
    FLAGS = vars(FLAGS)
    for name, value in sorted(FLAGS.items()):
        if type(value) == float:
            print('\t%s: %f'%(name, value))
        elif type(value) == int:
            print('\t%s: %d'%(name, value))
        elif type(value) == str:
            print('\t%s: %s'%(name, value))
        elif type(value) == bool:
            print('\t%s: %s'%(name, value))
        else:
            print('\t%s: %s' % (name, value))
    print('End of configuration')

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# VGG19 component
def vgg_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.
    Args:
    weight_decay: The l2 regularization coefficient.
    Returns:
    An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc

# VGG19 net
"""Oxford Net VGG 19-Layers version E Example.
Note: All the fully_connected layers have been transformed to conv2d layers.
    To use in classification mode, resize input to 224x224.
Args:
inputs: a tensor of size [batch_size, height, width, channels].
num_classes: number of predicted classes.
is_training: whether or not the model is being trained.
dropout_keep_prob: the probability that activations are kept in the dropout
  layers during training.
spatial_squeeze: whether or not should squeeze the spatial dimensions of the
  outputs. Useful to remove unnecessary dimensions for classification.
scope: Optional scope for the variables.
fc_conv_padding: the type of padding to use for the fully connected layer
  that is implemented as a convolutional layer. Use 'SAME' padding if you
  are applying the network in a fully convolutional manner and want to
  get a prediction map downsampled by a factor of 32 as an output. Otherwise,
  the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.
Returns:
the last op containing the log predictions and end_points dict.
"""
def vgg_19(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           reuse = False,
           fc_conv_padding='VALID'):
    
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2',reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4',reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5',reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    return net,end_points
vgg_19.default_image_size = 224
