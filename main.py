import tensorflow as tf
import numpy as np
import os
from lib.ops import *
from lib.model import *
import argparse
import scipy.misc
import time
#import collections

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir',help = 'the dir saving outputs',required = True)
parser.add_argument('--content_dir',help = 'the dir of data',required = True)
parser.add_argument('--style_dir',help = 'the dir of data')
parser.add_argument('--mode',help = 'train or inference',default = 'inference')
parser.add_argument('--normalizer',choices = ['in', 'bn'], default = 'in')
parser.add_argument('--content_layer', default = 'conv33')
parser.add_argument('--pre_trained',default = False,type = bool)
parser.add_argument('--checkpoint',help = 'the chkpt of saved model',default = None)
parser.add_argument('--vgg_ckpt',\
  help = 'checkpoint of vgg networks, the check point file of pretrained model should be downloaded',
  default = '/home/liaoqian/DATA/vgg19/vgg_19.ckpt'
)
parser.add_argument('--batch_size',help = 'size of a batch',default = 32,type = int)
parser.add_argument('--crop_size',help = 'size of a crop',default = 32,type = int)
parser.add_argument('--flip',help = ' ',default = True)
parser.add_argument('--name_queue_capacity',help = ' ',default = 256,type = int)
parser.add_argument('--image_queue_capacity',help = ' ',default = 256,type = int)
parser.add_argument('--queue_thread',help = ' ',default = 2,type = int)
parser.add_argument('--lr',help = 'learning rate',default = 1e-2,type = float)
parser.add_argument('--max_iter',help = 'the max iteration',default = 100000,type = int)
parser.add_argument('--decay_step',help = 'the frequency of summary',default = 100000,type = int)
parser.add_argument('--decay_rate',help = 'the frequency of summary',default = 0.1,type = float)
parser.add_argument('--summary_freq',help = 'the frequency of summary',default = 1,type = int)
parser.add_argument('--save_freq',help = 'the freq save output images',default = 1000,type =int)
parser.add_argument('--display_freq',help = 'the freq display losses',default = 1000,type = int)
parser.add_argument('--top_style_layer',default = 'conv54')

FLAGS = parser.parse_args()
print_configuration_op(FLAGS)

if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

log_dir = FLAGS.output_dir + 'log/'
    
if FLAGS.mode == 'train':
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    for _ in os.listdir(log_dir):
        os.remove(log_dir + _)
    
    data = data_loader(FLAGS)
    _, h, w, _ = data.style.shape
    #style = tf.placeholder(tf.float32,shape = [1, h, w, 3], name = 'style')
    Net = TransferNet(data.contents, data.style, FLAGS)
    with tf.name_scope('emit_output'):
        contents = deprocess(data.contents)    
        outputs = deprocess(Net.outputs)       # [-1, 1] -> [0, 1]
    with tf.name_scope('convert_tensor2image'):
        converted_outputs = tf.image.convert_image_dtype(outputs, \
                                                               dtype=tf.uint8, saturate=True)
        converted_contents = tf.image.convert_image_dtype(contents,\
                                                            dtype=tf.uint8, saturate=True)
        psnr = compute_psnr(converted_contents, converted_outputs)
        
        tf.summary.histogram('contents', contents)
        tf.summary.histogram('style', deprocess(data.style))
        tf.summary.histogram('outputs', outputs)
    
    tf.summary.scalar('0_gen_loss',Net.gen_loss) 
    tf.summary.scalar('1_content_loss',Net.content_loss)
    tf.summary.scalar('2_style_loss',Net.style_loss)
    tf.summary.scalar('3_tv_loss', Net.tv_loss)
    tf.summary.scalar('3_psnr',psnr)
    tf.summary.scalar('5_learning_rate',Net.learning_rate)
    tf.summary.image('0_images', deprocess(data.style))
    tf.summary.image('2_contents',converted_contents)
    tf.summary.image('1_outputs',converted_outputs)
        
    saver = tf.train.Saver(max_to_keep = 2)
    vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
    vgg_restore = tf.train.Saver(vgg_var_list)
    
    if FLAGS.pre_trained is True:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        weight_initiallizer = tf.train.Saver(var_list)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        vgg_restore.restore(sess, FLAGS.vgg_ckpt)
        print('VGG19 restored successfully!!')
        if (FLAGS.checkpoint is not None) and (FLAGS.pre_trained is True):
            print('Loading weights from the pre-trained model')
            weight_initiallizer.restore(sess, FLAGS.checkpoint)
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess, coord)
        print('Finish building the network!!!')
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)  
        train_op = Net.train_op
        start_time = time.time()
        print("Optimization starts !!!")
        for it in range(FLAGS.max_iter):
            fetches = {
                "train_op" : Net.train_op
            }
                
            if (it+1) % FLAGS.display_freq == 0 or (it + 1) in [20,100] :
                fetches["tv_loss"] = Net.tv_loss
                fetches["gen_loss"] = Net.gen_loss                  
                fetches["content_loss"] = Net.content_loss  
                fetches["style_loss"] = Net.style_loss
                fetches["learning_rate"] = Net.learning_rate
                
            if (it+1) % FLAGS.summary_freq == 0:
                fetches["summary"] = merged
            
            if (it + 1) % FLAGS.save_freq == 0 or (it + 1) in [20,100]:
                fetches["outputs"] = outputs
            
            results = sess.run(fetches)
            
            if (it+1) % FLAGS.summary_freq == 0:
                train_writer.add_summary(results["summary"], it+1)
            
            if (it+1) % FLAGS.display_freq == 0 or (it+1 in [20,100]):
                remaining = (time.time() - start_time)  * (FLAGS.max_iter - it) / (it+1)   
                print("progress global_step  %d  remaining %dm" % (it, remaining / 60))
                print("gen_loss", results["gen_loss"])
                print("tv_loss", results["tv_loss"])
                print("content_loss", results["content_loss"])
                print("style_loss", results["style_loss"])
                print("learning_rate", results['learning_rate'])
                
            if ((it+1) % FLAGS.save_freq) == 0  or (it + 1 in [20,100] ):
                total_time = time.time() - start_time
                gen_output = results['outputs']
                img_name = FLAGS.style_dir.split('/')[-1]
                im_name = img_name.split('.')[0]
                if FLAGS.batch_size == 1 :
                    gen_output = np.squeeze(gen_output)
                    print('shape : ',gen_output.shape)
                    scipy.misc.toimage(gen_output, cmin=0., cmax=1.0) \
                                    .save(FLAGS.output_dir + '%s_%i_%.4e_%.1f_%s.png'
                                          %(FLAGS.top_style_layer, it+1,results['style_loss'],\
                                            total_time, im_name))
                else:
                    for i in range(1):
                        print('shape : ',gen_output[i,:,:,:].shape)
                        scipy.misc.toimage(gen_output[i,:,:,:], cmin=0., cmax=1.0) \
                                        .save(FLAGS.output_dir + '%s_%i_%.4e_%.1f_%s_%d.png'
                                          %(FLAGS.top_style_layer, it+1,results['style_loss'],\
                                            total_time, im_name, i)) 
                        
                if ((it+1) % 10000) == 0:                
                    print('Save the checkpoint')
                    saver.save(sess, os.path.join(FLAGS.output_dir, 'model'), global_step=it+1)
        print('Optimization done!!!!!!!!!!!!')
            
elif FLAGS.mode == 'inference':
    '''
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    for _ in os.listdir(log_dir):
        os.remove(log_dir + _)
    '''
    
    # Check the checkpoint
    if FLAGS.checkpoint is None:
        raise ValueError('The checkpoint file is needed to performing the test.')
    if FLAGS.flip == True:
        FLAGS.flip = False
    
    inference_data = inference_data_loader(FLAGS)
    
    inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw') 
    
    with tf.variable_scope("transfer_net"):
        gen_output = transfer_net(inputs_raw, FLAGS, training = True)
        
    print('Finish building the network')
    with tf.name_scope('convert_image'):
        # Deprocess the images outputed from the model
        outputs = deprocess(gen_output) # [-1, 1] -> [0, 1]

    # tf.summary.histogram('features', features)

    
    # Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    weight_initiallizer = tf.train.Saver(var_list)

    # Define the initialization operation
    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        # Load the pretrained model
        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, FLAGS.checkpoint)
        print('Stylization starts!!')
        style_name = FLAGS.checkpoint.split('/')[1]
        for i in range(inference_data.image_count):
            print('Now Stylizing No.%i   %s' % (i + 1, inference_data.contents_names[i]))
            result = sess.run(outputs, feed_dict = {inputs_raw : inference_data.contents[i]})
            scipy.misc.toimage(np.squeeze(result), cmin=0, cmax=1.0) \
                      .save(os.path.join(FLAGS.output_dir,style_name + '_' + \
                                         inference_data.contents_names[i]))
        
        
        
         