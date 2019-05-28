# coding: utf-8
import tensorflow as tf
from tensorflow.contrib import rnn
from util import *
from dataset_util import *
from collections import OrderedDict

INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 100
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
MOMENTUM = 0.9

def conv(images, num_class, is_training):
  
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = tf.get_variable(name='weights', shape=[3, 3, 3, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16))
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable(name='biases', shape=[64], initializer=tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
#     conv1 = tf.cond(pred=is_training,
#                 true_fn=lambda: tf.nn.dropout(conv1, keep_prob=0.8),
#                 false_fn=lambda: conv1,
#                 name='cnn_dropout_conv1')

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='pool1')
    
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = tf.get_variable(name='weights',shape=[3, 3, 64, 128],initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16))
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable(name='biases', shape=[128], initializer=tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
#     conv2 = tf.cond(pred=is_training,
#                 true_fn=lambda: tf.nn.dropout(conv2, keep_prob=0.8),
#                 false_fn=lambda: conv2,
#                 name='cnn_dropout_conv2')

  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='VALID', name='pool2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = tf.get_variable(name='weights',shape=[3, 3, 128, 256],initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16))
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable(name='biases', shape=[256], initializer=tf.constant_initializer(0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)
    conv3 = tf.cond(pred=is_training,
                true_fn=lambda: tf.nn.dropout(conv3, keep_prob=0.5),
                false_fn=lambda: conv3,
                name='cnn_dropout_conv3')
    
  # conv3-1
  with tf.variable_scope('conv31') as scope:
    kernel = tf.get_variable(name='weights',shape=[3, 3, 256, 256],initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16))
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable(name='biases', shape=[256], initializer=tf.constant_initializer(0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv31 = tf.nn.relu(pre_activation, name=scope.name)
    conv31 = tf.cond(pred=is_training,
                true_fn=lambda: tf.nn.dropout(conv31, keep_prob=0.5),
                false_fn=lambda: conv31,
                name='cnn_dropout_conv31')  
    
    
    
  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = tf.get_variable(name='weights',shape=[3, 3, 256, 256],initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16))
    conv = tf.nn.conv2d(conv31, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable(name='biases', shape=[256], initializer=tf.constant_initializer(0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(pre_activation, name=scope.name)
    conv4 = tf.cond(pred=is_training,
                true_fn=lambda: tf.nn.dropout(conv4, keep_prob=0.8),
                false_fn=lambda: conv4,
                name='cnn_dropout_conv4')
  
  # pool3
  pool3 = tf.nn.max_pool(conv4, ksize=[1, 2, 1, 1],
                         strides=[1, 2, 1, 1], padding='VALID', name='pool2')


  # conv5
  with tf.variable_scope('conv5') as scope:
    kernel = tf.get_variable(name='weights',shape=[3, 3, 256, 512],initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16))
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable(name='biases', shape=[512], initializer=tf.constant_initializer(0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(pre_activation, name=scope.name)
  
  # bn1
  with tf.variable_scope('bn1') as scope:
    mean, variance = tf.nn.moments(conv5, list(range(len(conv5.shape[:-1]))), name='moments')
    epsilon = 1e-05
    beta = tf.get_variable('beta', conv5.shape[-1:], initializer=tf.zeros_initializer)
    gamma = tf.get_variable('gamma', conv5.shape[-1:], initializer=tf.ones_initializer)
    bn1 = tf.nn.batch_normalization(conv5, mean, variance, beta, gamma, epsilon)
  
  # conv6  
  with tf.variable_scope('conv6') as scope:
    kernel = tf.get_variable(name='weights',shape=[3, 3, 512, 512],initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16))
    conv = tf.nn.conv2d(bn1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable(name='biases', shape=[512], initializer=tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv6 = tf.nn.relu(pre_activation, name=scope.name)
  
  # bn2
  with tf.variable_scope('bn2') as scope:
    mean, variance = tf.nn.moments(conv6, list(range(len(conv6.shape[:-1]))), name='moments')
    epsilon = 1e-05
    beta = tf.get_variable('beta', conv6.shape[-1:], initializer=tf.zeros_initializer)
    gamma = tf.get_variable('gamma', conv6.shape[-1:], initializer=tf.ones_initializer)
    bn2 = tf.nn.batch_normalization(conv6, mean, variance, beta, gamma, epsilon) 

  # pool3
  pool4 = tf.nn.max_pool(bn2, ksize=[1, 2, 1, 1],
                         strides=[1, 2, 1, 1], padding='VALID', name='pool2')

  # conv7  
  with tf.variable_scope('conv7') as scope:
    kernel = tf.get_variable(name='weights',shape=[2, 2, 512, 512],initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16))
    conv = tf.nn.conv2d(bn1, kernel, [1, 3, 1, 1], padding='VALID')
    biases = tf.get_variable(name='biases', shape=[512], initializer=tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv7 = tf.nn.relu(pre_activation, name=scope.name)
    conv7 = tf.cond(pred=is_training,
                true_fn=lambda: tf.nn.dropout(conv7, keep_prob=0.5),
                false_fn=lambda: conv7,
                name='cnn_dropout_conv7')
   
  # map to sequence 1*36*512
  with tf.variable_scope('map_to_sequence'):
    shape = conv7.get_shape().as_list()
    assert shape[1] == 1  # H of the feature map must equal to 1
    seq = tf.squeeze(input=conv7, axis=1, name='squeeze')
    seq_len = seq.shape[1] * np.ones(100)    

  return seq, seq_len

def lstm(seq, seq_len, num_class, is_training):
    # bid-lstm
  with tf.variable_scope('bid-lstm'):
    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(256, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(256, forget_bias=1.0, state_is_tuple=True)
    cell_fw = [lstm_fw_cell]*2
    cell_bw = [lstm_bw_cell]*2
    outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                cell_fw, cell_bw, seq,
                dtype=tf.float32)
    
#     cell = tf.contrib.rnn.LSTMCell(512, state_is_tuple=True)
#     stack = tf.contrib.rnn.MultiRNNCell([cell] * 1, state_is_tuple=True)
#     outputs, _ = tf.nn.dynamic_rnn(cell, seq,  dtype=tf.float32)
    outputs = tf.cond(pred=is_training,
                true_fn=lambda: tf.nn.dropout(outputs, keep_prob=0.4),
                false_fn=lambda: outputs,
                name='lstm_dropout')
    
    batch_size = 100
    output = tf.reshape(outputs, [-1, 256 * 2])
    W = tf.get_variable(name='weights',shape=[256 * 2, num_class],initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16))
    b = tf.get_variable(name='biases', shape=[num_class],initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(output, W) + b
    
#     logits = tf.cond(pred=is_training,
#                 true_fn=lambda: tf.nn.dropout(logits, keep_prob=0.8),
#                 false_fn=lambda: logits,
#                 name='lstm_dropout_logits')
    
    #[batch_size,max_timesteps,num_classes]
    logits = tf.reshape(logits, [batch_size, -1, num_class])
    
    raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')
    
    #转置矩阵，第0和第1列互换位置=>[max_timesteps,batch_size,num_classes]
    logits = tf.transpose(logits, (1, 0, 2))
    return logits, raw_pred, seq_len

def compute_loss(logits, labels, seq_len):
    
  ctc_loss = tf.nn.ctc_loss(labels=labels,inputs=logits, sequence_length=seq_len)  
  loss = tf.reduce_mean(ctc_loss, name='ctc_loss')

  return loss


    
    
def train(index, num_class, num_epochs=1):
    tf.reset_default_graph()
    print_every=800
    
    training_accuracy=OrderedDict()
    validation_accuracy=OrderedDict()
    test_accuracy=OrderedDict()
    loss_re=OrderedDict()

    
    with tf.device("/gpu:0"):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                                global_step,
                                                DECAY_STEPS,
                                                LEARNING_RATE_DECAY_FACTOR,
                                                staircase=True)
        images = tf.placeholder(tf.float32, [None, 32, 150, 3])
        labels = tf.sparse_placeholder(tf.int32)
        is_training = tf.placeholder(tf.bool)
        seq, seq_len = conv(images, num_class, is_training)
        
        logits, raw_pred, seq_len = lstm(seq, seq_len, num_class, is_training)
        loss = compute_loss(logits, labels, seq_len)
        
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
            
    def check_accuracy(t, sess, datatype, images,decoded,log_prob, acc):
        num_correct, num_samples = 0, 0
        dataset = batch_data(datatype)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        sess.run(iterator.initializer)
        while True:
                try:
                    image, label = sess.run(next_element)
                    label = [l.decode() for l in label]
                    label = transferlabeltoInt(label)
                    feed_dict = {images: image, labels: label, is_training: False}
                    y_pred,log_probs, accuracy = sess.run([decoded[0],log_prob, acc], feed_dict=feed_dict)
                    
                    
                    y_pred = decode_sparse_tensor(y_pred)
                    y_batch = decode_sparse_tensor(label)
         
                    y_pred =[''.join(el) for el in y_pred]
                    y_batch =[''.join(el) for el in y_batch]
                     
                    num_samples += image.shape[0]
                    for idx, stri in enumerate(y_batch):
                        if(len(y_pred)<idx):
                            continue
                        detect_number = y_pred[idx]
                        if(stri == detect_number):
                            num_correct += 1
                except tf.errors.OutOfRangeError:
                    break  
        acc = float(num_correct) / num_samples
        
        
        print('Model in %s dataset got %d / %d correct (%.2f%%)' % (datatype, num_correct, num_samples, 100 * acc))
        return acc*100
    
    with tf.Session(config = tf.ConfigProto(allow_soft_placement = True)) as sess:
        saver = tf.train.Saver()
        t = 0
        if(index == 0):
            sess.run(tf.global_variables_initializer())
        elif(index == 1):
        ##
        
            training_accuracy, validation_accuracy, loss_re = merge_record_data()
       ##
       
            ckpt = tf.train.get_checkpoint_state('saved_model/')
            t = int(ckpt.model_checkpoint_path.split('-')[-1])*800
            print(t)
            saver.restore(sess, ckpt.model_checkpoint_path)
            check_accuracy(t, sess, 'validation', images, decoded,log_prob, acc)
            t+=1
        else:
            ckpt = tf.train.get_checkpoint_state('saved_model/')
            t = int(ckpt.model_checkpoint_path.split('-')[-1])*800
            print(t)
            saver.restore(sess, ckpt.model_checkpoint_path)
            check_accuracy(t, sess, 'training', images, decoded,log_prob, acc)
            check_accuracy(t, sess, 'validation', images, decoded,log_prob, acc)
            check_accuracy(t, sess, 'test', images, decoded,log_prob, acc)
            return 0
            
        dataset = batch_data( 'training', num_epoch=num_epochs, batchsize=100)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        sess.run(iterator.initializer)
        while True:
               try:
                   image, label = sess.run(next_element)
                   label = [l.decode() for l in label]
                   label = transferlabeltoInt(label)
                   feed_dict = {images: image, labels: label, is_training: True}
                   loss_np,b_logits, _,ded,accuray = sess.run([loss, logits, optimizer, decoded,acc], feed_dict=feed_dict)
                

                        
                   if t % print_every == 0:
                    epoch = t // 800
                    print('Starting epoch %d' % epoch)
                    print('Iteration %d, loss = %.4f, accuary = %.4f' % (t, loss_np, accuray))
                       
                    loss_re[t]=loss_np
           
                    acc1 = check_accuracy(t, sess, 'validation', images, decoded,log_prob, acc)
                    validation_accuracy[t]=acc1
                    
                    if(t%8000==0):
                        acc3 = check_accuracy(t, sess, 'training', images, decoded,log_prob, acc)
                        training_accuracy[t]=acc3
                        
                       
                    if(acc1-99 > 0):
                        acc2 = check_accuracy(t, sess, 'test', images, decoded,log_prob, acc)
                        if(acc2-99 > 0):
                            print('model train is successful')
                            
                            if(t%8000!=0):
                                acc3 = check_accuracy(t, sess, 'training', images, decoded,log_prob, acc)
                                training_accuracy[t]=acc3
                            
                            saver.save(sess=sess, save_path='saved_model/crnn_model', global_step=epoch)
                            record_training_data(training_accuracy, validation_accuracy, loss_re)
                            return 1
                        
                    if(epoch%10==0 and epoch!=0):
                        saver.save(sess=sess, save_path='saved_model/crnn_model', global_step=epoch)
                        
                        record_training_data(training_accuracy, validation_accuracy, loss_re)
                    
                    print()
              
                   t+=1
               except tf.errors.OutOfRangeError:
                print('model train is finished!!')
                return -1