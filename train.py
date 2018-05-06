# -*- coding:utf-8 -*-
# train模式
import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

import prepareData
import seq2seq_model

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
 
 # 定义参数   
tf.app.flags.DEFINE_float(
    'learning_rate',
    0.5,
    '学习率'
)
tf.app.flags.DEFINE_float(
    'max_gradient_norm',
    5.0,
    '梯度最大阈值'
)
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor',
    0.99,
    '衰减学习速率'
)
tf.app.flags.DEFINE_integer(
    'batch_size',
    64,
    '批量梯度下降的批量大小'
)
tf.app.flags.DEFINE_integer(
    'layer_size',
    256,
    'LSTM每层神经元数量'
)
tf.app.flags.DEFINE_integer(
    'num_layers',
    3,
    'LSTM的层数'
)
tf.app.flags.DEFINE_integer(
    'num_samples',
    512,
    '分批softmax的样本量'
)
tf.app.flags.DEFINE_integer(
    'max_train_data_size',
    50000,
    '最大训练集的大小'
)
tf.app.flags.DEFINE_integer(
    'enc_vocab_size',
    50000,
    '最大编码词典大小'
)
tf.app.flags.DEFINE_integer(
    'dec_vocab_size',
    50000,
    '最大解码词典大小'
)
tf.app.flags.DEFINE_integer(
    'steps_per_checkpoint',
    500,
    '训练多少步后保存模型'
)
tf.app.flags.DEFINE_string(
    'data_directory',
    './data',
    '数据集文件夹'
)
tf.app.flags.DEFINE_string(
    'working_directory',
    './model',
    '模型保存的目录'
)
tf.app.flags.DEFINE_string(
    'train_vector_enc',
    'data/train.enc.ids50000',
    '训练集的encoder输入'
)
tf.app.flags.DEFINE_string(
    'train_vector_dec',
    'data/train.dec.ids50000',
    '训练集的decoder输入'
)
tf.app.flags.DEFINE_string(
    'test_vector_enc',
    'data/test.enc.ids50000',
    '测试集的encoder输入'
)
tf.app.flags.DEFINE_string(
    'test_vector_dec',
    'data/test.dec.ids50000',
    '测试集的decoder输入'
)

FLAGS = tf.app.flags.FLAGS

# 设置桶的长度
_buckets = [(1, 10), (10, 15), (20, 25), (40, 50)]


def read_data(source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(prepareData.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set

def create_model(session, forward_only):
  model = seq2seq_model.Seq2SeqModel(
    FLAGS.enc_vocab_size,
    FLAGS.dec_vocab_size,
    _buckets,
    FLAGS.layer_size,
    FLAGS.num_layers, 
    FLAGS.max_gradient_norm,
    FLAGS.batch_size,
    FLAGS.learning_rate,
    FLAGS.learning_rate_decay_factor,
    forward_only=forward_only)

  ckpt = tf.train.get_checkpoint_state(FLAGS.working_directory)

  if ckpt and ckpt.model_checkpoint_path:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def train():
  print("数据集文件夹: %s" % FLAGS.data_directory)

  enc_train = FLAGS.train_vector_enc
  dec_train = FLAGS.train_vector_dec
  enc_dev = FLAGS.test_vector_enc
  dec_dev = FLAGS.test_vector_dec
  
 
  # setup config to use BFC allocator
  config = tf.ConfigProto()  
  config.gpu_options.allocator_type = 'BFC'

  with tf.Session(config=config) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.layer_size))
    model = create_model(sess, False)

    dev_set = read_data(enc_dev, dec_dev)
    train_set = read_data(enc_train, dec_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]

    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in range(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      if current_step % 10 == 0:
        perplexity = math.exp(step_loss)
        print ("step= %d learning rate= %.4f perplexity= %.2f" % (model.global_step.eval(), model.learning_rate.eval(), perplexity))

      if current_step % FLAGS.steps_per_checkpoint == 0:
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        
        checkpoint_path = os.path.join(FLAGS.working_directory, "seq2seq.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        
        for bucket_id in range(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()

if __name__ == '__main__':
  print("开始训练……")
  train()
