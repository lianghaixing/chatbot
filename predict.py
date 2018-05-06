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
    './model_old',
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
tf.app.flags.DEFINE_string(
    'result_path',
    'result/result',
    '测试结果'
)

FLAGS = tf.app.flags.FLAGS

# 设置桶的长度
_buckets = [(1, 10), (10, 15), (20, 25), (40, 50),(60,70)]


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


def test():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    enc_vocab_path = os.path.join(FLAGS.data_directory,"vocab%d.enc" % FLAGS.enc_vocab_size)
    dec_vocab_path = os.path.join(FLAGS.data_directory,"vocab%d.dec" % FLAGS.dec_vocab_size)

    enc_vocab, _ = prepareData.initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = prepareData.initialize_vocabulary(dec_vocab_path)

    sys.stdout.write("我> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      sentence = sentence.strip('\n')
      token_ids = prepareData.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)

      bucket_id = min([b for b in range(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])

      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if prepareData.EOS_ID in outputs:
        outputs = outputs[:outputs.index(prepareData.EOS_ID)]

      result = "".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])
      print("AI> " + result)
      print("我> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()

def test2():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    enc_vocab_path = os.path.join(FLAGS.data_directory,"vocab%d.enc" % FLAGS.enc_vocab_size)
    dec_vocab_path = os.path.join(FLAGS.data_directory,"vocab%d.dec" % FLAGS.dec_vocab_size)

    enc_vocab, _ = prepareData.initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = prepareData.initialize_vocabulary(dec_vocab_path)

    test_data_path = os.path.join(FLAGS.data_directory,"test.enc")

    with open(test_data_path, 'r', encoding = 'utf-8') as f:
      for line in f.readlines():
        #ask_sentence = line.strip()x_list = x.split(' ')
        x_list = line.split(' ')
        sentence = "".join(x_list)
        #sentence = line.strip(' ')
        token_ids = prepareData.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)
        bucket_id = min([b for b in range(len(_buckets)) if _buckets[b][0] > len(token_ids)])

        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if prepareData.EOS_ID in outputs:
          outputs = outputs[:outputs.index(prepareData.EOS_ID)]

        result = "".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])
        with open(FLAGS.result_path, 'a', encoding = 'utf-8') as ff:
          ff.write("ask: " + sentence + "\n")
          ff.write("answer: " + result + "\n")
          ff.write("\n")
        #print("AI> " + result)
        #print("我> ", end="")
      


if __name__ == '__main__':
  print("开始测试……")
  test2()
