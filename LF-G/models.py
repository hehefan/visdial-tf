import math
import numpy as np
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import EmbeddingWrapper
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import OutputProjectionWrapper
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
import tensorflow as tf


class Model(object):
  def __init__(self, imgSize, vocabSize, embedSize, use_lstm, rnnHiddenSize, rnnLayers, start, end, batch_size,
    learning_rate, learning_rate_decay_factor, min_learning_rate, training_steps_per_epoch, 
    keep_prob=0.5, max_gradient_norm=5.0, is_training=True):

    if is_training:
      self.global_step = tf.Variable(0, trainable=False)
      self.learning_rate = tf.maximum(
				tf.train.exponential_decay(
					learning_rate,
					self.global_step,
					training_steps_per_epoch,
					learning_rate_decay_factor,
					staircase=True),
        min_learning_rate)
      self.answers_ph = tf.placeholder(tf.int32, shape=[batch_size, 10, 20], name="answers")
      self.answer_lengths_ph = tf.placeholder(tf.int32, shape=[batch_size, 10], name="answer_lengths")
      self.targets_ph = tf.placeholder(tf.int32, shape=[batch_size, 10, 21], name="targets")

    self.image_feature_ph = tf.placeholder(tf.float32, shape=[batch_size, imgSize], name="image_feature")
    
    self.caption_ph = tf.placeholder(tf.int32, shape=[batch_size, 40], name="caption")
    self.caption_length_ph = tf.placeholder(tf.int32, shape=[batch_size], name="caption_length")

    self.questions_ph = tf.placeholder(tf.int32, shape=[batch_size, 10, 20], name="questions")
    self.question_lengths_ph = tf.placeholder(tf.int32, shape=[batch_size, 10], name="question_lengths")

    START = tf.constant(value=[start]*batch_size)
    END = tf.constant(value=[end]*batch_size)

    # Embedding (share)
    with ops.device("/cpu:0"):
      if vs.get_variable_scope().initializer:
        initializer = vs.get_variable_scope().initializer
      else:
        # Default initializer for embeddings should have variance=1.
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)
        embedding = vs.get_variable("embedding", [vocabSize, embedSize], initializer=initializer, dtype=tf.float32)

    START_EMB = embedding_ops.embedding_lookup(embedding, START)
    END_EMB = embedding_ops.embedding_lookup(embedding, END)

    # split placeholders and embed
    questions = tf.split(value=self.questions_ph, num_or_size_splits=10, axis=1)                # list with length 10; questions[0]: [batch_size, 1, 20]
    questions = [tf.squeeze(input=question, axis=1) for question in questions]                  # list with length 10; questions[0]: [batch_size, 20]
    questions = [embedding_ops.embedding_lookup(embedding, question) for question in questions] # list with length 10; questions[0]: [batch_size, 20, embedSize]

    question_lengths = tf.split(value=self.question_lengths_ph, num_or_size_splits=10, axis=1)
    question_lengths = [tf.squeeze(question_length) for question_length in question_lengths]

    if is_training:
      answers = tf.split(value=self.answers_ph, num_or_size_splits=10, axis=1)
      answers = [tf.squeeze(input=answer, axis=1) for answer in answers]
      answers = [embedding_ops.embedding_lookup(embedding, answer) for answer in answers]

      answer_lengths = tf.split(value=self.answer_lengths_ph, num_or_size_splits=10, axis=1)
      answer_lengths = [tf.squeeze(answer_length) for answer_length in answer_lengths]

      targets = tf.split(value=self.targets_ph, num_or_size_splits=10, axis=1)
      targets = [tf.squeeze(input=target, axis=1) for target in targets]

      weights = []
      for r in range(10):
        weight = []
        answer_length = answer_lengths[r]
        for i in range(21):
          weight.append(tf.greater_equal(x=answer_length, y=i))
        weight = tf.cast(x=tf.stack(values=weight, axis=1), dtype=tf.float32)                   # [batch_size, 21]
        weights.append(weight)

    # make RNN cell
    def single_cell():
      return GRUCell(rnnHiddenSize)
    if use_lstm:
      def single_cell():
        return BasicLSTMCell(rnnHiddenSize, state_is_tuple=False)

    make_cell = single_cell
    if rnnLayers > 1:
      def make_cell():
        return MultiRNNCell([single_cell() for _ in range(rnnLayers)], state_is_tuple=False)

    encoder_cell = make_cell()
    decoder_cell = OutputProjectionWrapper(cell=make_cell(), output_size=vocabSize, activation=None)

    # caption feature
    caption = embedding_ops.embedding_lookup(embedding, self.caption_ph)                        # [batch_size, 40, embedSize]
    caption_length = tf.squeeze(self.caption_length_ph)
    with tf.variable_scope('EncoderRNN') as varscope:
      _, captionState = dynamic_rnn(cell=encoder_cell, inputs=caption, sequence_length=caption_length, dtype=tf.float32, scope=varscope)       # [batch_size, encoder_cell.state_size]

    if is_training:
      losses = []
    else:
      ans_word_probs = []

    for r in range(10):
      # 1. question
      with tf.variable_scope('EncoderRNN', reuse=True) as varscope:
        _, questionState = dynamic_rnn(cell=encoder_cell, inputs=questions[r], sequence_length=question_lengths[r], dtype=tf.float32, scope=varscope)
      
      # 2. history
      if r == 0:
        historyState = captionState
      
      # 3. fusion
      concat = tf.concat(values=[self.image_feature_ph, questionState, historyState], axis=1)
      if is_training:
        concat = tf.nn.dropout(x=concat, keep_prob=keep_prob)
      with tf.variable_scope('Fusion', reuse=(r>0)) as varscope:
        encoder_state = tf.contrib.layers.fully_connected(inputs=concat, num_outputs=decoder_cell.state_size, activation_fn=tf.nn.tanh, scope=varscope)

      # 4. decoder
      with tf.variable_scope('DecoderRNN', reuse=(r>0)) as varscope:
        if is_training:
          answer = [tf.squeeze(input=word, axis=1) for word in tf.split(value=answers[r], num_or_size_splits=20, axis=1)]
          decoder_outputs, _ = rnn_decoder(decoder_inputs=[START_EMB]+answer, initial_state=encoder_state, cell=decoder_cell, loop_function=None, scope=varscope)
        else:
          self_answer = []
          self_answer_emb = []
          def loop_function(prev, _):
            prev_symbol = math_ops.argmax(prev, 1)
            self_answer.append(tf.cast(x=prev_symbol, dtype=tf.int32))
            emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
            self_answer_emb.append(emb_prev)
            return emb_prev
          decoder_outputs, _ = rnn_decoder(decoder_inputs=[START_EMB]*21, initial_state=encoder_state, cell=decoder_cell, loop_function=loop_function, scope=varscope)

      # 5. update history
      with tf.variable_scope('EncoderRNN', reuse=True) as varscope:
        _, historyState = dynamic_rnn(cell=encoder_cell, inputs=questions[r], sequence_length=question_lengths[r], initial_state=historyState, scope=varscope)
        if is_training:
          _, historyState = dynamic_rnn(cell=encoder_cell, inputs=answers[r], sequence_length=answer_lengths[r], initial_state=historyState, scope=varscope)
        else:
          self_answer = tf.stack(values=self_answer+[END], axis=1)                              # [batch_size, 21]
          self_answer_length = tf.argmax(input=tf.cast(x=tf.equal(x=self_answer, y=end), dtype=tf.float32), axis=1)
          self_answer_emb = tf.stack(values=self_answer_emb, axis=1)                            # [batch_size, 20, embSize]
          _, historyState = dynamic_rnn(cell=encoder_cell, inputs=self_answer_emb, sequence_length=self_answer_length, initial_state=historyState, scope=varscope)
      
      if is_training:
        decoder_outputs = tf.stack(values=decoder_outputs, axis=1)                              # [batch_size, 21, vocabSize]
        loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_outputs, targets=targets[r], weights=weights[r], average_across_batch=False)     # [batch_size]
        losses.append(loss)
      else:
        decoder_outputs = [tf.log(tf.nn.softmax(decoder_output)) for decoder_output in decoder_outputs]
        ans_word_probs.append(tf.stack(values=decoder_outputs, axis=1))                         # [batch_size, 21, vocabSize]
    if is_training:
      losses = tf.stack(values=losses, axis=1)                                                  # [batch_size, 10]
      self.loss = tf.reduce_mean(losses)
      params = tf.trainable_variables()
      gradients = tf.gradients(self.loss, params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.opt_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    else:
      self.ans_word_probs = tf.stack(values=ans_word_probs, axis=1)                             # [batch_size, 10, 21, vocabSize]

    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)

