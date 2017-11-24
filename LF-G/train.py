import os
import sys
import tensorflow as tf

import random
import math
import numpy as np
from opts import args
from dataloader import DataLoader
from models import Model

# seed for reproducibility
tf.set_random_seed(1234)

# ==============================================================================
#                               Loading dataset
# ==============================================================================
dataloader = DataLoader(args, 'train')
# creating the directory to save the model

if not os.path.exists(args.savePath):
  os.makedirs(args.savePath)

# Iterations per epoch
numIterPerEpoch = int(math.ceil(dataloader.num_dialogues/args.batchSize))*args.LRateDecay

print('%d iter per epoch.\n'%numIterPerEpoch)

# ==============================================================================
#                               Build Graph
# ==============================================================================

model = Model(imgSize=args.imgFeatureSize,
              vocabSize=dataloader.vocabSize,
              embedSize=args.embedSize,
              use_lstm=args.useLSTM,
              rnnHiddenSize=args.rnnHiddenSize,
              rnnLayers=args.numLayers,
              start=dataloader.word2ind['<START>'],
              end=dataloader.word2ind['<END>'],
              batch_size=args.batchSize,
              learning_rate=args.learningRate,
              learning_rate_decay_factor=args.lrDecayRate, 
              min_learning_rate=args.minLRate, 
              training_steps_per_epoch=numIterPerEpoch, 
              keep_prob=args.dropout, 
              max_gradient_norm=args.clipNorm,
              is_training=True)

with tf.Session() as sess:
  ckpt = tf.train.get_checkpoint_state(args.savePath)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(sess, ckpt.model_checkpoint_path)
    step = int(ckpt.model_checkpoint_path.split('-')[1])
  else:
    sess.run(tf.global_variables_initializer())
    step = 0

  idxs = [i for i in range(dataloader.num_dialogues)]

  for epoch in range(1, args.numEpochs+1):
    random.shuffle(idxs)
    for batch_id, (start, end) in enumerate(zip(range(0, dataloader.num_dialogues, args.batchSize), range(args.batchSize, dataloader.num_dialogues, args.batchSize))):
      batchImgFeat, batchQuestion, batchQuestionLength, batchCaption, batchCaptionLength, batchAnswer, batchAnswerLength, batchTarget = dataloader.getTrainBatch(idxs[start:end])
      input_feed = { model.image_feature_ph:batchImgFeat,
          model.questions_ph:batchQuestion,
          model.question_lengths_ph:batchQuestionLength,
          model.caption_ph:batchCaption,
          model.caption_length_ph:batchCaptionLength,
          model.answers_ph:batchAnswer,
          model.answer_lengths_ph:batchAnswerLength,
          model.targets_ph:batchTarget}

      output_feed = [model.loss, model.opt_op]

      outputs = sess.run(output_feed, input_feed)
      step += 1
      print ('%5d:\t%d-%d:\t%f'%(step, epoch, batch_id+1, outputs[0]))
      sys.stdout.flush()
      if (step % args.ckptInterval) == 0:
        checkpoint_path = os.path.join(args.savePath, "visdial")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
