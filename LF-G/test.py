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
dataloader = DataLoader(args, 'val')

args.batchSize = 100

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
              learning_rate=None,
              learning_rate_decay_factor=None, 
              min_learning_rate=None, 
              training_steps_per_epoch=None, 
              keep_prob=None, 
              max_gradient_norm=None,
              is_training=False)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5

with tf.Session(config=config) as sess:
  start_step = 8800
  idxs = [i for i in range(dataloader.num_dialogues)]
  MRR, Rank1, Rank5, Rank10, Mean = 0.0, 0.0, 0.0, 0.0, 1000.0
  for step in range(start_step, 100000000, args.ckptInterval):
    checkpoint_path = os.path.join(args.savePath, "visdial-%d"%step)
    if not os.path.exists(checkpoint_path+'.index'):
      exit(0)
    model.saver.restore(sess, checkpoint_path)
    mrr, rank1, rank5, rank10, mean, numRecords = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for batch_id, (start, end) in enumerate(zip(range(0, dataloader.num_dialogues, args.batchSize), range(args.batchSize, dataloader.num_dialogues, args.batchSize))):
      batchImgFeat, batchQuestion, batchQuestionLength, batchCaption, batchCaptionLength, batchAnswerID, batchOptions, batchOptionLengths = dataloader.getTestBatch(idxs[start:end])
      if batchAnswerID.shape[0] < args.batchSize:
        break
      input_feed = { model.image_feature_ph:batchImgFeat,
          model.questions_ph:batchQuestion,
          model.question_lengths_ph:batchQuestionLength,
          model.caption_ph:batchCaption,
          model.caption_length_ph:batchCaptionLength}
      ans_word_probs = sess.run(model.ans_word_probs, input_feed)           # [batch_size, 10, 21, vocabSize]
      labels = batchAnswerID
      for b in range(args.batchSize):
        for r in range(10):
          option = batchOptions[b][r]                                       # (100, 20) 
          option_length = batchOptionLengths[b][r]                          # (100)
          word_probs = ans_word_probs[b][r]                                 # (21, 8836)
          scores = []
          for opt, opt_len in zip(option, option_length):
            score = 0.0
            for l in range(opt_len):
              score += word_probs[l][opt[l]]
            score += word_probs[opt_len][dataloader.word2ind['<END>']]
            scores.append(score)
          prediction = np.argsort(scores)[::-1]
          lbl = labels[b][r]
          if prediction[0] == lbl:
            rank1 += 1
          if lbl in prediction[:5]:
            rank5 += 1
          if lbl in prediction[:10]:
            rank10 += 1
          idx = np.argwhere(prediction==lbl)[0][0] + 1.0
          mrr += 1.0/idx
          mean += idx
      numRecords += 10*args.batchSize
    mrr /= numRecords
    rank1 /= numRecords
    rank5 /= numRecords
    rank10 /= numRecords
    mean /= numRecords
    print '%5d:'%step
    print '\t%f\t%f\t%f\t%f\t%f'%(mrr, rank1, rank5, rank10, mean)
    if MRR < mrr:
      MRR = mrr 
    if Rank1 < rank1:
      Rank1 = rank1
    if Rank5 < rank5:
      Rank5 = rank5
    if Rank10 < rank10:
      Rank10 = rank10
    if Mean > mean:
      Mean = mean
    print '\t%f\t%f\t%f\t%f\t%f\n'%(MRR, Rank1, Rank5, Rank10, Mean)
    sys.stdout.flush() 

