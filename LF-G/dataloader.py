import numpy as np
import h5py
import utils
from opts import args
import tensorflow as tf

class DataLoader(object):
  '''
    read the data
    params: object itself, command line options, subset of data to load (train, val, test)
  '''
  def __init__(self, opt, subset):
    '''
      subsets: 'train' or 'val'
    '''
    # read additional info like dictionary, etc
    print('DataLoader loading json file: %s'% opt.inputJson)
    info = utils.readJSON(opt.inputJson)

    # add <START> and <END> to vocabulary
    self.word2ind = info['word2ind']
    self.vocabSize = len(info['word2ind'])
    print('Vocabulary size (with <PAD>, <START> and <END>): %d'%self.vocabSize)

    # construct ind2word
    ind2word = {}
    for word, ind in info['word2ind'].iteritems():
      ind2word[ind] = word
    self.ind2word = ind2word

    # read questions, answers and options
    print('DataLoader loading h5 file: %s'%opt.inputQues)
    quesFile = h5py.File(opt.inputQues, 'r')

    print('DataLoader loading h5 file: %s'%opt.inputImg)
    imgFile = h5py.File(opt.inputImg, 'r')

    # read question related information
    questions = np.array(quesFile['ques_'+subset])                          # 82783 * 10 * 20
    question_lengths = np.array(quesFile['ques_length_'+subset])            # 82783 * 10

    options = np.array(quesFile['opt_'+subset])                             # (82783, 10, 100)
    option_lengths = np.array(quesFile['opt_length_'+subset])               # (252298,)
    option_list = np.array(quesFile['opt_list_'+subset])                    # (252298, 20)


    answers = np.array(quesFile['ans_'+subset])                             # 82783 * 10 * 20
    answer_lengths = np.array(quesFile['ans_length_'+subset])               # 82783 * 10
    answer_ids = np.array(quesFile['ans_index_'+subset])                    # 82783 * 10

    captions = np.array(quesFile['cap_'+subset])                            # (82783, 40)
    caption_lengths = np.array(quesFile['cap_length_'+subset])              # (82783,)

    print('DataLoader loading h5 file: %s'%opt.inputImg)
    imgFile = h5py.File(opt.inputImg, 'r')
    print('Reading image features..')
    imgFeats = np.array(imgFile['images_'+subset+'_1'])

    # Normalize the image features (if needed)
    if opt.imgNorm:
      print('Normalizing image features..')
      imgFeats = imgFeats / np.expand_dims(a=np.linalg.norm(x=imgFeats,axis=1), axis=1)

    # done reading, close files
    quesFile.close()
    imgFile.close()

    # print information for data type
    self.num_dialogues = questions.shape[0]
    self.num_rounds = questions.shape[1]
    self.caption_max_length = captions.shape[1]
    self.question_max_length = questions.shape[2]
    self.answer_max_length = option_list.shape[1]

    print('\n%s:\n\tNo. of dialogues: %d\n\tNo. of rounds: %d\n\tMax length of captions: %d\n\tMax length of questions: %d\n\tMax length of answers: %d\n'%(subset, self.num_dialogues, self.num_rounds, self.caption_max_length, self.question_max_length, self.answer_max_length))

    self.imgFeats = imgFeats

    self.captions = captions
    self.caption_lengths = caption_lengths

    self.questions = questions
    self.question_lengths = question_lengths

    self.answers = answers
    self.answer_lengths = answer_lengths
    self.answer_ids = answer_ids

    targets = np.zeros((self.num_dialogues,10,21), dtype=np.int)
    for d in range(self.num_dialogues):
      for r in range(10):
        targets[d][r] = np.insert(answers[d][r], answer_lengths[d][r], self.word2ind['<END>'])
    self.targets = targets

    self.options = options
    self.option_list = option_list
    self.option_lengths = option_lengths

  def getTrainBatch(self, batch_idx):
    batchImgFeat, batchQuestion, batchQuestionLength, batchCaption, batchCaptionLength, batchAnswer, batchAnswerLength, batchTarget = [], [], [], [], [], [], [], []
    for idx in batch_idx:
      batchImgFeat.append(self.imgFeats[idx])

      batchQuestion.append(self.questions[idx])
      batchQuestionLength.append(self.question_lengths[idx])

      batchCaption.append(self.captions[idx])
      batchCaptionLength.append(self.caption_lengths[idx])

      batchAnswer.append(self.answers[idx])
      batchAnswerLength.append(self.answer_lengths[idx])
      batchTarget.append(self.targets[idx])

    return np.array(batchImgFeat), np.array(batchQuestion), np.array(batchQuestionLength), np.array(batchCaption), np.array(batchCaptionLength), np.array(batchAnswer), np.array(batchAnswerLength), np.array(batchTarget)


  def getTestBatch(self, batch_idx):
    batchImgFeat, batchQuestion, batchQuestionLength, batchCaption, batchCaptionLength, batchAnswerID, batchOptions, batchOptionLengths = [], [], [], [], [], [], [], []
    for idx in batch_idx:
      batchImgFeat.append(self.imgFeats[idx])

      batchQuestion.append(self.questions[idx])
      batchQuestionLength.append(self.question_lengths[idx])

      batchCaption.append(self.captions[idx])
      batchCaptionLength.append(self.caption_lengths[idx])

      batchAnswerID.append(self.answer_ids[idx])

      options = np.zeros(shape=(10, 100, 20), dtype=int)
      option_lengths = np.zeros(shape=(10, 100), dtype=int)
      option_ids = self.options[idx]   # [10, 100]
      for r in range(10):
        for o in range(100):
          i = option_ids[r][o]
          options[r][o] = self.option_list[i]
          option_lengths[r][o] = self.option_lengths[i]
      batchOptions.append(options)
      batchOptionLengths.append(option_lengths)

    return np.array(batchImgFeat), np.array(batchQuestion), np.array(batchQuestionLength), np.array(batchCaption), np.array(batchCaptionLength), np.array(batchAnswerID), np.array(batchOptions), np.array(batchOptionLengths)


