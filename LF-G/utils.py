# script containing supporting code/methods
import json
import numpy as np
# right align the question tokens in 3d volume
def rightAlign(sequences, lengths):
  '''
  clone the sequences
    sequences: 82783 * 10 * 20
    lengths: 82783 * 10
  '''
  numDims = sequences.ndim
  numShape = sequences.shape
  numImgs = numShape[0]
  rAligned = np.zeros(numShape, dtype=int)

  if numDims == 3:
    M = numShape[2] # maximum length of question
    maxCount = numShape[1] # number of questions/image
    for imId in range(numImgs):
      for quesId in range(maxCount):
        # do only for non zero sequence counts
        if lengths[imId][quesId] == 0:
          break
        # copy based on the sequence length
        length = lengths[imId][quesId]
        sequence = sequences[imId][quesId]
        rAligned[imId][quesId][M - length:] = sequence[:length]
  elif numDims == 2:
    # handle 2 dimensional matrices as well
    M = numShape[1] # maximum length of question
    for imId in range(numImgs):
      # do only for non zero sequence counts
      length = lengths[imId]
      sequence = sequences[imId]
      if length > 0:
        # copy based on the sequence length
        rAligned[imId][M - length:] = sequence[:length]

  return rAligned

# read a json file and python dictionary
def readJSON(fileName):
  with open(fileName, 'r') as f:
    jsonFile = json.load(f)
  return jsonFile
