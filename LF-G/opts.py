import argparse

parser = argparse.ArgumentParser(description='Train the Visual Dialog model')
# Data input settings
parser.add_argument('--inputImg', type=str,  default='../data/vgg_img.h5', help='HDF5 file with image features')
parser.add_argument('--inputQues', type=str,  default='../data/visdial_data.h5', help='HDF5 file with preprocessed questions')
parser.add_argument('--inputJson', type=str,  default='../data/visdial_params.json', help='JSON file with info and vocab')
parser.add_argument('--savePath', type=str,  default='checkpoints/', help='Path to save checkpoints')

parser.add_argument('--imgNorm', type=bool,  default=True, help='normalize the image feature.')

# model params
parser.add_argument('--imgFeatureSize', type=int,  default=4096, help='Size of the image feature');
parser.add_argument('--embedSize', type=int,  default=300, help='Size of input word embeddings')
parser.add_argument('--useLSTM', type=bool,  default=True, help='LSTM or GRU')
parser.add_argument('--rnnHiddenSize', type=int,  default=256, help='Size of the LSTM state')
parser.add_argument('--numLayers', type=int,  default=1, help='Number of layers in LSTM')
# optimization params
parser.add_argument('--batchSize', type=int,  default=32, help='Batch size (number of threads) (Adjust base on GPU memory)')
parser.add_argument('--learningRate', type=float,  default=1e-3, help='Learning rate')
parser.add_argument('--dropout', type=float,  default=0.5, help='Dropout')
parser.add_argument('--numEpochs', type=int,  default=40, help='Epochs')
parser.add_argument('--LRateDecay', type=int,  default=1, help='After lr_decay epochs lr reduces to lrDecayRate*lr')
parser.add_argument('--lrDecayRate', type=float,  default=0.9997592083, help='Decay for learning rate')
parser.add_argument('--minLRate', type=float,  default=5e-4, help='Minimum learning rate')
parser.add_argument('--clipNorm', type=float,  default=5.0, help='Clip gradients to this norm')
parser.add_argument('--ckptInterval', type=int,  default=800, help='Store checkpoint for every ckptInterval batchs')

args = parser.parse_args()
