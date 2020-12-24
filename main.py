
#################################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--img")
parser.add_argument("--out",default='.')
args = parser.parse_args()
#python main.py --device 1 --img img --out example


from model import *
from utils import *
from tensorflow.keras.callbacks import CSVLogger
import tqdm
from tensorflow.keras import optimizers

#data load
print('data are loaded')
data = test_loader(args.img, input_shape = (512,512))
print('Total {0} imgs were loaded'.format(len(data.data)))

#model construct
model  = Unet((512,512,3))
model.load_weights('weights.h5')


#Prediction step
print('######################')
print('##                  ##')
print('##    Prediction    ##')
print('##                  ##')
print('######################')

pred = model.predict(data.X, verbose = 1)

print('Segmentation results are savet at {0}'.format(args.out))
for i in tqdm.tqdm(range(len(data.X))):
    maskImageGen(data.X[i],pred[i],data.data[i],args.out)

