
#################################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--img")
parser.add_argument("--mask")
parser.add_argument("--out",default='.')
parser.add_argument("--device", help="input the column of y",default = '1')
args = parser.parse_args()
device = args.device
'''
class args():
    def __init__(self):
        self.img = 'data\\img'
        self.mask = 'data\\mask'
        self.out = '.'
        self.device = '1'

        
args = args()
'''
#python train.py --device 1 --img data\\img --mask data\\mask --out .
###hyper parameter setting######
config = {
"lr" : 1e-4,
"decay" : 0.1,
'epochs' : 50,
'input_shape' : (512,512,3),
'batch_size' : 4,
}
###############################

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=args.device
#####################################

from model import *
from utils import *
from tensorflow.keras.callbacks import CSVLogger
import tqdm
from tensorflow.keras import optimizers


#data load
data = train_loader(args.img,args.mask,
                    input_shape = config['input_shape'][:-1],
                    batch_size = config['batch_size'])


#model construct
model  = Unet(config['input_shape'])

#Call back set
ckp = round_decay(r = 40, decay = config['decay'], weights = args.out+'\\weights.h5')
logger = CSVLogger(args.out+'\\'+"history.txt", append=True, separator='\t')
adam = optimizers.Adam(lr=config["lr"])
model.compile(loss = 'binary_crossentropy',
            metrics = ['accuracy'], optimizer = adam)  

#trainning step
print('######################')
print('##                  ##')
print('##   Train Models   ##')
print('##                  ##')
print('######################')
#model.load_weights(args.out+'\\weights.h5')
model.fit(data, epochs = config['epochs'],
        steps_per_epoch = len(data),callbacks= [ckp,logger])
model.save_weights(args.out+'\\weights.h5')


