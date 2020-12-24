from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import numpy as np
import math
from PIL import Image
import copy
import pandas as pd
import tqdm
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras import backend as K
import os

class train_loader():
    #
    def __init__(self, img_path,mask_path,input_shape = (512,512),batch_size = 8):
        self.data = [i.rstrip('.PNG') for i in os.listdir(img_path)]
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.img_path = img_path
        self.mask_path = mask_path
        #
        self.datagen = ImageDataGenerator(rescale = 1./255,
                                       rotation_range=0.2,
                                       width_shift_range=0.05,
                                       height_shift_range=0.05,
                                       shear_range=0.05,
                                       zoom_range=0.05,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
        self.least = set(self.data)
    def __iter__(self):
        return self
    def __next__(self):
        if len(self.least) == 0:
            self.least = set(self.data)
        if len(self.least) >= self.batch_size:
            batch = set(random.sample(self.least,self.batch_size))
        else:
            batch = copy.copy(self.least)
        self.least -= batch
        
        X = []
        GT = []
        for sample in batch:
            img = Image.open(self.img_path+'\\'+sample+'.PNG').resize(self.input_shape)
            mask = Image.open(self.mask_path+'\\'+sample+'.TIFF').resize(self.input_shape)

            X.append(np.array(img,dtype= np.uint8))
            GT.append(np.array(mask,dtype= np.uint8))
            
        GT = np.array(GT)
        GT = GT.reshape(GT.shape + (1,))
        X = np.array(X)
        X = next(self.datagen.flow(X,batch_size = self.batch_size,seed = 1,shuffle=False))
        GT = next(self.datagen.flow(GT,batch_size = self.batch_size,seed = 1,shuffle=False))

        return X, GT[:,:,:,0]

    def __len__(self):
        return math.ceil(len(self.data)/self.batch_size)

class test_loader():
    #
    def __init__(self, img_path, input_shape = (512,512)):
        self.data = [i.rstrip('.PNG') for i in os.listdir(img_path)]
        self.input_shape = input_shape
        self.img_path = img_path
        self.data_load()

    def data_load(self):    
        X = []
        for sample in tqdm.tqdm(self.data):
            img = Image.open(self.img_path+'\\'+sample+'.PNG').resize(self.input_shape)
            X.append(np.array(img,dtype= np.uint8))
            

        X = np.array(X)
        self.X = X/255  
        


def maskImageGen(arr,pre_mask,name,out = '.'):
    from PIL import ImageDraw

    arr_ = np.concatenate((arr,np.ones(arr.shape[:-1]+(1,))),axis =2)
    arr_ *= 255
    img = Image.fromarray(arr_.astype(np.uint8),'RGBA')

    pre_ = np.where(pre_mask >= 0.5, 255, 0)
    mask = Image.fromarray(pre_[...,0].astype(np.uint8))
    mask_img = np.concatenate([pre_,pre_,pre_],axis = 2).astype(np.uint8)
   
    overlay = Image.new('RGBA',img.size, (255,255,255,0))
    drawing = ImageDraw.Draw(overlay)
    drawing.bitmap((0, 0), mask, fill=(76, 207, 183, 128))

    img = Image.alpha_composite(img, overlay)
    
    mask_img_ = Image.fromarray(mask_img)
    mask_img_.save(out+'\\GP_'+name+'.png')
    img.save(out+'\\pre_'+name +'.png')

def cal_IoU(true,pred):
    
    pre_ = pred[...,0]
    pre_1 = np.where(pre_ >= 0.5, 1,0)

    tru = true[...,0]
    tru_1 = np.where(tru >= 0.5, 1,0)

    inse = np.sum(pre_1 * tru_1)
    union = np.sum(np.where((pre_1 + tru_1) >= 1, 1,0))

    return float(inse/union)

class round_decay(Callback):

    def __init__(self,  r, decay, weights,monitors = 'loss'):
        self.round = r
        self.decay = decay
        self.best_loss = 9999999
        self.out = weights
        self.cur_R = 1
        self.monitors = monitors

    def on_epoch_begin(self, epoch, logs=None):
        print('Epoch {0}: lr: {1}'.format(epoch+1, float(K.get_value(self.model.optimizer.lr))))

    def on_epoch_end(self, epoch, logs=None):
        R = (epoch)//self.round + 1
        if R > self.cur_R:
            self.cur_R = R
            new_lr = K.get_value(self.model.optimizer.lr) * self.decay
            K.set_value(self.model.optimizer.lr, new_lr)
            print('\nlr is fixed to {0}'.format(new_lr))
        
        cur_loss = logs.get(self.monitors)
        if cur_loss <= self.best_loss:
            print('\nbest model weights are saved at {0}'.format(self.out))
            self.model.save_weights(self.out)
            self.best_loss = cur_loss
            self.best_weights = self.model.get_weights()
            self.best_epochs = epoch

    def on_train_end(self, logs = None):
        print('weights are setted to best weights (epochs {0})'.format(self.best_epochs + 1))
        self.model.set_weights(self.best_weights)


