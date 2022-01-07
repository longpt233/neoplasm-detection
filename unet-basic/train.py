from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from keras import backend as K

# import segmentation_models_3D as sm 


# local 
from model import unet
from model import unet_mobi
from model import unet_nested_v2

from data_prepare import *
import config

"""
@input model structure  
@return H and moddel trained 
"""
def fit_model(model_and_path):
   
  # split 
  images_path_list, masks_path_list = load_data() 
  train_x, valid_x, train_y, valid_y = train_test_split(images_path_list, masks_path_list, test_size=0.2, random_state=42) 

  train_dataset = convert2TfDataset(train_x, train_y, config.BATCH_SIZE)
  valid_dataset = convert2TfDataset(valid_x, valid_y, config.BATCH_SIZE)

  # get step 
  train_step = len(train_x)//config.BATCH_SIZE
  if len(train_x) % config.BATCH_SIZE != 0:
      train_step += 1
  valid_step = len(valid_x)//config.BATCH_SIZE
  if len(valid_x) % config.BATCH_SIZE != config.BATCH_SIZE:
      valid_step += 1


  callbacks = [
          ModelCheckpoint(model_and_path["path"], verbose=1, save_best_model=True),
          ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6),
          EarlyStopping(monitor='val_loss', patience=5, verbose=1)
      ]

  model = model_and_path["model"]

  model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=[ 
                    iou_score,
                    dice_coef
                ])


  # fit 
  H = model.fit(train_dataset,
              validation_data=valid_dataset,
              steps_per_epoch=train_step,
              validation_steps=valid_step,
              epochs=config.EPOCH,
              callbacks = callbacks
              )

  return H ,model

def dice_coef_v2(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def iou_score(y_true, y_pred, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(y_pred)
    inputs = K.cast(inputs,"float32")
    targets = K.flatten(y_true)
    
    intersection = K.sum(targets* inputs)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return  IoU


def tversky(y_true, y_pred, smooth = 1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    y_true_pos = K.cast(y_true_pos,"float32")
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_loss(targets, inputs, alpha=0.8, gamma=2):    
    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    targets = K.cast(targets,"float32")

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss

'''
name = name model 
@ re
'''
def load_model(name, is_train= True):

    # nếu train rồi thì lấy ra model thôi 
    if is_train: 

        if name == "unet-v1":
            model =  unet.build_unet()
            model.load_weights(config.MODEL_PATH_UNET_V1)
            print("load unet v1 ")
            return model

        # elif name == "unet-v2":
        #     model=unet_v2.unet()
        #     model.load_weights(config.MODEL_PATH_UNET_V2)
        #     print("load unet v2 ")
        #     return model

        elif name == "unet-v3":
            model=unet_mobi.build_model()
            model.load_weights(config.MODEL_PATH_UNET_V3)
            print("load unet v3 ")
            return model
        
        elif name == "unet++":
            model= unet_nested_v2.Nest_Net(deep_supervision=False)
            model.load_weights(config.MODEL_PATH_NESTED)
            return model

        else :
            raise Exception("name do not match any case")

    # chưa train thì trả về cả model lẫn path 
    else: 
        model_and_path = {}
        if name == "unet-v1":
            model_and_path["model"] =  unet.build_unet()
            model_and_path["path"] =config.MODEL_PATH_UNET_V1
            print("load unet v1 ")
            return model_and_path

        # elif name == "unet-v2":
        #     raise Exception("model loi ")
        #     model_and_path["model"] = unet_v2.unet()
        #     model_and_path["path"] =  config.MODEL_PATH_UNET_V2
        #     print("load unet v2 ")
        #     return model_and_path

        elif name == "unet-v3":
            model_and_path["model"] = unet_mobi.build_model()
            model_and_path["path"] =  config.MODEL_PATH_UNET_V3
            print("load unet v3 ")
            return model_and_path
        
        elif name == "unet++":
            model_and_path["model"] = unet_nested_v2.Nest_Net(256,256,3,3,deep_supervision=False)
            model_and_path["path"] =  config.MODEL_PATH_NESTED
            return model_and_path

        else :
            raise Exception("name do not match any case")


        


