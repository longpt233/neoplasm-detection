from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam


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

  model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=[
                  tf.keras.metrics.Precision(),
                  tf.keras.metrics.Recall(),
                  tf.keras.metrics.CategoricalAccuracy(name='acc'),
                  tf.keras.metrics.MeanIoU(num_classes=3)
              ])

  # model.compile(optimizer=Adam(lr=1e-3), loss='mse', metrics=['accuracy'])

  # fit 
  H = model.fit(train_dataset,
              validation_data=valid_dataset,
              steps_per_epoch=train_step,
              validation_steps=valid_step,
              epochs=config.EPOCH,
              callbacks = callbacks
              )

  return H ,model

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
            model= unet_nested_v2.Nest_Net( deep_supervision=False)
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


        


