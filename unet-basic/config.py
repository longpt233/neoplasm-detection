# image 
WIDTH = 256 
HEIGHT = 256

# hyper
BATCH_SIZE = 8   # defaul 8
EPOCH = 10

# path 
IMAGE_PATH = '/content/drive/MyDrive/20211/prj3/data/train/train'
MASK_PATH =  '/content/drive/MyDrive/20211/prj3/data/train_gt/train_gt'

# model 
# unet tren kaggle   :
MODEL_PATH_UNET_V1 = '/content/drive/MyDrive/20211/prj3/neoplasm-detection/unet-basic/model/unet.h5'

# mobi -unnet tren kaggle
MODEL_PATH_UNET_V3 = '/content/drive/MyDrive/20211/prj3/neoplasm-detection/unet-basic/model/unet_mobi.h5'

# nested
MODEL_PATH_NESTED = '/content/drive/MyDrive/20211/prj3/neoplasm-detection/unet-basic/model/unet_nested_v2.h5'

TEST_PATH =  "/content/drive/MyDrive/20211/prj3/data/test/test"
PREDICT_PATH="/content/drive/MyDrive/20211/prj3/neoplasm-detection/unet-basic/predict_collection" 
SUBMIT_PATH = "/content/drive/MyDrive/20211/prj3/neoplasm-detection/unet-basic/predict_collection/submit"