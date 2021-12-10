import config
import os
import tensorflow as tf
import cv2 as cv
import numpy as np

def load_data():
    images = [os.path.join(config.IMAGE_PATH, f'{x}') for x in os.listdir(config.IMAGE_PATH)]
    masks = [os.path.join(config.MASK_PATH, f'{x}') for x in os.listdir(config.MASK_PATH)]
    images.sort()
    masks.sort()
    return images, masks

def read_image(image_path):
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image = cv.resize(image, (config.WIDTH, config.HEIGHT))
    image = image/255.0
    image = image.astype(np.float32)
    return image

def read_mask(mask_path):
    image = cv.imread(mask_path)
    image = cv.resize(image, (config.WIDTH, config.HEIGHT))
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)  
 
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255]) 
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    lower_mask = cv.inRange(image, lower1, upper1) 
    upper_mask = cv.inRange(image, lower2, upper2)

    red_mask = lower_mask + upper_mask    # concat  
    red_mask[red_mask != 0] = 1
    
    green_mask = cv.inRange(image, (36, 25, 25), (70, 255,255))
    green_mask[green_mask != 0] = 2
    
    full_mask = cv.bitwise_or(red_mask, green_mask)
    full_mask = full_mask.astype(np.uint8)
    return full_mask


def convert2TfDataset(x, y, batch_size=8):
    def preprocess(image_path, mask_path):
        def f(image_path, mask_path):
            image_path = image_path.decode()
            mask_path = mask_path.decode()
            image = read_image(image_path)  # 256*256*3 (0-1)
            mask = read_mask(mask_path)     # 256*256   (0 or 1 or 2)
            return image, mask

        # tf.numpy_function : Wraps a python function and uses it as a TensorFlow op.
        # tf.numpy_function(
        #     func, inp, Tout, name=None
        # )
        image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.uint8])
        
        # Returns a one-hot tensor.
        # tf.one_hot(
        #     indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None
        # )
        mask = tf.one_hot(mask, 3, dtype=tf.uint8)
        
        image.set_shape([config.HEIGHT, config.WIDTH, 3])      # 256*256*3 [a,b,c] BRG 0-1
        mask.set_shape([config.HEIGHT, config.WIDTH, 3])       # 256*256*3 [1,0,0] or [0,1,0] or [0,0,1]
        return image, mask

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset


