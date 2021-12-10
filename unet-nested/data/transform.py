import os
import cv2 as cv
import numpy as np

MASK_RGB_PATH = "/content/drive/MyDrive/20211/prj3/final-code-pytorch/data/masks_in_rgb"
MASK_BLACK_PATH = "/content/drive/MyDrive/20211/prj3/final-code-pytorch/data/masks_in_black"

IMAGE_ORIGIN ="/content/drive/MyDrive/20211/prj3/final-code-pytorch/data/images_origin"
IMAGE_256 = "/content/drive/MyDrive/20211/prj3/final-code-pytorch/data/images_256"

IMAGE_TEST = "/content/drive/MyDrive/20211/prj3/final-code-pytorch/data/test/input"
IMAGE_TEST_256 = "/content/drive/MyDrive/20211/prj3/final-code-pytorch/data/test/input_size_256"

def read_mask(mask_path):
    image = cv.imread(mask_path)
    image = cv.resize(image, (256, 256))
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


def read_image(image_path):
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image = cv.resize(image, (256, 256))
    image = image.astype(np.float32)
    return image

def transform_masks():
    images = [os.path.join(MASK_RGB_PATH, f'{x}') for x in os.listdir(MASK_RGB_PATH)]
    for image_path in images:
        name_image =str(image_path).split("/")[-1]
        name_image = str(name_image).replace(".jpeg",".png")
        path_write = MASK_BLACK_PATH +"/"+ name_image

        mask_in_black = read_mask(image_path)

        list_unique = np.unique(mask_in_black)
        if len(list_unique) >=4:
          print(list_unique)
          print(mask_path)

        cv.imwrite(path_write, mask_in_black)

def transform_image():
    images = [os.path.join(IMAGE_ORIGIN, f'{x}') for x in os.listdir(IMAGE_ORIGIN)]
    for image_path in images:
        name_image =str(image_path).split("/")[-1]
        name_image = str(name_image).replace(".jpeg",".png")
        path_write = IMAGE_256 +"/"+ name_image

        image_256 = read_image(image_path)
        cv.imwrite(path_write, image_256)


def transform_image_test():
    images = [os.path.join(IMAGE_TEST, f'{x}') for x in os.listdir(IMAGE_TEST)]
    for image_path in images:
        name_image =str(image_path).split("/")[-1]
        name_image = str(name_image).replace(".jpeg",".png")
        path_write = IMAGE_TEST_256 +"/"+ name_image

        image_256 = read_image(image_path)
        cv.imwrite(path_write, image_256)

