from data_prepare import *
import config
import cv2 as cv
import cv2


def predict(model):
  test_image_list = [os.path.join(config.TEST_PATH, f'{x}') for x in os.listdir(config.TEST_PATH)]
  iter_path = 0 
  for path_test in test_image_list: 
    iter_path= iter_path+1
    name_image =str(path_test).split("/")[-1]
    path_write = config.PREDICT_PATH +"/"+ name_image
    test_img = read_image(path_test)

    original_image = cv2.imread(path_test)
    h, w, c = original_image.shape

    predict = model.predict(np.expand_dims(test_img, axis=0))[:]

    predict_max = np.argmax(predict, axis=-1)   # Returns the indices of the maximum values along an axis
    img =predict_max[0]
    test = np.zeros((256,256,3), dtype=np.int64)   # why int 64 ?
    for i in range(255):
      for j in range(255):
        if img[i,j] == 1:
          test[i,j] =  [0,0,255]
        if img[i,j] == 2:
          test[i,j] = [0,255,0] 

    resized_img = cv2.resize(test, ( w , h ), interpolation=cv2.INTER_LINEAR_EXACT)

    cv.imwrite(path_write, resized_img)
    if iter_path%20 ==0 : 
      print("done batch 20 image, current", iter_path)




# def predict(model):
#   test_image_list = [os.path.join(config.TEST_PATH, f'{x}') for x in os.listdir(config.TEST_PATH)]
#   i = 0 
#   for path_test in test_image_list: 
#     i= i+1
#     name_image =str(path_test).split("/")[-1]
#     path_write = config.PREDICT_PATH +"/"+ name_image
#     test_img = read_image(path_test)

#     predict = model.predict(np.expand_dims(test_img, axis=0))[:]

#     predict_max = np.argmax(predict, axis=-1)   # Returns the indices of the maximum values along an axis
#     img =predict_max[0]
#     test = np.zeros((256,256,3), dtype=np.int64)   # why int 64 ?
#     for i in range(255):
#       for j in range(255):
#         if img[i,j] == 1:
#           test[i,j] = [0,0,255]
#         if img[i,j] == 2:
#           test[i,j] = [0,255,0] 

#     resized_img = cv2.resize(test, (1280, 995), interpolation=cv2.INTER_LINEAR_EXACT)

#     cv.imwrite(path_write, resized_img)
#     print(i)
#     if i%20 ==0 : 
#       print("done batch 20 image, current", i)

