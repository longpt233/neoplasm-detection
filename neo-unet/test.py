from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from glob import glob
import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt


# custom import 
from model.neo_unet import NeoUNet
from model.loss import *
from data_loader import TestDataset
from model.unet import Unet

save_path_root = '/content/drive/MyDrive/20211/prj3/neoplasm-detection/output/' 
test_path = glob('/content/drive/MyDrive/20211/prj3/data/test/test/*.jpeg')

def run_test_neo(model_path):

    snapshot_path = model_path
    model_name = str(snapshot_path).split("/")[-1]
    save_folder = str(model_name).replace(".pth","")

    save_path = save_path_root+ save_folder
    os.makedirs(save_path, exist_ok=True)

    device = torch.cuda if torch.cuda.is_available() else torch.device('cpu')
    model = NeoUNet()
    model.load_state_dict(torch.load(snapshot_path, map_location=torch.device('cuda')))
    test_dataset = TestDataset(test_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=False
    )
    for idx, item in tqdm(enumerate(test_loader)):
        image, image_id, size = item['image'], item['id'], item['size']
        h, w = size[0], size[1]
        with torch.no_grad():
            predict = model(image) # trả về một list do thằng F.interpolate upscale lên 
  
        predict = predict[3]  # lấy có thể lấy ra 0,1,2,3 -> nhưng 3 là cái rõ nhất (đầu ra )
        predict = F.upsample(predict, size=(h, w), mode='bilinear', align_corners=False)

        # F  cx là kiểu biến dổi nhưng k cố tham số 


        neo_predict = predict[:, [0], :, :]   # [0] get slice 
        non_predict = predict[:, [1], :, :]

        neo_predict = torch.sigmoid(neo_predict).squeeze().data.cpu().numpy()  # gpu-> cpu-> .numpy
        non_predict = torch.sigmoid(non_predict).squeeze().data.cpu().numpy()

        output = np.zeros(
            (predict.shape[-2], predict.shape[-1], 3)).astype(np.uint8)
        output[(neo_predict > non_predict) * (neo_predict > 0.5)] = [0, 0, 255]  # * is elemment wise 
        output[(non_predict > neo_predict) * (non_predict > 0.5)] = [0, 255, 0]
        
        cv2.imwrite(os.path.join(save_path, f'{image_id[0]}.png'), output)



def run_test_unet(model_path):

    snapshot_path = model_path
    model_name = str(snapshot_path).split("/")[-1]
    save_folder = str(model_name).replace(".pth","")

    save_path = save_path_root+ save_folder
    os.makedirs(save_path, exist_ok=True)

    device = torch.cuda if torch.cuda.is_available() else torch.device('cpu')
    model = Unet(in_ch=3, out_ch=2)
    model.load_state_dict(torch.load(snapshot_path, map_location=torch.device('cuda')))
    test_dataset = TestDataset(test_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=False
    )
    for idx, item in tqdm(enumerate(test_loader)):
        image, image_id, size = item['image'], item['id'], item['size']
        h, w = size[0], size[1]
        with torch.no_grad():
            predict = model(image)

        predict = predict[0]
        predict = predict.permute(1, 2, 0).numpy()   # [2, 352, 352] => (352, 352, 2)  
        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 30))
        ax[0].set_title('ori')
        ax[1].set_title('neo')
        ax[2].set_title('non-neo')
        ax[0].imshow(image[0].permute(1, 2, 0).numpy())
        ax[1].imshow(predict[:,:,0])
        ax[2].imshow(predict[:,:,1])

        output = np.zeros(
            (predict.shape[-2], predict.shape[-1], 3)).astype(np.uint8)
        output[(neo_predict > non_predict) * (neo_predict > 0.5)] = [0, 0, 255]
        output[(non_predict > neo_predict) * (non_predict > 0.5)] = [0, 255, 0]
        
        cv2.imwrite(os.path.join(save_path, f'{image_id[0]}.png'), output)






