from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from glob import glob
import os 
import numpy as np
import cv2


# custom import 
from model.neo_unet import NeoUNet
from model.loss import *
from data_loader import TestDataset

save_path = '/content/drive/MyDrive/20211/prj3/final-code-neo/output/save_output'
snapshot_path = '/content/drive/MyDrive/20211/prj3/final-code-neo/model/pth/epoch 003_loss:1.11730.pth'
test_path = glob('/content/drive/MyDrive/20211/prj3/data/test/test/*.jpeg')

def run_test():

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
            predict = model(image)
        predict = predict[3]
        predict = F.upsample(predict, size=(h, w), mode='bilinear', align_corners=False)
        neo_predict = predict[:, [0], :, :]
        non_predict = predict[:, [1], :, :]

        neo_predict = torch.sigmoid(neo_predict).squeeze().data.cpu().numpy()
        non_predict = torch.sigmoid(non_predict).squeeze().data.cpu().numpy()

        output = np.zeros(
            (predict.shape[-2], predict.shape[-1], 3)).astype(np.uint8)
        output[(neo_predict > non_predict) * (neo_predict > 0.5)] = [0, 0, 255]
        output[(non_predict > neo_predict) * (non_predict > 0.5)] = [0, 255, 0]
        
        cv2.imwrite(os.path.join(save_path, f'{image_id[0]}.png'), output)





