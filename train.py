import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F  
import numpy as np
from torch.utils.data import  DataLoader
import shutil
import os 

# custom import 
from model.neo_unet import NeoUNet
from model.loss import *


class CFG: 
    train_image_size = 352
    prin_freq = 30
    num_epochs = 1
    batch_size = 16
    num_workers = 4   #  woker read ? 
    lr = 0.001  
    weight_decay = 1e-4
    seed = 297
    save = '/content/drive/MyDrive/20211/prj3/final-code-neo/model/pth'


def train(train_loader, model, custom_loss, optimizer, epoch, device):
    trainsize = CFG.train_image_size
    size_rates = [8/11, 1, 14/11]
    losses = AvgMeter()
    dice_scores = AvgMeter()
    for step, item in enumerate(train_loader):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = item['image'], item['gt'] 
            batchsz = images.size(0)
            images = images.to(device) # BxCxHxW
            gts = gts.to(device) # BxCxWxW
            
            trainsize = int(round(trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                
            out = model(images)
            loss = custom_loss(out, gts)
            dice = dice_score(out, gts)
            loss.backward()
            optimizer.step()
            if rate == 1:
                losses.update(loss.item(), batchsz)
                dice_scores.update(dice.item(), batchsz)
                

        if step % CFG.prin_freq == 0 or step == len(train_loader) - 1:
            print('Epoch {:03d} | Step {:04d}/{:04d} | Train loss: {:.4f} | Learning_rate: {:.7f}'
                  .format(epoch, step, len(train_loader), losses.avg, optimizer.param_groups[0]['lr']))
    return losses.avg, dice_scores.avg


def train_loop(train_dataset):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = NeoUNet()
    model.to(device)
    model.train()
    np.random.seed(CFG.seed)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CFG.batch_size,
        sampler=torch.utils.data.RandomSampler(train_dataset),
        num_workers=CFG.num_workers,
        pin_memory=True
    )
    
    save_root = CFG.save
    os.makedirs(save_root, exist_ok=True) 
    min_loss = np.inf

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(CFG.lr),
        weight_decay=float(CFG.weight_decay),
        momentum=0.9,
        nesterov=True
    )
    
    custom_loss = NeoUNetLoss() 
    
    for epoch in range(1, CFG.num_epochs):
        print('Epoch {:03d}/{:03d}'.format(epoch, CFG.num_epochs))

        train_loss, train_dice = train(train_loader, model, custom_loss, optimizer, epoch, device)

        print('>> train dice: {:.3f}'.format(train_dice))
        if train_loss < min_loss: 

            min_loss = train_loss
            shutil.rmtree(save_root) # xoa model cu 
            os.makedirs(save_root, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_root, 'epoch {:03d}_loss:{:.5f}.pth'.format(epoch, min_loss)))












