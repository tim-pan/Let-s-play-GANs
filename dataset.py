import json
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
trans = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),        
])
#dont use normalize on generated imgs, this only on train data

def denorm(tensor, device):
    std = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)#reduce tensor to raw images
    return res
def get_iCLEVR_data(root_folder, mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root_folder,'train.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        img = list(data.keys())#file names
        label = list(data.values())#correspond labels
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]#convert to numbers
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
    else:
        data = json.load(open(os.path.join(root_folder,f'{mode}.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return None, label


class ICLEVRLoader(data.Dataset):
    def __init__(self, root_folder, trans=trans, cond=False, mode='train'):
        self.root_folder = root_folder
        self.mode = mode
        self.trans = trans
        self.img_list, self.label_list = get_iCLEVR_data(root_folder, mode)
        if self.mode == 'train':
            print("> Found %d images..." % (len(self.img_list)))
        
        self.cond = cond
        self.num_classes = 24
        
                
    def __len__(self):
        """'return the size of dataset"""
        return len(self.label_list)     

    def __getitem__(self, index):
        onehot_label = self.label_list[index]
        if self.img_list is None:  
            return onehot_label
        else:
            path = os.path.join(self.root_folder,'images', self.img_list[index])
            img = Image.open(path).convert('RGB')
            #this img is mode'rgba', we should convert it as rgb
            
            #scipy.misc.imsave('./save1.png', I)
            #plt.imshow()
            img = self.trans(img)
            return img, onehot_label
        
