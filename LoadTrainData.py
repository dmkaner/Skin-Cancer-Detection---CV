import glob
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pandas as pd 
import numpy as np 
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from NNArchitecture import Net

class SkinDataset(Dataset):
    def __init__(self, melanoma_path, nevus_path, keratosis_path, transform=None):
        self.melanoma_path = melanoma_path
        self.nevus_path = nevus_path
        self.keratosis_path = keratosis_path
        
        self.transform = transform

        self.melanoma_image_path = self.appendToPath(os.listdir(self.melanoma_path), self.melanoma_path)
        self.nevus_image_path = self.appendToPath(os.listdir(self.nevus_path), self.nevus_path)
        self.keratosis_image_path = self.appendToPath(os.listdir(self.keratosis_path), self.keratosis_path)

        self.img_names = self.melanoma_image_path + \
                            self.nevus_image_path + \
                            self.keratosis_image_path

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image = mpimg.imread(self.img_names[idx])
        label = self.img_names[idx].split('/')[3]
        print(label)

        if (image.shape[2] == 4):
            image = image[:,:,0:3]

        if self.transform:
            image = self.transform(image)

        return { 'image': image, 'label': label } 

    def appendToPath(self, pathList, dir):
        for n in range(len(pathList)):
            pathList[n] = dir + '/' + pathList[n]

        return pathList

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))
        return img

class Crop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = 0
        left = 0
        image = image[top: top + new_h,
                      left: left + new_w]
        return image

class ToTensor(object):
    def __call__(self, sample):
        image = sample
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


data_transform = transforms.Compose([Rescale(224), Crop(224), ToTensor()])
skin_dataset = SkinDataset('./data/train/melanoma', './data/train/nevus', './data/train/seborrheic_keratosis', transform=data_transform)

batch_size = 5
train_loader = DataLoader(skin_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)

# images = []
# for i, sample in enumerate(train_loader):
#     image = sample
#     image = image.type(torch.FloatTensor)
#     if i == 0:
#         images = image
#         break

# for i in range(batch_size):
#     image = images[i].data
#     image = image.numpy()
#     image = np.transpose(image, (1, 2, 0))
#     print(image.shape)
#     plt.imshow(cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

# plt.show()

net = Net()

criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)

def train_net(n_epochs):
    for epoch in range(n_epochs):
        running_loss = 0.0

        for batch_i, data in enumerate(train_loader):
            images = data['image']
            label = data['label']
            images = images.type(torch.FloatTensor)
            output_label = net.forward(images)
            # loss = criterion(output_label, label)
            loss = criterion(label.len)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0
    print('Finished Training')

n_epochs = 1
train_net(n_epochs)

# model_dir = 'saved_models/'
# model_name = 'skin_model.pt'
# torch.save(net.state_dict(), model_dir+model_name)
