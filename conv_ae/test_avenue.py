import torch
import torch.nn as nn
import os
from sklearn.neighbors import KernelDensity
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from PIL import Image
import random
import numpy as np
import argparse
# import faiss
from video_dataset2 import VideoDatasetWithFlows, img_tensor2numpy
import os
# from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture
import sys
from sklearn.mixture import GaussianMixture


path = r"C:\Users\srini\OneDrive\Desktop\internship\VAD_avenue\data\avenue\testing\frames"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])


# image = Image.open(r"C:\Users\srini\OneDrive\Desktop\internship\Attribute_based_VAD\Accurate-Interpretable-VAD\data\avenue\testing\frames\16\234.jpg") #normal
# image = Image.open(r"C:\Users\srini\OneDrive\Desktop\internship\Attribute_based_VAD\Accurate-Interpretable-VAD\data\avenue\testing\frames\06\0841.jpg")
# image = Image.open(r"C:\Users\srini\OneDrive\Desktop\internship\sreeni\images\training\frames\01\004.jpg") #normal2
# image = Image.open(r"C:\Users\srini\OneDrive\Desktop\internship\sreeni\images\testing\frames\01\0577 (3).jpg")  # bag throw
# image = Image.open(r"C:\Users\srini\OneDrive\Desktop\internship\sreeni\images\training\frames\01\0001 (3).jpg")  # shady normal
# image = Image.open(r"C:\Users\srini\OneDrive\Desktop\internship\sreeni\images\testing\frames\01\0000.jpg")  # red bag
# image = Image.open(r"C:\Users\srini\OneDrive\Desktop\internship\sreeni\images\testing\frames\01\401 (2).jpg")  # man running
# image = Image.open(r"C:\Users\srini\OneDrive\Desktop\internship\sreeni\images\testing\frames\01\0360 (3).jpg") #police men
# image = Image.open(r"C:\Users\srini\OneDrive\Desktop\internship\sreeni\images\testing\frames\01\0512.jpg")  # kid close
# image = Image.open(r"C:\Users\srini\OneDrive\Desktop\internship\sreeni\images\testing\frames\01\0530 (3).jpg")  # shady man
# image = Image.open(r"C:\Users\srini\OneDrive\Desktop\internship\sreeni\images\testing\frames\01\0955.jpg")  # lady block camera
# image = Image.open(r"C:\Users\srini\OneDrive\Desktop\internship\sreeni\images\testing\frames\01\0177.jpg")  # two men
# image.show()

path = r"C:\Users\srini\OneDrive\Desktop\internship\VAD_avenue\data\avenue\testing\frames" + "\\"


mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])


transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128),antialias=True), transforms.Normalize(mean, std), transforms.Lambda(lambda x: torch.clamp(x, -1, 1))])


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, padding = 1, stride = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, padding = 1, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, padding = 1, stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, padding = 1, stride = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 8)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 8),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, padding = 1, stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, padding = 1, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, padding = 1, stride = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, padding = 1, stride = 2),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


criterion = nn.MSELoss()

model = Autoencoder().to(device)
checkpoint = torch.load("checkpoint_overfit_80.pth")
model.load_state_dict(checkpoint["model_state"])
model.eval()
# model.load_state_dict(torch.load("model.pth", map_location = "cuda:0"))
# model.eval()

result=[]

# output = model(image)
for i in os.listdir(path):
    image_dir=path +'\\'+i
    for j in os.listdir(image_dir):
        image_path = image_dir + "\\" + j
        image = Image.open(image_path)
        image = transform(image)
        # batch.append(image)
        image = image.view(1, 3, 128, 128)
        image = image.to(device)

        output = model(image)
        with torch.no_grad():

            # np_output = output.view(3, 128, 128)
            # np_output = np_output.cpu().numpy()
            # np_output = np.transpose(np_output, (1, 2, 0)) / 2 + 0.5

            loss = criterion(output, image)
            result.append(loss.cpu().numpy())
            # if(loss>0.1):#for 100 used 0.11
            #     result.append(1)
            # else:
            #     result.append(0)

            # print(loss.item())

            # plt.imshow(np_output)
            # plt.show()

result=np.array(result)
print(result)
np.save('conv_ae.npy',result)

# result=torch.tensor(result)

# root = 'data/'
# test_dataset = VideoDatasetWithFlows(dataset_name='avenue', root=root,train=False, normalize=False)

# print('Micro AUC: ', roc_auc_score(test_dataset.all_gt, result) * 100)
# # print(type(test_dataset.all_gt))

# correct = 0

# for i in range(len(result)):
#     if (result[i]==test_dataset.all_gt[i]):
#         correct = correct + 1
# print((correct/len(result))*100)