from torchvisin import models, transforms
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np



# importing restnet model

restnet = models.resnet34(pretrained=True)

preprocess = transforms.Compose(
   [transforms.Resize(256),
    transforms.CenterCrop(224), # resize the images towards the center
    transforms.ToTensor(), #transforming the image to a tensor
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], # why are we normalizing with these weights?
    std=[0.229, 0.224, 0.225])]
)

img = Image.open('../scraped_images/foods/pizza/00000001.jpg')

tensorito = transforms.Compose([transforms.ToTensor()])

img_t = preprocess(img)

with open('../utils/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
    
import torch
batch_t = torch.unsqueeze(img_t, 0) # transforms from [3, 224, 224] to torch.Size([1, 3, 224, 224])
batch_t.size()

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()

_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

_, index = torch.max(img_original_out, 1)

# WITHOUT PRE-PROCESSING THIS HAS NOT WORKED!
percentage = torch.nn.functional.softmax(img_original_out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn


transform = transforms.Compose(
    [transforms.Resize([256, 256]),
    transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

resnet = models.resnet34(pretrained=True)