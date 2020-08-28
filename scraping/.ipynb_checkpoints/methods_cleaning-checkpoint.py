import os
import torch
import numpy as np
from torchvision import models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn


# Loading the resnet model
resnet = models.resnet34(pretrained=True)

# Cutting off the tail of the resnet
resnet_food = nn.Sequential(*list(resnet.children())[:-1])

# Transformations for the dataset
transform = transforms.Compose(
    [transforms.Resize([256, 256]),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def create_dl(path):
    """ Creates a dataloader for the class"""
    data_food_hamburger = ImageFolder(
        path,
        transform=transform
    )
    
    dl_hamburger = DataLoader(
        data_food_hamburger,
        batch_size=1,
        num_workers=3
    )
    return dl_hamburger

    
# def create_embedding(dl_hamburger):
#     """ Creates an embedding for each image in the data loader"""
#     counter = 0
#     embedding_pics_hamburger = {}
    
#     for b in dl_hamburger:
#         embedding_pics_hamburger[counter] = resnet_food(b[0])[0].reshape(-1)
#         counter += 1
        
        
#     return embedding_pics_hamburger


# Done with Jacub:


def create_embedding(dl_hamburger):
    """ Creates an embedding for each image in the data loader"""
    counter = 0
    embedding_pics_hamburger = []
    
    for b in dl_hamburger:
        embedding_pics_hamburger.append(resnet_food(b[0])[0].reshape(-1).detach().cpu().numpy())
        counter += 1
        # if counter % 100 == 0:print(counter)
        
    return embedding_pics_hamburger


def idx_repeated_images(embedding_pics_hamburger):
    """ Searches through the embedding the images that are similar """
    counter = 0
    repeated_images=[]
    repeated_array = np.array([None, None])
    
    # Looping through all the items in the embedding
#     for i, emb in enumerate(embedding_pics_hamburger.values()):
    for i, emb in enumerate(embedding_pics_hamburger):
    
        # Creating a list without the actual img index 
        list_loops = list(range(len(embedding_pics_hamburger)))
        list_loops.pop(i)

        # Go through the list
        for j in list_loops:
            # By
#             import pdb; pdb.set_trace()
            if torch.dist(torch.from_numpy(emb), torch.from_numpy(embedding_pics_hamburger[j])) < 1.5:
                #print(f'Img{i} is == Img{j} by {torch.dist(emb, embedding_pics_hamburger[j])} ')
                
                # This is the array that cointains all the repeated images, but it has repeated idx
                repeated_array = np.concatenate([repeated_array, np.array([i,j])])
        
    # Reshaping the data to get a list of [x, y]
    pairs_repeated = repeated_array.reshape(-1, 2)[1:]
        
    # Creating loop to get non-duplicated
    images_to_delete = []
    for i, pair in enumerate(pairs_repeated):
        if pair[0] < pair[1]:
            images_to_delete.append(pair[1])
        
    # Deleting duplicates with a set
    images_to_delete_np = np.array(list(set(images_to_delete)))
        
    return images_to_delete_np


def repeated_img_path(images_to_delete_np, dl_hamburger):
    delete_img_path = []
    for i in images_to_delete_np:
        delete_img_path.append((dl_hamburger.dataset.imgs[i][0]).rsplit('/', 1)[1])
    return delete_img_path


# Set first your cwd to the path of the folder
def delete_repeated_img(delete_img_path):
    for i in delete_img_path:
        if os.path.exists(i):
            os.remove(i)
        else:
            print("Can not delete the file as it doesn't exists")