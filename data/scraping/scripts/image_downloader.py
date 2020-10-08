# Using the code from: https://github.com/deepanprabhu/duckduckgo-images-api
from helpers.duckduckgo_imageURLdownloader import ImageURLDownloader
import os
import wget
from pathlib import Path

max_imgs = 1
dloader = ImageURLDownloader(max_results=max_imgs)

# Classes of foods to download
def get_list_foods(path_list, max_imgs):
    """ path_list = path to a .txt file with the class names"""
    food_classes = []
    with open(path_list, 'r') as f:
        for l in f:
            food_classes.append(l.split('\n')[0])
    
    for food in food_classes:
        print(f'Working in {food}')
        dloader.get(food) # Saves a .csv with the url of the image
    return food_classes
    
    
def create_paths_images(food_list):
    for food in food_list:
        path = '../scraped_images/'#Path().absolute()/'scraped_images/'
        os.makedirs(path + food, exist_ok=True)
        
    
def download_images(food_list):
    for food in food_list:
        path = '../scraped_images/'#Path().absolute()/'scraped_images/'
        food_urls = []
        with open(path + food + '.csv', 'r') as f:
            for l in f:
                food_urls.append(l.split('\n')[0])
                
        counter = 0
        script_bash = []
        for url in food_urls:
            counter += 1
            script_bash.append(f'wget -O {path}{food}/{food}_{counter}.jpg {url}')
        # try:
        #     wget.download(url, out=f'{path}/{food}/{food}_{counter}.jpg')
        # except:
        #     print(f'{food} #{counter} failed to download')
        return script_bash


def final_downloader(script_bash):
  counter=0
  for i in bash:
    counter +=1
    if counter % 100: print(counter)
    try:
      os.system(i)
    except:
      print(f'error #{counter}')
