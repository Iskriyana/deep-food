"""Configuration file
"""

import os

#from os import environ as env
#import multiprocessing

#PORT = int(env.get("PORT", 8501))
#DEBUG_MODE = int(env.get("DEBUG_MODE", 1))

# Gunicorn config
#bind = ":" + str(PORT)
#workers = multiprocessing.cpu_count() * 2 + 1
#threads = 2 * multiprocessing.cpu_count()

CLASSES = ['almond', 'apple', 'apricot', 'avocado', 'banana', 'beef', 'blackberry', 'blueberry', 'broccoli', 'cabbage', 
           'carrot', 'cauliflower', 'celery', 'cheese', 'cherry', 'chicken_breast', 'chocolate', 'corn', 'cucumber',
           'egg', 'eggplant', 'fig', 'grapefruit', 'grapes', 'grated_cheese', 'kiwi', 'lemon', 'lettuce', 'lime',
           'mango', 'melon', 'mushroom', 'olive', 'onion', 'orange', 'other', 'paprika', 'passionfruit','peach',
           'pear','pineapple', 'plum', 'pomegranate', 'pork', 'radish', 'raspberry', 'salami', 'scallion',
           'strawberry', 'tomato', 'watermelon', 'whole_chicken', 'zucchini']

IND2CLASS = {0: 'almond', 1: 'apple', 2: 'apricot', 3: 'avocado', 4: 'banana', 5: 'beef', 6: 'blackberry', 7: 'blueberry',
             8: 'broccoli', 9: 'cabbage', 10: 'carrot', 11: 'cauliflower', 12: 'celery', 13: 'cheese', 14: 'cherry',
             15: 'chicken_breast', 16: 'chocolate', 17: 'corn', 18: 'cucumber', 19: 'egg', 20: 'eggplant', 21: 'fig',
             22: 'grapefruit', 23: 'grapes', 24: 'grated_cheese', 25: 'kiwi', 26: 'lemon', 27: 'lettuce', 28: 'lime',
             29: 'mango', 30: 'melon', 31: 'mushroom', 32: 'olive', 33: 'onion', 34: 'orange', 35: 'other', 36: 'paprika',
             37: 'passionfruit', 38: 'peach', 39: 'pear', 40: 'pineapple', 41: 'plum', 42: 'pomegranate', 43: 'pork',
             44: 'radish', 45: 'raspberry', 46: 'salami', 47: 'scallion', 48: 'strawberry', 49: 'tomato', 50: 'watermelon',
             51: 'whole_chicken', 52: 'zucchini'}


CLASS2IND = {v:k for (k,v) in IND2CLASS.items()}

FINAL_MODEL_PATH = '../food_identification/models/final_model/'
EXPORT_DIR = 'output'
EXPORT_MODEL = 'exported_model'

# set path to key credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "deepfoodie-0d2d322738d7.json"

MODEL_SPECS = {'project': 'deepfoodie',
               'model': 'deepfoodie_img_ident',
               'version': 'v1'}

