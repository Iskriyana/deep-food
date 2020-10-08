# Artificial Creation of Images # 

This script iterates through the downloaded images in data/scraping/scraped_images and: 
1. converts the image to png in order to add an alpha-channel for transparency 
2. loads a random background image from data/create_images/background
3. pastes the newly created png-images onto the random background image using the image as its mask. 

The result is an artificially created image on a new background. This is done in order to introduce more diversity into the food identification model (CNN, food_identification).