# Scraping #

### Workflow ###

1. Get 50 classes from /utils/50_food_classes.txt and ca. 150 samples for each class --> /scripts/scrape_google.py

2. Through a resnet34, run the downloaded images through the neural network and get the representation in the vector space --> /cleaning_duplicates/identifier_duplicates.py

3. Get rid of duplicates --> /cleaning_duplicates/methods_cleaning.py

In order to get cleaner and more diverse results, we executed queries as below:

apple:
  * apple breakfast
  * apple food
  * apple fresh
  * manzana
  * manzana fresca
