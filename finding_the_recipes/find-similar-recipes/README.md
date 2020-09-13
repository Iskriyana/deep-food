# One hot encoding of the recipes

This is the first step toward finding the similar recipes. Here, we one hot encode the recipes. This enables us to compare the recipes and find the similar ones.

As the list of the ingredients is long, here (and in the detection of ingredients from the pictures) we confine ourselved to a limited number of the ingredients. 

The script ```one_hot_encoding_recipes.py``` requires 
* the number of important ingredients, i.e. ```nr_important_ingredients``` 
* and a list composed of id of these elements in the database, i.e. ```important_ingredients```. 

Here the database refers to the simplified 1M recipes.

One can one hot encode the whole recipe list by

```
python one_hot_encoding_recipes.py
```

The scripts saves the one hot encoded recipes into 

```
recipes_one_hotted.gz
```

and the list of the important ingredients into

```
important_ingredients.gz
```

One can later read the ```recipes_one_hotted.gz``` and use it to find the similar recipes. One should note that ```important_ingredients.gz``` is also required to find out which elements of the one hot coded vector refers to which ingredient.
