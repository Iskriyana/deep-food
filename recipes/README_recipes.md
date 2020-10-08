# Recipes Generation


### One hot encoding of the recipes

* One-hot encoding of the recipes is the first step toward finding the similar recipes. This enables us to compare the recipes and find similar ones.

* As the list of the ingredients is long, here (and in the detection of ingredients from the pictures) we confine ourselved to a limited number of the ingredients. 

* The recipe data set refers to the simplified 1M recipes in 1_data/recipe_data_sets/input.

* One can one hot encode the whole recipe list by

```
python one_hot_encoding_recipes.py
```

The script saves the one hot encoded recipes into 

```
recipes_one_hotted.gz
```

and the list of the important ingredients into

```
important_ingredients.gz
```


### Similarity finder 

The output ```recipes_one_hotted.gz``` is used to find the similar recipes in similarity_finder_for_app.py. One should note that ```important_ingredients.gz``` is also required to find out which element of the one hot encoded vector refers to which ingredient.
