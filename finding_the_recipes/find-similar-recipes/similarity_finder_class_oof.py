import numpy as np
import pandas as pd
from tensorflow.keras.backend import one_hot
from random import sample

# read this from the recipes, currently hot coded
MAX_NR_INGREDIENTS = 3500



print("... loading the original dataset")
#with np.load('../find_the_most_common_ingredients/simplified-recipes-1M.npz', allow_pickle=True) as data:
#    recipes = data['recipes']
#    ingredients = data['ingredients']
recipes = np.load('../find_the_most_common_ingredients/recipes_clean.npy', allow_pickle=True)

ingredients = np.load('../find_the_most_common_ingredients/ingredients_clean.npy', allow_pickle=True).item()
print("... loading finished.\n")

# list of important ingredients
important_ingredients = np.array([])
file_classifier_labels = '~/git/deep-food/data/recipe_classifier_labels.csv'
df_classifier = pd.read_csv(file_classifier_labels, delimiter=';')
np_classifier = df_classifier.to_numpy()
important_ingredients = np.append(important_ingredients, np_classifier)

file_out_of_fridge_labels = '~/git/deep-food/data/recipe_out_of_fridge.csv'
df_out_of_fridge = pd.read_csv(file_out_of_fridge_labels, delimiter=';')
np_out_of_fridge = df_out_of_fridge.to_numpy()
important_ingredients = np.append(important_ingredients, np_out_of_fridge)

important_ingredients_ids = []

for ingr in important_ingredients:
    idx_ingr = ingredients[ingr]
    #idx_ingr = np.where(ingredients == important_ingredients[ingr])
    important_ingredients_ids.append(idx_ingr)

# important_ingredients = np.random.randint(MAX_NR_INGREDIENTS, size=nr_important_ingredients)
nr_important_ingredients = len(important_ingredients_ids) #200
important_ingredients = np.array(range(nr_important_ingredients))

print("... creating a one_hotted catalogue of the important ingredients")

one_hot_catelogue = [[0]] * (MAX_NR_INGREDIENTS + 1)

#for i in range(len(important_ingredients)):
#    ingredient_id = important_ingredients[i]
#    one_hot_catelogue[ingredient_id] = one_hot(i, nr_important_ingredients)
for i in range(len(important_ingredients_ids)):
    ingredient_id = important_ingredients_ids[i]
    one_hot_catelogue[ingredient_id] = one_hot(i, nr_important_ingredients)

one_hot_catelogue = [np.array(x) for x in one_hot_catelogue]
one_hot_catelogue = np.array(one_hot_catelogue)

print("... creating the catalogue is finished.\n")


print("... one hot encoding the recipes")

# one liner which I wish it would work. The data is not clean
# one_hotted_recipes = [np.sum(one_hot_catelogue[recipe], axis=0) for recipe in recipes]

one_hotted_recipes = [[]] * len(recipes)
for i in range(len(recipes)):

    # if the recipe number i is not empty
    if (len(recipes[i]) > 0):
        one_hotted_recipes[i] = np.sum(one_hot_catelogue[recipes[i]], axis=0)

        # if the recipe does not have any of the important ingredients
        # in that case the one_hotted version is np.array([0])
        if np.array_equal(one_hotted_recipes[i], np.array([0])):
            one_hotted_recipes[i] = np.zeros(nr_important_ingredients)
    else:
        one_hotted_recipes[i] = np.zeros(nr_important_ingredients)
        print("...     the recipe nr.", i, "is empty")

print("... one hot encoding finished.\n")

# converting one_hotted_recipes to integer

print("... converting the one hotted recipes to boolean for minimal space.")

one_hotted_recipes = np.array(one_hotted_recipes).astype(int)

print(one_hotted_recipes.shape)

print("size of the one hotted recipes (as integers) is", one_hotted_recipes.size * one_hotted_recipes.itemsize / 1_048_576, "Mb")

one_hotted_recipes_bool = one_hotted_recipes > 0

print("size of the one hotted recipes (as boolean) is", one_hotted_recipes_bool.size * one_hotted_recipes_bool.itemsize / 1_048_576, "Mb")

np.savetxt('recipes_one_hotted.gz', one_hotted_recipes_bool, fmt='%.1i')
np.savetxt('./important_ingredients.gz', important_ingredients, fmt='%.i')


print("... finding the similar recipes to the fridge")
target_recipe = sample(range(nr_important_ingredients), k=6)
target_recipe_one_hotted = np.sum(one_hot_catelogue[target_recipe], axis=0)
similarity = np.sum(one_hotted_recipes * target_recipe_one_hotted, axis=1)
max_sim = np.max(similarity)
similar_recipes = recipes[similarity == max_sim]


print("... ", len(similar_recipes), "similar recipes are found")

print("... now lets find the missing ingredients")

diffs = [[]] * len(similar_recipes)
diff_in_important_ingredients = [[]] * len(similar_recipes)
for counter, recipe in enumerate(similar_recipes):
    diffs[counter] = set(recipe) - set(target_recipe)
    diff_in_important_ingredients[counter] = diffs[counter].intersection(set(important_ingredients))
    diffs[counter] = np.array(list(diffs[counter]))
    diff_in_important_ingredients[counter] = np.array(list(diff_in_important_ingredients[counter]))
