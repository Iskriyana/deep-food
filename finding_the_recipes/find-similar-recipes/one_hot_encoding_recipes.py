import numpy as np
from tensorflow.keras.backend import one_hot
from random import sample

# read this from the recipes, currently hot coded
MAX_NR_INGREDIENTS = 3500

nr_important_ingredients = 200

print("... loading the original dataset")
with np.load('../simplified-recipes-1M.npz', allow_pickle=True) as data:
    recipes = data['recipes']
    ingredients = data['ingredients']
print("... loading finished.\n")

# list of important ingredients
# curently this is random, but there should be a proper input
np.random.seed(1)

# important_ingredients = np.random.randint(MAX_NR_INGREDIENTS, size=nr_important_ingredients)
important_ingredients = np.array(range(nr_important_ingredients))

print("... creating a one_hotted catalogue of the important ingredients")

one_hot_catelogue = [[0]] * (MAX_NR_INGREDIENTS + 1)

for i in range(len(important_ingredients)):
    ingredient_id = important_ingredients[i]
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

print("size of the one hotted recipes (as integers) is", one_hotted_recipes.size * one_hotted_recipes.itemsize / 1_048_576, "Mb")

one_hotted_recipes_bool = one_hotted_recipes > 0

print("size of the one hotted recipes (as boolean) is", one_hotted_recipes_bool.size * one_hotted_recipes_bool.itemsize / 1_048_576, "Mb")

np.savetxt('recipes_one_hotted.gz', one_hotted_recipes_bool, fmt='%.1i')
np.savetxt('./important_ingredients.gz', important_ingredients, fmt='%.i')

#target_recipe = sample(range(nr_important_ingredients), k=6)
#target_recipe_one_hotted = np.sum(one_hot_catelogue[target_recipe], axis=0)
#granular_diffs = one_hotted_recipes_bool - target_recipe_one_hotted
#granular_diffs = (granular_diffs + abs(granular_diffs)) / 2
#aggregated_diffs = np.sum(granular_diffs, axis=1)
#exact_recipes = recipes[aggregated_diffs == 0]
