import numpy as np
import pandas as pd


def find_similar_recipes(target_recipe_txt):
    '''
    '''
    print("... loading the datasets")

    ingredients = np.load('ingredients_clean.npy', allow_pickle=True).item()

    one_hot_catelogue = np.load('one_hot_cat.npy', allow_pickle=True)
    one_hotted_recipes_bool = np.load('recipes_one_hotted.npy', allow_pickle=True)

    print("... finding the similar recipes to the fridge")
    target_recipe = [ingredients[i] for i in target_recipe_txt]
    target_recipe_one_hotted = np.sum(one_hot_catelogue[target_recipe], axis=0)
    similarity = np.sum(one_hotted_recipes_bool * target_recipe_one_hotted, axis=1)
    max_sim = np.max(similarity)

    df = pd.read_csv('recipes_1M_shortened.csv')

    all_ids = np.array(list(range(df.shape[0])))

    similar_ids = []

    while (len(similar_ids) < 3 and max_sim >= 0):
        similar_ids = similar_ids + list(all_ids[similarity == max_sim])
        max_sim = max_sim - 1

    final_suggestions = df.iloc[similar_ids[0:3]]
    # inverse = {v: k for (k, v) in ingredients.items()}
    # print([inverse[i] for i in target_recipe])
    return final_suggestions
