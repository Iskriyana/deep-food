import numpy as np
import pandas as pd
from pathlib import Path

def find_similar_recipes(target_recipe_txt):
    '''
    '''
    print("... loading the datasets")
    
    path_ingr = str(Path().absolute().parents[0]/'data/recipe_data_sets/output/ingredients_clean.npy')
    ingredients = np.load(path_ingr, allow_pickle=True).item()

    one_hot_catelogue_path = str(Path().absolute().parents[0]/'recipes/input/one_hot_cat.npy')
    one_hot_catelogue = np.load(one_hot_catelogue_path, allow_pickle=True)
    
    one_hotted_recipes_bool_path = str(Path().absolute().parents[0]/'recipes/input/recipes_one_hotted.npy')
    one_hotted_recipes_bool = np.load(one_hotted_recipes_bool_path, allow_pickle=True)

    print("... finding the similar recipes to the fridge")
    target_recipe = [ingredients[i] for i in target_recipe_txt]
    target_recipe_one_hotted = np.sum(one_hot_catelogue[target_recipe], axis=0)
    similarity = np.sum(one_hotted_recipes_bool * target_recipe_one_hotted, axis=1) / (np.sum(one_hotted_recipes_bool, axis=1)+0.00001)
    #max_sim = np.max(similarity)

    path_rec = str(Path().absolute().parents[0]/'data/recipe_data_sets/output/recipes_1M_shortened.csv')
    df = pd.read_csv(path_rec)

    #all_ids = np.array(list(range(df.shape[0])))

    #similar_ids = []

    #while (len(similar_ids) < 3 and max_sim >= 0):
    #    similar_ids = similar_ids + list(all_ids[similarity == max_sim])
    #    max_sim = max_sim - 1
        
    ###
    tmp = np.argsort(similarity)[-3:]

    #final_suggestions = df.iloc[similar_ids[0:3]]
    final_suggestions = df.iloc[tmp]
    
    # inverse = {v: k for (k, v) in ingredients.items()}
    # print([inverse[i] for i in target_recipe])
    return final_suggestions
