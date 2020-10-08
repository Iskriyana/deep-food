import numpy as np
import itertools
import pandas as pd
import json
import sys

# 1 -- loading the data
with np.load('../input/simplified-recipes-1M.npz', allow_pickle=True) as data:
    recipes = data['recipes']
    ingredients = data['ingredients']

print("data is read")
# an example
# print(ingredients[recipes[0]])


# 2 -- flatten the whole recipes

merged = np.array(list(itertools.chain(*recipes)))

# 3 -- histogram
histo = np.histogram(merged, bins=np.max(merged))

# 4 -- converting the frequencies to a data frame
df = pd.DataFrame({'ingredient': ingredients[histo[1][:-1].astype(int)], 'freq': histo[0]})

# 5 -- sorting to find the most common ones
df.sort_values('freq', ascending=False, inplace=True)

# 6 -- extract the common 100
nr_comm = int(sys.argv[1])

tmp = list(df['ingredient'].iloc[0:nr_comm])
print("the ", nr_comm, " most common ingredients are: \n")
print(tmp)

# 7 -- saving to a file
filename = '../output/common_ingredients_' + str(nr_comm) + '.dat'
with open(filename, 'w') as f:
    f.write(json.dumps(tmp))
print('\nthe info is saved into file: ', filename)

#with open('test.txt', 'r') as f:
#    tmp = json.loads(f.read())
