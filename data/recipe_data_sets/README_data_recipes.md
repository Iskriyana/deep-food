# Recipes # 


### Finding the most common ingredients ###

* to find the most common ingredients you need to run

```
python 1_find_the_most_common_ingredients.py 100
```
where 100 is the number of ingredients you like to fetch.

* The result will be written in a file named, 'common_ingredients_100.dat' and can be read as

```
nr_comm = 100
filename = 'common_ingredients_' + str(nr_comm) + '.dat'

with open(filename, 'r') as f:
    tmp = json.loads(f.read())
```

* Your list of ingredients are in variable tmp.