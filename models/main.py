import tensorflow as tf
from classifier_model import single_ingredient_classifier
my_model = single_ingredient_classifier()
test_image = tf.keras.backend.random_uniform((1, 150, 150, 3), minval=0.0, maxval=1.0, dtype=None, seed=None)
print(my_model.model.predict(test_image))
print(my_model.predict(test_image))

my_model.compile()

# TODO: 
# 1- Creating a new dataset with the images of ingredients using Michi's image_downloader
# Following https://www.tensorflow.org/tutorials/load_data/images
# the ingrediesnt can be arranged into directories like
# single_ingredient_pics/
#  daisy/
#  dandelion/
#  roses/
#  sunflowers/
#  tulips/
# to use the image downloader one should be careful that preferably each image should have one item
# (include one/a/an in the search with the approriate unit (ex: 1 bowl rice)

# 2- after creaing the datasets one can start training the last layer of the classifier
# and if the performance is not good, start fine-tuning:
# be careful  of points mentioned in https://keras.io/guides/transfer_learning/
# my_model.model.fit(new_dataset, validation_data=??, epochs=10)

