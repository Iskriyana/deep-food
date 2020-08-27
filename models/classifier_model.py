import numpy as np
import tensorflow as tf
from tensorflow import keras


class single_ingredient_classifier():
    '''
    Classifier model based on ResNet50 to classify single ingredient from an image
    '''
    def __init__(self, input_shape=(150, 150, 3), num_basic_ingredients=5):
        '''
        instantiate the model based on ResNet50

        keyword arguments:
        input_shape -- the shape of the input image
        num_basic_ingredients -- number of categories of the ingredients.
        
        returns:
        an un-compiled model for classification.
        The model outputs an array of size num_basic_ingredients with the probabilities associated with each ingredient
        '''
        self.input_shape = input_shape
        self.num_basic_ingredients = num_basic_ingredients
        
        # Get the base_model without its top
        base_model = keras.applications.ResNet50(
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=self.input_shape,
            include_top=False)  # Do not include the ImageNet classifier at the top.


        # The input layer
        inputs = keras.Input(shape=self.input_shape)

        # The normalization layer
        # ToDo: check that the ResNet is getting images normalized to -1, 1
        norm_layer = keras.layers.experimental.preprocessing.Normalization()
        mean = np.array([127.5] * 3)
        var = mean ** 2
        # Scale inputs to [-1, +1]
        x = norm_layer(inputs)
        norm_layer.set_weights([mean, var])

        # No image augumentation is included.
        # It shouldnt be difficult to find enough images

        # the basel model should run in training=False mode
        x = base_model(x, training=False)
        # Convert features of shape `base_model.output_shape[1:]` to vectors
        x = keras.layers.GlobalAveragePooling2D()(x)
        # Classifier with returns the probabilities
        outputs = keras.layers.Dense(num_basic_ingredients, activation='softmax')(x)
        self.model = keras.Model(inputs, outputs)

    def compile(self):
        '''
        compiles self.model
        '''
        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss=keras.losses.CategoricalCrossentropy(),
                           metrics=[keras.metrics.CategoricalAccuracy()])

    def predict(self, input_image):
        '''
        predict the category of the input_image.
        keyword arguments:
        input_image -- the image to be classified
        returns:
        the most probable class id
        '''
        tmp = self.model(input_image)
        return np.argmax(tmp)

