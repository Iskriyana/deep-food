Workflow:

1) Get 50 classes and 150 samples for each class

2) Through a resnet34, run the downloaded images through the neural network and get the representation in the vector space.

3) Get rid of duplicates.

In order to get better results, we might need to gather the data like:

1) apple:
  1.1) apple breakfast
  1.2) apple food
  1.3) apple fresh
  1.4) manzana
  1.5) manzana fresca

To get the images for apple, so they are cleaner and be better at generalizing
