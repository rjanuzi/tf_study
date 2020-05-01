import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import models

MODEL_FILE = 'cifar_model'
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 
                'frog', 'horse', 'ship', 'truck']


# Load model
model = models.load_model(MODEL_FILE)

# Load the images and transform into normalized numpy arrays (3, 32, 32, 3)
charlies = np.asarray([np.asarray(Image.open('charlie.png'))/255.0,
                        np.asarray(Image.open('charlie_2.png'))/255.0,
                        np.asarray(Image.open('charlie_3.png'))/255.0,
                        np.asarray(Image.open('cat.png'))/255.0,
                        np.asarray(Image.open('cat_2.png'))/255.0])

# Predict the classes
predictions = model.predict(charlies)

# Show results
for p in predictions:
    # Each prediction is an array p, where p[i] is the confidence of the model
    # that the input is of class 'i'.
    print('This is probably a %s' % CLASS_NAMES[np.argmax(p)])
