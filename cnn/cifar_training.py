import tensorflow as tf

from tensorflow.keras import datasets, layers, models, optimizers, losses
import matplotlib.pyplot as plt

MODEL_FILE = 'cifar_model'

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 
                'frog', 'horse', 'ship', 'truck']

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)

#     # The CIFAR labels happen to be arrays,
#     # wich is why you need the extra index
#     plt.xlabel(CLASS_NAMES[train_labels[i][0]])

# plt.show()

# Creating the convolutional base
model = models.Sequential()

# The number of filters will become the channels in the output of the layer
model.add(layers.Conv2D(filters=32,  kernel_size=(4, 4), activation='relu', input_shape=(32, 32, 3)))

# Reducese each a (2, 2) max_pooling reduces each input width and height by half (30 x 30) -> (15 x 15)
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# To do the classification we need to feed the conv operations into one or more simplse dense layers.
# The dense layer expects a 1D vector, so we need to Flatten the conv output.
model.add(layers.Flatten())

# A dense layers with 64 neurons and relu activation function
model.add(layers.Dense(units=64, activation='relu'))

# Output layer (one neuron to each class)
# The default activation is a "linear" function: a(x) = x
model.add(layers.Dense(units=10))

# Build the model and run
model.compile(optimizer=optimizers.Adam(),
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(x=train_images,
            y=train_labels,
            epochs=50,
            validation_data=(test_images, test_labels))

# Saving the trained model 
model.save(filepath=MODEL_FILE)

# Ploting the train history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

# Evaluating the trained model over the test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Evaluation loss: %.2f and acc: %.2f' % (test_loss, test_acc))

