import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


print("Welcome to the Handwritten Digits Recognition program")

# Decide whether to load an existing model or train a new one.
new_model = True

if new_model:
    # dividing and adding samples to the MNIST data set
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizing the data for making length = 1
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Creating neural network model
    # Adding one flattened input layer for the pixels, two dense hidden layers and one dense output layer for the 10 digits
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # model compilation and improvement
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # educating the model
    model.fit(X_train, y_train, epochs=3)

    # assessing the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    model.save('handwritten_digits.model')                     #Saving the model
else:
    model = tf.keras.models.load_model('handwritten_digits.model')        #Loading the model

# Use them to load custom images and predict them
image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The most probable number is {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error with currentimage tryig with another image")
        image_number += 1