from keras import utils as kutils
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt


#data
X_train = np.load("data/MNIST_X_train.npy")
y_train = np.load("data/MNIST_y_train.npy")
X_test  = np.load("data/MNIST_X_test.npy")
y_test  = np.load("data/MNIST_y_test.npy")

def preprocess_data(X, y):
   """ Manipulate X and y from their given form in the MNIST dataset to
       whatever form your model is expecting."""

   # This function does nothing but return what it is sent

   return X/np.argmax(X), to_categorical(y)

def make_model(filters, activation_function='sigmoid', addlayers = 0):
    """ relu activation, mse as loss and adam optimizer"""
    #Dense is connecting the nodes to the next layer 
    model = Sequential()
    model.add(Input(shape=(28,28,1)))
    model.add(Conv2D(filters, kernel_size = (3,3), activation = activation_function))
    if addlayers > 0:
        for i in range(1, addlayers):
              model.add(Conv2D(filters*(i * 2), kernel_size = (3,3), activation = activation_function))

    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dropout(.15))

    model.add(Dense(10, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def train_model(X, y, model, epochs, batch_size, IG = False):
    ''' train the model for specified number of epochs, batch_size'''
    if IG:
        X = X.reshape(X.shape[0], 28,28,1)
        dataGen = ImageDataGenerator(rotation_range=15,width_shift_range=0.2,height_shift_range=0.2, 
            shear_range=0.15,zoom_range=[0.5,2],validation_split=0.2)
        dataGen.fit(X)

        train_generator = dataGen.flow(X, y, batch_size=64, shuffle=True,
        seed=2, save_to_dir=None, subset='training')
        validation_generator = dataGen.flow(X, y, batch_size=64, shuffle=True,
        seed=2, save_to_dir=None, subset='validation')

        filepath_val_acc="model_gen_cnnALL_acc.best.hdf5"
        checkpoint_val_acc = ModelCheckpoint(filepath_val_acc, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_val_acc]

        h = model.fit(train_generator, steps_per_epoch = 600, 
            epochs=epochs, validation_data = validation_generator, 
            validation_steps = 150, callbacks = callbacks_list, batch_size = batch_size)
    else:
        h = model.fit(X, y, validation_split=0.2,
                   epochs=epochs,
                   batch_size=batch_size,
                   verbose=1)


    return model,h


if __name__ == "__main__":
    #Test Build and predict 
    epochs    = 15
    batch_size = 1000
    numfilters  = 25

    xtrain, ytrain = preprocess_data(X_train, y_train)
    xtest, ytest = preprocess_data(X_test, y_test)

    dev = tf.config.list_physical_devices('GPU')
    print(f"===================\n {len(dev)} \n===================")
    tf.config.experimental.set_memory_growth(dev[0], True)
    with tf.device("/GPU:0"):
        
        modelNumFilters = make_model(50, activation_function='relu')
        modelNumFilters, hNF = train_model(xtrain, ytrain, modelNumFilters, epochs, batch_size)

        modelNumLayers = make_model(numfilters, activation_function='relu', addlayers=2)
        modelNumLayers, hNL = train_model(xtrain, ytrain, modelNumLayers, epochs, batch_size)

        modelImageGen = make_model(numfilters, activation_function='relu')
        modelImageGen, hIG = train_model(xtrain, ytrain, modelImageGen, epochs, batch_size, IG = True)

    plt.figure(figsize=(20,5))
    plt.plot(hNF.history['loss'])
    plt.plot(hNF.history['val_loss'])

    plt.plot(hNL.history['loss'])
    plt.plot(hNL.history['val_loss'])

    plt.plot(hIG.history['loss'])
    plt.plot(hIG.history['val_loss'])
    plt.title('Figure 1 Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['cnn1 loss', 'cnn1 val_loss', 'cnn2 loss', 'cnn2 val_loss','cnn3 loss', 'cnn3 val_loss',], loc='upper right')
    plt.savefig("Loss.pdf")

    plt.figure(figsize=(20,5))
    plt.plot(hNF.history['accuracy'])
    plt.plot(hNF.history['val_accuracy'])

    plt.plot(hNL.history['accuracy'])
    plt.plot(hNL.history['val_accuracy'])

    plt.plot(hIG.history['accuracy'])
    plt.plot(hIG.history['val_accuracy'])
    plt.title('Figure 2 Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['cnn1 loss', 'cnn1 val_loss', 'cnn2 loss', 'cnn2 val_loss','cnn3 loss', 'cnn3 val_loss',], loc='lower right')
    plt.savefig("Acc.pdf")
    
    # print("**********************\n")
    # a = np.array(hNF.history['accuracy'])
    # print(f"shape = {a.shape}   max = {np.max(a)}   and  {a}")
    # b = modelNumFilters.count_params()
    # print(f"weights = {np.sum(b)}  and {b}")
    plt.figure(figsize=(20,5))
    plt.scatter(modelNumFilters.count_params(), np.max(np.array(hNF.history['accuracy'])), c='blue', marker='o', label = 'cnn1')
    plt.scatter(modelNumLayers.count_params(), np.max(np.array(hNL.history['accuracy'])), c='red', marker='s', label = 'cnn2')
    plt.scatter(modelImageGen.count_params(), np.max(np.array(hIG.history['accuracy'])), c='green', marker='v', label = 'cnn3')
    plt.title('Figure 3 Memory(params) to Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Num Parameters')
    plt.legend(loc='lower right')
    plt.savefig("memVSacc.pdf")
    plt.show()
