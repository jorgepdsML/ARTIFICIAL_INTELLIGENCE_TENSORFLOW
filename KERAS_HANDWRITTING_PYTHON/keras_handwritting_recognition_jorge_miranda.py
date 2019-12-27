#written by jorge orlando miranda ñahui
#import the neccesary modules
import numpy as np
import mnist
from keras.models import  Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from keras.utils import to_categorical
#load the data from the mnits data set
#training images (60000 records)
train_images=mnist.train_images()
train_labels=mnist.train_labels()
#testing images (10000 records)
test_images=mnist.test_images()
test_labels=mnist.test_labels()
#normalize the values to the range of [-0.5 0.5]
norm_train=train_images/255 -0.5
test_images=test_images/255 -0.5
norm_train = np.expand_dims(norm_train, axis=3)
test_images = np.expand_dims(test_images, axis=3)
CNN_model=Sequential()
#number of filters  (9)
num_filters=9
#size of each filter (3x3)
filter_size=3
#dimension of input 28x28x1 (grayscale image)
shape_input=(28,28,1)
CNN_model.add(Conv2D(num_filters, filter_size, input_shape=shape_input))
#size of the pooling layer (2x2)
size_pool=2
#max Polling layer
CNN_model.add(MaxPooling2D(pool_size=size_pool))
#add Flatten to the model
CNN_model.add(Flatten())
#add a hidden layer  using the relu activation function
CNN_model.add(Dense(100,activation="relu"))
#add a fully connected neural network using softmax activation function
CNN_model.add(Dense(10,activation="softmax"))
#configure the model for the training of the CNN
#------------compile method------------------------------
#algorithm
optimizer="adam"
#cost function
loss_="categorical_crossentropy"
#metric
metric=["accuracy"]

CNN_model.compile(optimizer,loss=loss_,metrics=metric)
#-------------------------fit method-----------------
#use of the fit method for the training of the CNN
#number of epochs , you can tailor it for your application
#the more epochs there are , the longer the algorithm will take
N_epochs=3

#number of data points for using in the batch
Batch_size=50
#use of to_categorical to convert integer data to a binary matrix data to use in the fit method
CNN_model.fit(norm_train,to_categorical(train_labels),epochs=N_epochs,batch_size=Batch_size,shuffle=True,validation_data=(test_images, to_categorical(test_labels)))

#save the weights in a hdf5 file to use in other scripts
CNN_model.save_weights("pesos_w.h5")

#use of the testing data
predi=CNN_model.predict(test_images)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
#compute the confusion matrix
Matrix=confusion_matrix(test_labels,np.argmax(predi,axis=1))
print("CONFUSION MATRIX Nclasses x Nclasses ")
print(Matrix)
#accuracy
accuracy=accuracy_score(test_labels,np.argmax(predi,axis=1))
print("ACCURACY ",accuracy)
#--------------------finish-------------
#youtube channel jorge miranda redes neuronales
