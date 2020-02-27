#written by jorge orlando miranda ñahui
#ARTIFICIAL INTELLIGENCE AND IMAGE PROCESSING
#github => jorgepdsML
import tensorflow as tf
import numpy as np
from keras.initializers import Zeros,Ones
from keras.models import load_model,Sequential
from keras.layers import Conv2D , MaxPooling2D,Dropout,BatchNormalization
#crear el modelo Sequential
modelo=Sequential()
conv1=Conv2D(filters=1,
             kernel_size=(3,3),
             padding="same",
             strides=1,
             data_format="channels_last",
             input_shape=(4,4,1),
             kernel_initializer=Ones())

max1=MaxPooling2D(
             pool_size=(2,2),
             data_format="channels_last")
#----------------AGREGAR CAPA DE CONVOLUCION CONV2D
modelo.add(conv1)
#----------------AGREGAR CAPA DE MAXPOOLING2D
modelo.add(max1)
#-----------------METODO COMPILE PARA ESPECIFICAR PARAMETROS EN EL ENTRENAMIENTO
modelo.compile(optimizer="rmsprop",
                loss="categorical_crossentropy",
                metrics=["accuracy"])


#------MATRIZ DE PRUEBA  (1,4,4,1) = > (BATCH,FILAS,COLUMNAS,CANALES)
img=np.array([[1,1,1,0],[1,0,2,3],[10,3,5,6],[10,2,3,4]])
img=np.reshape(img,(4,4,1))
img=np.expand_dims(img,axis=0)
#-----------------METODO PREDICT()
resultado=modelo.predict(img)
#---------------- RESULTADO DE LA OPERACION----
print(np.reshape(resultado,(2,2)))

#--------------GUARDAR EL MODELO
modelo.save("braintels_model.hdf5")
