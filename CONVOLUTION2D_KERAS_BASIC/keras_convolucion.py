"""
PYTHON, NUMPY , TENSORFLOW AND KERAS
written by jorge orlando miranda ñahui
github:https://github.com/jorgepdsML/ARTIFICIAL_INTELLIGENCE_TENSORFLOW/upload/master
youtube: jorge miranda redes neuronales
Fanpage : BRAINTELS LABS FACEBOOK

"""
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
#------------------construir el modelo
modelo=Sequential()
#--------------------Crear capa Conv2D
layer1=Conv2D(filters=1,kernel_size=3,strides=1,padding="valid",input_shape=(4,4,1))
#filters indica la cantidad de filtros
#kernel_size la dimension del kernel 3x3
#strides=1 paso de uno en uno
#padding="valid" datos validos de entrada
#input_shape=(Nfi,Nci,channels_Entrada)
#---------------agregar la capa Conv2D al modelo Sequential
modelo.add(layer1)
#dimensión del kernel (Nfilas,Ncolumnas,Ncanales,Nfilters)
# (3 , 3 , 1 , 1)
#------------------Inicializar el Kernel ConV2D
W=np.array([[1,-1,1],
            [2,5,10],
            [0,1,2]],dtype=float).reshape((3,3,1,1))

Ws=[W,np.array([0.0])]
#--------------- Establecer Kernel Conv2D igual a W
layer1.set_weights(Ws)
#-------------------MATRIZ DE ENTRADA
I=np.array([[10,15,0,10],
            [20,0,0,10],
            [0,1,10,20],
            [0,10,20,10]],dtype=float).reshape((1,4,4,1))
#--------------Realizar la convolución 2D
Ir=modelo.predict(I)
print("RESULTADO DE LA CONVOLUCIÓN ES: \n {} ".format(Ir))
print(np.shape(Ir))
