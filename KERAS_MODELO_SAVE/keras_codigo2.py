#written by jorge orlando miranda ñahui
#ARTIFICIAL INTELLIGENCE AND IMAGE PROCESSING
#github => jorgepdsML
import tensorflow as tf
import numpy as np
from keras.models import load_model
#----------CARGAR EL MODELO GUARDADO------------
modelo=load_model("braintels_model.hdf5")
#----------IMAGEN DE ENTRADA DE PRUEBA--------(1,4,4,1)
img=np.array([[1,1,1,0],[1,0,2,3],[10,3,5,6],[10,2,3,4]])
img=np.reshape(img,(4,4,1))
#agregar dimensiones
img=np.expand_dims(img,axis=0)

#---------METODO PREDICT DEL MODELO CARGADO
resultado=modelo.predict(img)
#---------MOSTRAR LOS RESULTADOS -----------------
print(np.reshape(resultado,(2,2)))