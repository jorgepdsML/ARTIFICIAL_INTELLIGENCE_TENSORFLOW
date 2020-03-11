#!/usr/bin/env python
#coding: utf8 MAC
#written by jorge orlando miranda ñahui
#BRAINTELS LABS
#---ARTIFICIAL INTELLIGENCE AND IMAGE PROCESSING
#---------------COMENZAMOS-------------------------
#Modulo para pasar comandos desde el terminal
import argparse
#crear objeto ArgumentParser()
ap = argparse.ArgumentParser()
#agregar información necesaria al momento de ingresar argumentos
ap.add_argument("-id", "--id", required=True,
	help = "---BRAINTELS LABS--- ")
ap.add_argument("-network", "--network", required=True,
	help="CNN VGG16")
ap.add_argument("-source", "--source", required=True,
	help="NOMBRE DE LA IMAGEN")
#conseguir los elementos que se le pasaron por el terminal
args = vars(ap.parse_args())
print(args)
#------------------------DETERMINAR SI SON CORRECTAS LOS COMANDOS----------------------
if args["id"].lower()=="braintels" and args["network"].lower()=="vgg16":
    #conseguir el nombre de la imagen a cargar
    namefile=args["source"]
    #--IMPORTAR TENSORFLOW
    import tensorflow as tf
    from keras.applications import VGG16
    from keras.applications import imagenet_utils
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    import numpy as np
    import cv2
    import time
    import sys
    MODELS = {"vgg16": VGG16}
    # 224x224
    input_shape = (224, 224)
    # REMOVER LA MEDIA Y NORMALIZAR LOS VALORES
    preprocess = imagenet_utils.preprocess_input
    # CARGAR EL MODELO SELECCIONADO DESDE EL TERMINAL
    modelo = MODELS[args["network"].lower()]
    #MODEO PRE ENTRENADO BASADO EN IMAGENET ILSVRC
    modelo = modelo(weights="imagenet")
    print("BLOQUE TRY--")
    #Cargar imagen
    try:
        imagen=load_img(namefile,target_size=input_shape)
        imagen=np.expand_dims(imagen,axis=0)
        #REMOVER LA MEDIA Y NORMALIZAR LA IMAGEN
        img = preprocess(imagen)
        #determinar el tiempo de predicción
        t1 = time.time()
        #realizar la predicción
        preds = modelo.predict(img)
        #terminar la temporización
        t2 = time.time()
        print("---TIEMPO PARA REALIAR LA PREDICCIÓN----", t2 - t1)
        #----------
        source=cv2.imread(namefile)
        source=cv2.resize(source,(480,480),cv2.INTER_AREA)
        source=np.uint8(source)
        #CONSEGUIR INFORMACIÓN DE LAS CLASES MAS PROBABLES
        P = imagenet_utils.decode_predictions(preds)
        #PARA LAS 5 PRIMERAS CLASES CON MAYOR PROBABILIDAD
        for (i, (id, lab, prob)) in enumerate(P[0]):
            #MOSTRAR DETALLE DE LAS 5 CLASES CON MAYOR PROBABILIDAD
            print("{}. {}: {:.2f}%".format(i + 1, lab, 100 * prob))
        #CONSEGUIR LA CLASE MAS PROBABLE
        (iId, label, pro) = P[0][0]
    #ESCRIBIR EL TEXTO EN LA IMAGEN CON LA CLASE IDENTIFICADA
        cv2.putText(source, "Objeto: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX
              , 0.9, (30, 0, 255), 3)
        cv2.imshow("---RECONOCIMIENTO DE OBJETOS CON CNN VGG-16---",source)
    #PRESIONAR UNA TECLA
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("----FIN DEL PROGRAMA ----","ARQUITECTURA CNN VGG-16")
    except:
        print("ERROR AL CARGAR LA IMAGEN")
  
 



