"""
written by jorge orlando miranda ñahui
lo que se pretende es crear 2 tensores de 5 elementos que
emularan las respuesta de 5 neuronas en la etapa de clasificación
en la cual se utilizara la función tf.equal para determinar
si son iguales y la función tf.cast para convertir los elementos
booleanos en valores enteros y luego cuantos coinciden
"""
import numpy as np
import tensorflow as tf
#modo eager deshabilitado
tf.compat.v1.disable_eager_execution()
#crear grafo para agregar operaciones
grafo=tf.Graph()
#agregar operaciones dentro del grafo
with grafo.as_default():
    y=tf.constant([1,0,0,1,1],dtype=tf.float32,name="y")
    d=tf.constant([0,0,0,1,1],dtype=tf.float32,name="d")
    #comparar mediante la función tf.equal
    comp=tf.equal(y,d,name="comparar")
    #convertir a tf.float32
    m=tf.cast(comp,dtype=tf.float32)
    #determinar una metrica de exactitud
    exactitud=tf.reduce_mean(m)
#crear sesión
with tf.compat.v1.Session(graph=grafo) as s1:
    print("====== y ========")
    print(s1.run(y))
    print("====== d ========")
    print(s1.run(d))
    print("====== ACCURACY ========")
    print(100*s1.run(exactitud),"%")
