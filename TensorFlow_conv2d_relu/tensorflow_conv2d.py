"""
written by jorge orlando miranda ñahui
en este ejemplo se presenta como poder realizar la operación
de convolucion 2d mediante tensorflow utilizando los parametros
de strides=1 y padding ="SAME" .
luego se utiliza la función de activación tf.nn.relu para poder
realizar la operación del Rectified Linear Unit
las operaciónes se realizan en modo eager deshabilitado
"""
import numpy as np
import tensorflow as tf
#modo eager deshabilitado
tf.compat.v1.disable_eager_execution()
#crear grafo para agregar operaciones
grafo=tf.Graph()
#agregar operaciones dentro del grafo
with grafo.as_default():
    #==========ENTRADA===================
    #crear un arreglo como entrada constante
    img=tf.constant([[1,0,3],
                     [1,10,2],
                     [1,10,20]],dtype=tf.float32)
    #dimensiones igual a [batch,height,width,in_channels]
    img=tf.reshape(img,[1,3,3,1])
    #===============================================
    #CREAR KERNEL
    kernel=np.array([[1,0],
                     [0,-1]])
    kernel=tf.Variable(kernel,dtype=tf.float32)
    #dimensiones igual a [height,width,in_channels,out_channels]
    kernel=tf.reshape(kernel,shape=[2,2,1,1])
    #=====================================
    #convolución 2D
    conv1=tf.nn.conv2d(input=img,filters=kernel,data_format="NHWC",strides=1,padding="SAME")
    #función de activación relu
    conv1_relu=tf.nn.relu(conv1,name="RELU")
#crear una sesión para ejecutar las operaciones sobre los tensores

with tf.compat.v1.Session(graph=grafo) as sesion:
    #inicializar variables globales
    sesion.run(tf.compat.v1.global_variables_initializer())
    print("========== INPUT =========")
    print(np.reshape(sesion.run(img),(3,3)))
    print("========= KERNEL ========")
    print(np.reshape(sesion.run(kernel),(2,2)))
    print("========= OUTPUT  CONV2D ========")
    print(np.reshape(sesion.run(conv1),(3,3)))
    print("========= OUTPUT CONV2D + RELU ========")
    print(np.reshape(sesion.run(conv1_relu), (3, 3)))
    #realizar la operación de convolución
