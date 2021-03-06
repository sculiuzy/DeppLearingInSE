
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
'''
x_data = np.array([[1,2],[2,1]])

y_data=[[7,7,8,156],[50,70,80,7]]
'''
def return_in_out_data(input_data_path,output_data_path):
    x_data = pd.read_csv(input_data_path).values
    y_data = pd.read_csv(output_data_path).values.T
    return x_data,y_data
    
    
#x_x=np.linspace(1,3,1001).reshape((1,1001))
x_data ,y_data= return_in_out_data('h:\\tensorflowDLInSE\\preprocessingdata\\x_data.csv',
                                   'h:\\tensorflowDLInSE\\preprocessingdata\\y_data.csv')




'''
x_x=np.linspace(1,3,1001).reshape((1,1001))
data=pd.read_csv('h:\\tensorflowDLInSE\\data2.csv')
x_data=[[300,120,300]]
y_data=data['mag'].tolist()
y_data=np.array(y_data).reshape((1,1001))
#y_data[0]=y_d
'''

xs=tf.placeholder(tf.float32,[None,7])
ys=tf.placeholder(tf.float32,[None,1001])

L1= add_layer(xs,7,10,activation_function=tf.nn.sigmoid)
predition=add_layer(L1,10,1001,activation_function=None)

loss = tf.reduce_mean((tf.square(predition-ys)))
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
saver=tf.train.Saver()
sess=tf.Session()
sess.run(init)

for i in range (10000):
    sess.run(train,feed_dict={xs:x_data,ys:y_data})
    if i %500 ==0:
       print(sess.run(predition,feed_dict={xs:[[300,120,300,60,10,10,90]]}))
#aa=sess.run(predition,feed_dict={xs:[[300,120,300]]}).reshape((1001,1))
#np.savetxt("H:\pre_values1.txt",aa)
saver.save(sess,"mynetwork\my_net.ckpt")
aa=sess.run(predition,feed_dict={xs:[[300,120,300,60,10,10,90]]}).reshape((1001,1))
np.savetxt("H:\pre_values1.txt",aa)
print('Its ok!')
