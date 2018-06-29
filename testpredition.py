import tensorflow as  tf
import numpy as np
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
xs=tf.placeholder(tf.float32,[None,7])
ys=tf.placeholder(tf.float32,[None,1001])

L1= add_layer(xs,7,10,activation_function=tf.nn.sigmoid)
predition=add_layer(L1,10,1001,activation_function=None)

loss = tf.reduce_mean((tf.square(predition-ys)))
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
saver=tf.train.Saver()

sess= tf.Session()
saver.restore(sess,tf.train.latest_checkpoint ('mynetwork'))
aa=sess.run(predition,feed_dict={xs:[[300,120,300,60,10,10,90]]}).T
np.savetxt("H:\pre_values2.txt",aa)
print('Its ok la')
