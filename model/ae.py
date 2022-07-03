import numpy as np
import tensorflow as tf
import os
import time
import sys
from collections import Counter
from time import strftime, localtime


def batches(l, n):
    for i in range(0, l, n):
        yield range(i, min(l,i+n))

class Deep_Autoencoder(object):
    def __init__(self, sess, input_dim_list, learning_rate):
        self.W_list = []
        self.encoding_b_list = []
        self.decoding_b_list = []
        self.dim_list = input_dim_list
        self.learning_rate = learning_rate
        self.save_path = "./tmp/"+strftime("%a-%d-%b-%Y-%H%M%S", localtime())+"-model.ckpt"
        print("AE structure: ", self.dim_list)

        ## Encoder parameters
        for i in range(len(input_dim_list)-1):
            init_max_value = np.sqrt(6. / (self.dim_list[i] + self.dim_list[i+1]))
            self.W_list.append(tf.Variable(tf.random_uniform([self.dim_list[i], self.dim_list[i+1]],
                                                             np.negative(init_max_value), init_max_value)))
            self.encoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i+1]], -0.1, 0.1)))
        ## Decoder parameters
        for i in range(len(input_dim_list)-2,-1,-1):
            self.decoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i]], -0.1, 0.1)))
        ## Placeholder for input
        self.x = tf.placeholder(tf.float32,[None,self.dim_list[0]])
        self.saver = tf.train.Saver()

    def pre_nn_constructing(self, sess, i, saver, regu_coef):
        self.i = i
        print("layer-by-layer pretraining:",len(self.dim_list) - i + 1)
        ## coding graph:
        last_layer = self.x
        for weight,bias,j in zip(self.W_list[:i-1], self.encoding_b_list[:i-1], range(len(self.dim_list))):
            if j == len(self.dim_list) - 1:
                hidden = tf.matmul(last_layer, weight) + bias
            else:
                hidden = tf.sigmoid(tf.matmul(last_layer, weight) + bias)
            last_layer = hidden
        self.hidden = hidden 
        ## decode graph:
        for weight,bias,j in zip(reversed(self.W_list[:i-1]), self.decoding_b_list[1-i:], range(len(self.dim_list))):
            hidden = tf.sigmoid(tf.matmul(last_layer, tf.transpose(weight)) + bias)
            last_layer = hidden
        self.recon = last_layer
        self.cost = 2e2 * tf.reduce_mean(tf.square(self.x - self.recon))
        trainable_var_list = [self.W_list[i-2]]
        trainable_var_list.append(self.encoding_b_list[i-2])
        trainable_var_list.append(self.decoding_b_list[1-i])
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, var_list=trainable_var_list)
        sess.run(tf.global_variables_initializer())
        if i != 2:
            saver.restore(sess, self.save_path) 

    def global_nn_constructing(self, sess, saver, regu_coef):
        last_layer = self.x
        for weight,bias,j in zip(self.W_list, self.encoding_b_list, range(len(self.dim_list))):
            if j == len(self.dim_list) - 1:
                hidden = tf.matmul(last_layer, weight) + bias
            else:
                hidden = tf.sigmoid(tf.matmul(last_layer, weight) + bias)
            last_layer = hidden
        self.hidden = hidden
        print("Global Training.")
        for weight,bias,j in zip(reversed(self.W_list), self.decoding_b_list, range(len(self.dim_list))):
            hidden = tf.sigmoid(tf.matmul(last_layer, tf.transpose(weight)) + bias)
            last_layer = hidden
        self.recon = last_layer
        self.cost = 2e2 * tf.reduce_mean(tf.square(self.x - self.recon))
        self.sample_wise_recon_error = tf.reduce_mean(tf.square(self.x - self.recon), 1)
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, self.save_path) 

    def fit(self, X, saver, sess, learning_rate, iteration=50, batch_size=133, saving=True):
        sample_size = X.shape[0]
        for i in range(iteration):
            for one_batch in batches(sample_size, batch_size):
                sess.run(self.train_step, feed_dict={self.x:X[one_batch]})
            if i%20==0:
                h = self.cost.eval(session = sess,feed_dict = {self.x: X})
                print ("    iteration : ", i, ", cost : ", h)
        if saving:
            save_path = saver.save(sess, self.save_path)

    def transform(self, X, sess):
        return self.hidden.eval(session=sess, feed_dict={self.x: X})

    def getRecon(self, X, sess):
        return self.recon.eval(session=sess, feed_dict={self.x: X})

    def get_sample_wise_recon_error(self, X, sess):
        return self.sample_wise_recon_error.eval(session=sess, feed_dict={self.x: X})

def CalLayersSizes(layers, structure_parameter, input_dim):
    layers_sizes = []
    min_units = 3
    for i in range(layers):
        layers_sizes.append(max(int(input_dim*(structure_parameter**i)), min_units))
    return layers_sizes

def AutoEncoderDimReduction(data, layersNum, compressionRate, iteration, learning_rate, regulationCoefficient):
    tf.keras.backend.clear_session()
    layersSizes = CalLayersSizes(layersNum, compressionRate, data.shape[-1])
    # layersSizes = [data.shape[-1], 500, 500, 2000, 100]
    learning_rate=0.002
    batch_size=128
    with tf.Graph().as_default():
        with tf.Session() as sess:
            ae = Deep_Autoencoder(sess=sess, input_dim_list=layersSizes, learning_rate=learning_rate)
            saver = ae.saver
            for i in range(2, len(layersSizes)):
                ae.pre_nn_constructing(sess=sess, i=i, saver=saver, regu_coef=regulationCoefficient)
                ae.fit(X=data, sess=sess, saver=saver, iteration=iteration, learning_rate=learning_rate, batch_size=batch_size)
            ae.global_nn_constructing(sess=sess, saver=saver, regu_coef=regulationCoefficient)
            ae.fit(X=data, sess=sess, saver=saver, iteration=iteration, learning_rate=learning_rate, batch_size=batch_size)
            hidden = ae.transform(data, sess)
            reconstructionError = ae.get_sample_wise_recon_error(data, sess)
    return hidden, reconstructionError

def AutoEncoderDimReduction2(data, layers, iteration, learning_rate, regulationCoefficient):
    tf.keras.backend.clear_session()
    layersSizes = layers
    # layersSizes = [data.shape[-1], 500, 500, 2000, 100]
    learning_rate=0.002
    batch_size=128
    with tf.Graph().as_default():
        with tf.Session() as sess:
            ae = Deep_Autoencoder(sess=sess, input_dim_list=layersSizes, learning_rate=learning_rate)
            saver = ae.saver
            for i in range(2, len(layersSizes)):
                ae.pre_nn_constructing(sess=sess, i=i, saver=saver, regu_coef=regulationCoefficient)
                ae.fit(X=data, sess=sess, saver=saver, iteration=iteration, learning_rate=learning_rate, batch_size=batch_size)
            ae.global_nn_constructing(sess=sess, saver=saver, regu_coef=regulationCoefficient)
            ae.fit(X=data, sess=sess, saver=saver, iteration=iteration, learning_rate=learning_rate, batch_size=batch_size)
            hidden = ae.transform(data, sess)
            reconstructionError = ae.get_sample_wise_recon_error(data, sess)
    return hidden, reconstructionError

if __name__ == "__main__":
    from util import generatedDataLoading
    data,groundTruth = generatedDataLoading.get_data('USPS')
    # print(data.shape)
    hidden, _ = AutoEncoderDimReduction(data, 4, 0.8, 300, 100)
