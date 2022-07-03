import PIL.Image as Image
import numpy as np
import tensorflow as tf
import os
import time
import sys
from time import strftime, localtime
from collections import Counter


def batches(l, n):
    for i in range(0, l, n):
        yield range(i, min(l,i+n))

class Deep_Autoencoder(object):
    def __init__(self, sess, input_dim_list, num_neighbours, learning_rate):
        assert len(input_dim_list) >= 2
        self.W_list = []
        self.encoding_b_list = []
        self.decoding_b_list = []
        self.dim_list = input_dim_list
        self.num_neighbours = num_neighbours
        self.learning_rate = learning_rate
        self.save_path = "./tmp/"+strftime("%a-%d-%b-%Y-%H%M%S", localtime())+"-model.ckpt"
        print("NNAE structure: ", self.dim_list)
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
        self.input_x = tf.placeholder(tf.float32,[None,self.num_neighbours*2+1,self.dim_list[0]])
        self.x = self.input_x[:,0,:]
        self.neighbors_1 = self.input_x[:,1:self.num_neighbours+1,:]
        self.neighbors_2 = self.input_x[:,self.num_neighbours+1:,:]
        self.saver = tf.train.Saver()

    def calculate_pointstructure_maintaining_regularization_1(self, pretraining):
        i = self.i
        last_layer_neighbors = tf.reshape(self.neighbors_1, (-1,self.dim_list[0]))
        if pretraining:
            for weight,bias,j in zip(self.W_list[:i-1], self.encoding_b_list[:i-1], range(len(self.dim_list))):
                hidden_neighbors = tf.sigmoid(tf.matmul(last_layer_neighbors, weight) + bias)
                last_layer_neighbors = hidden_neighbors
        else:
            for weight,bias,j in zip(self.W_list, self.encoding_b_list, range(len(self.dim_list))):
                hidden_neighbors = tf.sigmoid(tf.matmul(last_layer_neighbors, weight) + bias)
                last_layer_neighbors = hidden_neighbors
        self.hidden_neighbors = tf.reshape(hidden_neighbors, [-1, self.num_neighbours, self.dim_list[i-1]])
        return tf.reduce_mean(tf.square(tf.expand_dims(self.hidden, 1) - self.hidden_neighbors))

    def calculate_pointstructure_maintaining_regularization_2(self, pretraining):
        i = self.i
        last_layer_neighbors = tf.reshape(self.neighbors_2, (-1,self.dim_list[0]))
        if pretraining:
            for weight,bias,j in zip(self.W_list[:i-1], self.encoding_b_list[:i-1], range(len(self.dim_list))):
                hidden_neighbors = tf.sigmoid(tf.matmul(last_layer_neighbors, weight) + bias)
                last_layer_neighbors = hidden_neighbors
        else:
            for weight,bias,j in zip(self.W_list, self.encoding_b_list, range(len(self.dim_list))):
                hidden_neighbors = tf.sigmoid(tf.matmul(last_layer_neighbors, weight) + bias)
                last_layer_neighbors = hidden_neighbors
        self.hidden_neighbors = tf.reshape(hidden_neighbors, [-1, self.num_neighbours, self.dim_list[i-1]])
        return tf.reduce_mean(tf.square(tf.expand_dims(self.hidden, 1) - self.hidden_neighbors))

    def pre_nn_constructing(self, sess, i, saver, training_with_regu, regu1, regu2):
        self.i = i
        print("layer-by-layer pretraining:",len(self.dim_list) - i + 1)
        ## coding graph:
        last_layer = self.x
        for weight,bias,j in zip(self.W_list[:i-1], self.encoding_b_list[:i-1], range(len(self.dim_list))):
            if j == len(self.dim_list) - 1:
                hidden = tf.matmul(last_layer, weight) + bias
            else:
                hidden = tf.sigmoid(tf.matmul(last_layer, weight) + bias)
            ### *** ###
            # hidden = tf.sigmoid(tf.matmul(last_layer, weight) + bias)

            last_layer = hidden
        self.hidden = hidden 
        ## decode graph:
        for weight,bias,j in zip(reversed(self.W_list[:i-1]), self.decoding_b_list[1-i:], range(len(self.dim_list))):
            hidden = tf.sigmoid(tf.matmul(last_layer, tf.transpose(weight)) + bias)
            last_layer = hidden
        self.recon = last_layer
        self.recon_cost = 2e2 * tf.reduce_mean(tf.square(self.x - self.recon))
        self.regularization_cost_1 = regu1 * self.calculate_pointstructure_maintaining_regularization_1(True)
        self.regularization_cost_2 = regu2 * self.calculate_pointstructure_maintaining_regularization_2(True)
        self.cost = self.recon_cost + self.regularization_cost_1 - self.regularization_cost_2
        if not training_with_regu:
            self.cost = 2e2 * tf.reduce_mean(tf.square(self.x - self.recon))
        trainable_var_list = [self.W_list[i-2]]
        trainable_var_list.append(self.encoding_b_list[i-2])
        trainable_var_list.append(self.decoding_b_list[1-i])
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, var_list=trainable_var_list)
        sess.run(tf.global_variables_initializer())
        if i != 2:
            saver.restore(sess, self.save_path) 

    def global_nn_constructing(self, sess, saver, training_with_regu, regu1, regu2):
        last_layer = self.x
        for weight,bias,j in zip(self.W_list, self.encoding_b_list, range(len(self.dim_list))):
            if j == len(self.dim_list) - 1:
                hidden = tf.matmul(last_layer, weight) + bias
            else:
                hidden = tf.sigmoid(tf.matmul(last_layer, weight) + bias)
            ### *** ###
            # hidden = tf.sigmoid(tf.matmul(last_layer, weight) + bias)

            last_layer = hidden
        self.hidden = hidden
        print("Global Training.")
        for weight,bias,j in zip(reversed(self.W_list), self.decoding_b_list, range(len(self.dim_list))):
            hidden = tf.sigmoid(tf.matmul(last_layer, tf.transpose(weight)) + bias)
            last_layer = hidden
        self.recon = last_layer
        self.recon_cost = 2e2 * tf.reduce_mean(tf.square(self.x - self.recon))
        self.regularization_cost_1 = regu1 * self.calculate_pointstructure_maintaining_regularization_1(False)
        self.regularization_cost_2 = regu2 * self.calculate_pointstructure_maintaining_regularization_2(False)
        self.cost = self.recon_cost + self.regularization_cost_1 - self.regularization_cost_2
        if not training_with_regu:
            self.cost = 2e2 * tf.reduce_mean(tf.square(self.x - self.recon))
        self.sample_wise_recon_error = tf.reduce_mean(tf.square(self.x - self.recon), 1)
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, self.save_path) 

    def fit(self, X, saver, sess, iteration=50, batch_size=133, saving=True):
        assert X.shape[2] == self.dim_list[0]
        sample_size = X.shape[0]
        for i in range(iteration):
            for one_batch in batches(sample_size, batch_size):
                sess.run(self.train_step, feed_dict={self.input_x:X[one_batch]})
            if i%20==0:
                f = self.recon_cost.eval(session=sess, feed_dict={self.input_x:X})
                g1 = self.regularization_cost_1.eval(session=sess, feed_dict={self.input_x:X})
                g2 = self.regularization_cost_2.eval(session=sess, feed_dict={self.input_x:X})
                h = self.cost.eval(session = sess,feed_dict = {self.input_x: X})
                print ("    itr : ", i, ", r : ", f, ", g1 : ", g1, ", g2 : ", g2, ", cost : ", h)
        if saving:
            save_path = saver.save(sess, self.save_path)

    def transform(self, X, sess):
        return self.hidden.eval(session=sess, feed_dict={self.input_x: X})

    def getRecon(self, X, sess):
        return self.recon.eval(session=sess, feed_dict={self.input_x: X})

    def get_sample_wise_recon_error(self, X, sess):
        return self.sample_wise_recon_error.eval(session=sess, feed_dict={self.input_x: X})

def CalLayersSizes(layers, structure_parameter, input_dim):
    layers_sizes = []
    min_units = 3
    for i in range(layers):
        layers_sizes.append(max(int(input_dim*(structure_parameter**i)), min_units))
    return layers_sizes

def NNAEDimReduction(data, layersNum, compressionRate, iteration, learning_rate, regu1, regu2):
    tf.keras.backend.clear_session()
    batch_size=128
    assert len(data.shape) == 3
    layers_sizes = CalLayersSizes(layersNum, compressionRate, data.shape[-1])
    # layers_sizes = [data.shape[-1], 500, 500, 2000, 100]
    num_neighbours = (data.shape[1] - 1)/2
    assert num_neighbours == int(num_neighbours)
    num_neighbours = int(num_neighbours)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            ae = Deep_Autoencoder(sess=sess, input_dim_list=layers_sizes, num_neighbours=num_neighbours, learning_rate=learning_rate)
            saver = ae.saver
            for i in range(2, len(layers_sizes)+1):
                ae.pre_nn_constructing(sess=sess, i=i, saver=saver, training_with_regu=True, regu1=regu1, regu2=regu2)
                ae.fit(X=data, sess=sess,
                                saver=saver,
                                iteration=iteration,
                                batch_size=batch_size,
                                )
            ae.global_nn_constructing(sess=sess, saver=saver, training_with_regu=True, regu1=regu1, regu2=regu2)
            ae.fit(X=data, saver=saver,
                        sess=sess,
                        iteration=iteration,
                        batch_size=batch_size,
                        )
            hidden = ae.transform(data, sess)
            recon_error = ae.get_sample_wise_recon_error(data, sess)
    return hidden, recon_error

def NNAEDimReduction2(data, layers, iteration, learning_rate, regulationCoefficient):
    tf.keras.backend.clear_session()
    batch_size=128
    assert len(data.shape) == 3
    layers_sizes = layers
    num_neighbours = data.shape[1] - 1
    with tf.Graph().as_default():
        with tf.Session() as sess:
            ae = Deep_Autoencoder(sess=sess, input_dim_list=layers_sizes, num_neighbours=num_neighbours, learning_rate=learning_rate)
            saver = ae.saver
            for i in range(2, len(layers_sizes)+1):
                ae.pre_nn_constructing(sess=sess, i=i, saver=saver, training_with_regu=True, regu_coef=regulationCoefficient)
                ae.fit(X=data, sess=sess,
                                saver=saver,
                                iteration=iteration,
                                batch_size=batch_size,
                                )
            ae.global_nn_constructing(sess=sess, saver=saver, training_with_regu=True, regu_coef=regulationCoefficient)
            ae.fit(X=data, saver=saver,
                        sess=sess,
                        iteration=iteration,
                        batch_size=batch_size,
                        )
            hidden = ae.transform(data, sess)
            recon_error = ae.get_sample_wise_recon_error(data, sess)
    return hidden, recon_error

def AEDimReduction(data, layersNum, compressionRate, iteration, learning_rate, regulationCoefficient):
    tf.keras.backend.clear_session()
    batch_size=128
    assert len(data.shape) == 3
    layers_sizes = CalLayersSizes(layersNum, compressionRate, data.shape[-1])
    # layers_sizes = [data.shape[-1], 500, 500, 2000, 100]
    num_neighbours = data.shape[1] - 1
    with tf.Graph().as_default():
        with tf.Session() as sess:
            ae = Deep_Autoencoder(sess=sess, input_dim_list=layers_sizes, num_neighbours=num_neighbours, learning_rate=learning_rate)
            saver = ae.saver
            for i in range(2, len(layers_sizes)+1):
                ae.pre_nn_constructing(sess=sess, i=i, saver=saver, training_with_regu=False, regu_coef=regulationCoefficient)
                ae.fit(X=data, sess=sess,
                                saver=saver,
                                iteration=iteration,
                                batch_size=batch_size,
                                )
            ae.global_nn_constructing(sess=sess, saver=saver, training_with_regu=False, regu_coef=regulationCoefficient)
            ae.fit(X=data, saver=saver,
                        sess=sess,
                        iteration=iteration,
                        batch_size=batch_size,
                        )
            hidden = ae.transform(data, sess)
            recon_error = ae.get_sample_wise_recon_error(data, sess)
    return hidden, recon_error, ae.save_path

def NNAEDimReductionFT(data, layersNum, compressionRate, iteration, learning_rate, regulationCoefficient, path):
    tf.keras.backend.clear_session()
    batch_size=128
    assert len(data.shape) == 3
    layers_sizes = CalLayersSizes(layersNum, compressionRate, data.shape[-1])
    # layers_sizes = [data.shape[-1], 500, 500, 2000, 100]
    num_neighbours = data.shape[1] - 1
    with tf.Graph().as_default():
        with tf.Session() as sess:
            ae = Deep_Autoencoder(sess=sess, input_dim_list=layers_sizes, num_neighbours=num_neighbours, learning_rate=learning_rate)
            saver = ae.saver
            ae.save_path = path
            ae.i = len(layers_sizes)
            ae.global_nn_constructing(sess=sess, saver=saver, training_with_regu=True, regu_coef=regulationCoefficient)
            ae.fit(X=data, saver=saver,
                        sess=sess,
                        iteration=iteration,
                        batch_size=batch_size,
                        )
            hidden = ae.transform(data, sess)
            recon_error = ae.get_sample_wise_recon_error(data, sess)
    return hidden, recon_error

def ae_sampling_nnae_finetune(
        data, 
        k, 
        ae_layersNum, 
        ae_compressionRate, 
        partition_size_list, 
        prob_list, 
        iteration, 
        learning_rate, 
        regulationCoefficient
        ):
    import nn
    data_nn, path = nn.find_nearest_neighbors_with_ae_anomaly_score_3_partitions_sampling_ft(
        data=data, 
        k=k, 
        ae_layersNum=ae_layersNum, 
        ae_compressionRate=ae_compressionRate, 
        iteration=iteration, 
        partition_size_list=partition_size_list, 
        prob_list=prob_list)
    return NNAEDimReductionFT(data_nn, ae_layersNum, ae_compressionRate, iteration, learning_rate, regulationCoefficient, path)

def nnae(data, layers_sizes=[784,400,200,100,50,25,6],
                    iteration = 80,
                    learning_rate = 0.0020,
                    # learning_rate = 0.0025,
                    batch_size = 128,
                    training_with_regu=True,
                    regu_coef=100):
    num_neighbours = data.shape[1]-1
    layers_sizes[0] = data.shape[2]
    print(data.shape)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            ae = Deep_Autoencoder(sess=sess, input_dim_list=layers_sizes, num_neighbours=num_neighbours)
            saver = ae.saver
            for i in range(2, len(layers_sizes)+1):
                ae.pre_nn_constructing(sess=sess, i=i, saver=saver, training_with_regu=True, regu_coef=regu_coef)
                ae.fit(X=data, sess=sess,
                                saver=saver,
                                iteration=iteration,
                                learning_rate=learning_rate,
                                batch_size=batch_size,
                                )
            ae.global_nn_constructing(sess=sess, saver=saver, training_with_regu=True, regu_coef=regu_coef)
            ae.fit(X=data, saver=saver,
                        sess=sess,
                        iteration=iteration,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        )
            hidden = ae.transform(data, sess)
            recon_error = ae.get_sample_wise_recon_error(data, sess)
    return hidden, recon_error

def print_trainable_variables(sess):
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print("Variable: ", k)
        print("Shape: ", v.shape)
        # print(v)

if __name__ == "__main__":
    # dataset_name = "fashion_mnist"
    # ## mnist USPS fashion_mnist cifar10 cifar100
    # data,groundTruth = imagedataLoading.get_data(dataset_name, 100, [0])
    # TIP.image_saving(data, dataset_name, "test.png")
    data,groundTruth = imagedataLoading.get_data('USPS', 700, [0])
    print(data.shape)
