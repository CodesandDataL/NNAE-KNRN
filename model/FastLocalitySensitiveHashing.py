# -*- coding: utf-8 -*-

__doc__ = '''
@title
INTRODUCTION:

    The LocalitySensitiveHashing module is an implementation of the
    Locality Sensitive Hashing (LSH) algorithm for nearest neighbor search.
    The main idea in LSH is to avoid having to compare every pair of data
    samples in a large dataset in order to find the nearest similar
    neighbors for the different data samples.  With LSH, one can expect a
    data sample and its closest similar neighbors to be hashed into the
    same bucket with a high probability.  By treating the data samples
    placed in the same bucket as candidates for similarity checking, we
    significantly reduce the computational burden associated with finding
    nearest neighbors in large datasets.

                                dataSamples.T 
         
                       x1     x2     x3     x4     x5
                     ----------------------------------
                    |                                             
                h1  |  1      .      1      .      .        b=0  r=0
                h2  |  0      .      0      .      .        b=0  r=1
    hash        h3  |  1      .      1      .      .        b=0  r=2
    functions       |
                h4  |  .      1      .      1      .        b=1  r=0
                h5  |  .      1      .      1      .        b=1  r=1 
                h6  |  .      0      .      0      .        b=1  r=2
                    |
                h7  |  1      .      .      .      1        b=2  r=0
                h8  |  1      .      .      .      1        b=2  r=1
                h9  |  1      .      .      .      1        b=2  r=2
                    |

'''


import numpy as np
import tensorflow as tf
import random
import sys,os
from BitVector import *

class FastLocalitySensitiveHashing(object):
	def __init__(self, data, bandsWidth, bandsNum):
		assert len(data.shape) == 2
		self.dataDim = data.shape[1]
		self.dataNum = data.shape[0]
		self.dataNumpy = data
		self.bandsWidth = bandsWidth
		self.bandsNum = bandsNum
		self.howManyHashes = bandsWidth * bandsNum
		self.band_hash = {}
		with tf.Graph().as_default():
			with tf.Session() as sess:
				self.data = tf.convert_to_tensor(data, dtype=tf.float32)
				self.hashAllData(sess)
		self.basicForNearestNeighbors()
		# print(self.similarity_neighborhoods)
		# print(self.sample_name_to_hash_key)


	def hashAllData(self, sess):
		hplanes = tf.random.uniform(shape=[self.howManyHashes, self.dataDim], minval=-1., maxval=1.) 
		hp_norm = tf.reshape(tf.linalg.norm(hplanes, axis=1), [-1,1])
		hplanes = hplanes / hp_norm
		self.hplanes = hplanes
		bin_val = tf.matmul(self.data, tf.transpose(self.hplanes))
		self.hashTable = tf.math.greater(bin_val, 0.)
		self.hashTableNumpy = self.hashTable.eval(session=sess)

	def basicForNearestNeighbors(self):
		for k, sample in enumerate(self.dataNumpy, 0):
			for band_index in range(self.bandsNum):
				bits_in_band_k = self.hashTableNumpy[k][band_index*self.bandsWidth:(band_index+1)*self.bandsWidth]
				# bits_in_band_k is a numpy array slice
				bits_in_band_k = BitVector(bitlist=list(bits_in_band_k))
				key_index = str(band_index) + " " + str(bits_in_band_k)
				if key_index not in self.band_hash:
					self.band_hash[key_index] = set()
					self.band_hash[key_index].add(k)
				else:
					self.band_hash[key_index].add(k)
		similarity_neighborhoods = {sample_name : set() for sample_name in range(self.dataNum)}
		sample_name_to_hash_key = {}
		# print(self.band_hash)
		for key in self.band_hash:
			for sample_name in self.band_hash[key]:
				# print(key)
				# print(self.band_hash[key])
				# print(similarity_neighborhoods[sample_name])
				similarity_neighborhoods[sample_name].update(set(self.band_hash[key]) - set([sample_name]))
				sample_name_to_hash_key[sample_name] = key
		self.similarity_neighborhoods = similarity_neighborhoods
		self.sample_name_to_hash_key = sample_name_to_hash_key
		# return similarity_neighborhoods, sample_name_to_hash_key



if __name__ == "__main__":
	from util import generatedDataLoading as loading
	data, groungTruth = loading.get_data("USPS")
	print(data.shape)
	lsh = LocalitySensitiveHashing(data, 4, 2)
