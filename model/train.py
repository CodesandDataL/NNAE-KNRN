import ae
import nnae3
import anomaly_detection as ad
from util import generatedDataLoading as loading
import os
import numpy as np
from util import log
from sklearn import metrics
import nn
import sys
import PIL.Image as Image
from util import ImShow as I

learning_rate = 0.002


def generate_sample_data_and_original_img(dataset_name):
	import random
	data, gt = loading.get_data(dataset_name)
	sampled_index_list = random.choices(list(range(data.shape[0])), k=10)
	Image.fromarray(I.tile_raster_images(X=data[sampled_index_list],img_shape=(int(data.shape[1]**0.5), int(data.shape[1]**0.5)), tile_shape=(2, 5),tile_spacing=(1, 1))).save(r"/static/pic/original.png")
	return sampled_index_list


data, gt = loading.get_data("USPS")
generate_sample_data_and_original_img(data)