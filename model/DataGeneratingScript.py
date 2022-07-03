import nnae
import anomaly_detection as ad
from util import generatedDataLoading as loading
from time import strftime, localtime
import time
import os
import numpy as np
from util import log
from util import imagedataLoading
from util import multiVariableDataLoading
from util import LocalitySensitiveHashing
from sklearn import metrics
import nn
from collections import Counter
import sys

dataset_list1 = ['mnist', 'fashion_mnist', 'USPS', 'STL10', '20newsgroups', 'reuters2000']

for datasetName in dataset_list1:
	data, groundTruth = imagedataLoading.get_data(datasetName, 0)
	try:
		os.mkdir("/home/LAB/liusz/data/"+datasetName+"/")
	except OSError as Error:
		print(Error)
	np.save("/home/LAB/liusz/data/"+datasetName+"/data", data)
	np.save("/home/LAB/liusz/data/"+datasetName+"/gt.npy", groundTruth)
	print(datasetName+" done.")
