import ae
import nnae
import nnae2
import nnae3
import anomaly_detection as ad
from util import generatedDataLoading as loading
import time
import os
import numpy as np
from util import log
from sklearn import metrics
import nn
import gnn
from collections import Counter
import sys

# ['mnist', 'fashion_mnist', 'USPS', 'cifar10', 'STL10', '20newsgroups', 'reuters800', 'reuters2000']
# ['cardio', 'ecoli', 'kddcup99', 'lymphography', 'waveform']
## dataset size cate layers nnae knn co reco
## MNIST 5000 0 784,400,280 2 25 200 100
## fashion-mnist 5000 0 784,400,280 2 25 200 100
## USPS 0 0 784,400,280 2 25 200 100
## STL10 3000 0 4096,2048,1024 2 100 200 100
## 20newsgroups 0 1 2000,1000,500 2 100 200 100
## cardio 0.8 7

MAX_N = 200
learning_rate = 0.002
iteration_list = [80,200,300]
repeat_times = 5

def CalAUC(gt, anomaly_score):
	fpr, tpr, thresholds = metrics.roc_curve(gt, anomaly_score)
	auc = metrics.auc(fpr, tpr)
	return round(auc, 4)

def CalAcc(gt, pred):
	return round(accuracy_score(gt, pred), 4)

def get_iteration(shape):
	if shape >= 20000:
		return iteration_list[0]
	elif shape >= 3000:
		return iteration_list[1]
	else:
		return iteration_list[2]

def KNN(data, groundTruth, anomalyNeighborsNum):
	if data.shape[0] < 20000:
		startTime = time.time()
		_, anomalyScore = ad.knn(data, contamination=0.15, n_neighbors=anomalyNeighborsNum)
		timeDuration = time.time() - startTime
		AUC = CalAUC(groundTruth, anomalyScore)
	else:	
		print("Data shapes are too large, skipping original KNN. ")
		timeDuration = 0
		AUC = 0
	return timeDuration, AUC

def AEKNN(data, groundTruth, layersNum, compressionRate, anomalyNeighborsNum):
	iteration = get_iteration(data.shape[0])
	regulationCoefficient = 100
	startTime = time.time()
	hidden, reconstructionError = ae.AutoEncoderDimReduction(data, layersNum, compressionRate, iteration, learning_rate, regulationCoefficient)
	_, anomalyScore = ad.knn(hidden, contamination=0.15, n_neighbors=anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC = CalAUC(groundTruth, anomalyScore)
	return timeDuration, AUC

def NNAEKNN(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum, regu):
	iteration = get_iteration(data.shape[0])
	startTime = time.time()
	data_nn = nn.find_nearest_neighbors(data=data, k=dimReductionNeighborsNum)
	hidden, reconstructionError = nnae.NNAEDimReduction(data_nn, layersNum, compressionRate, iteration, learning_rate, regu)
	_, anomalyScore = ad.knn(hidden, contamination=0.15, n_neighbors=anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC = CalAUC(groundTruth, anomalyScore)
	return timeDuration, AUC

# 本身recon_error * 第k距离，网络无新约束
def NNAEKNN_1(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum):
	iteration = get_iteration(data.shape[0])
	regulationCoefficient = 100
	startTime = time.time()
	data_nn = nn.find_nearest_neighbors(data=data, k=dimReductionNeighborsNum)
	hidden, reconstructionError = nnae.NNAEDimReduction(data_nn, layersNum, compressionRate, iteration, learning_rate, regulationCoefficient)
	anomalyScore = ad.krnn(hidden, reconstructionError, n_neighbors=anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC = CalAUC(groundTruth, anomalyScore)
	return timeDuration, AUC

# 本身recon_error * 第k距离，网络无新约束
def NNAEKNN_2(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum):
	iteration = get_iteration(data.shape[0])
	regulationCoefficient = 100
	startTime = time.time()
	data_nn = nn.find_nearest_neighbors(data=data, k=dimReductionNeighborsNum)
	hidden, reconstructionError = nnae.NNAEDimReduction(data_nn, layersNum, compressionRate, iteration, learning_rate, regulationCoefficient)
	anomalyScore = ad.krann(hidden, reconstructionError, n_neighbors=anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC = CalAUC(groundTruth, anomalyScore)
	return timeDuration, AUC

# 比例*第k距离
def NNAEKNN_3(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum, a):
	iteration = get_iteration(data.shape[0])
	regulationCoefficient = 100
	startTime = time.time()
	data_nn = nn.find_nearest_neighbors(data=data, k=dimReductionNeighborsNum)
	hidden, reconstructionError = nnae.NNAEDimReduction(data_nn, layersNum, compressionRate, iteration, learning_rate, regulationCoefficient)
	anomalyScore = ad.knn_recon_error_rate(hidden, a, reconstructionError, n_neighbors=anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC = CalAUC(groundTruth, anomalyScore)
	return timeDuration, AUC

# recon_error和距离的向量积
def NNAEKNN_4(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum):
	iteration = get_iteration(data.shape[0])
	regulationCoefficient = 100
	startTime = time.time()
	data_nn = nn.find_nearest_neighbors(data=data, k=dimReductionNeighborsNum)
	hidden, reconstructionError = nnae.NNAEDimReduction(data_nn, layersNum, compressionRate, iteration, learning_rate, regulationCoefficient)
	anomalyScore = ad.knn_recon_error_distance_vector_product(hidden, reconstructionError, anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC = CalAUC(groundTruth, anomalyScore)
	return timeDuration, AUC

# 第k近邻的recon_error和第k近邻距离
def NNAEKNN_5(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum):
	iteration = get_iteration(data.shape[0])
	regulationCoefficient = 100
	startTime = time.time()
	data_nn = nn.find_nearest_neighbors(data=data, k=dimReductionNeighborsNum)
	hidden, reconstructionError = nnae.NNAEDimReduction(data_nn, layersNum, compressionRate, iteration, learning_rate, regulationCoefficient)
	anomalyScore = ad.knn_k_recon_error_k_distance_product(hidden, reconstructionError, anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC = CalAUC(groundTruth, anomalyScore)
	return timeDuration, AUC

def NNAEKNN_6(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum, regu1, regu2):
	iteration = get_iteration(data.shape[0])
	regulationCoefficient = 100
	startTime = time.time()
	data_nn = nn.find_nearest_neighbors2(data=data, k=dimReductionNeighborsNum, n=anomalyNeighborsNum)
	hidden, reconstructionError, recon_rate, regu1_rate, regu2_rate, total_rate = nnae3.NNAEDimReduction(data_nn, layersNum, compressionRate, iteration, learning_rate, regu1, regu2)
	_, anomalyScore = ad.knn(hidden, contamination=0.15, n_neighbors=anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC = CalAUC(groundTruth, anomalyScore)
	return timeDuration, AUC, recon_rate, regu1_rate, regu2_rate, total_rate

# all kinds of normalization
def NNAEKNN_7(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum, regu1, regu2):
	iteration = get_iteration(data.shape[0])
	startTime = time.time()
	data_nn = nn.find_nearest_neighbors2(data=data, k=dimReductionNeighborsNum, n=anomalyNeighborsNum)
	hidden, reconstructionError, recon_rate, regu1_rate, regu2_rate, total_rate = nnae3.NNAEDimReduction(data_nn, layersNum, compressionRate, iteration, learning_rate, regu1, regu2)
	anomalyScore1 = ad.krnn_normalization1(hidden, reconstructionError, anomalyNeighborsNum, dimReductionNeighborsNum)
	anomalyScore2 = ad.krnn_normalization2(hidden, reconstructionError, anomalyNeighborsNum)
	anomalyScore3 = ad.krnn_normalization3(hidden, reconstructionError, anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC1 = CalAUC(groundTruth, anomalyScore1)
	AUC2 = CalAUC(groundTruth, anomalyScore2)
	AUC3 = CalAUC(groundTruth, anomalyScore3)
	print("AUC: ", AUC1, AUC2, AUC3)
	return timeDuration, AUC1, AUC2, AUC3, recon_rate

def IForest(data, groundTruth):
	iteration = get_iteration(data.shape[0])
	startTime = time.time()
	_, anomalyScore = ad.iforest(data)
	timeDuration = time.time() - startTime
	AUC = CalAUC(groundTruth, anomalyScore)
	print("if: ", AUC)
	return timeDuration, AUC



def NNAEIForest(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, regu1, regu2):
	iteration = get_iteration(data.shape[0])
	startTime = time.time()
	data_nn = nn.find_nearest_neighbors2(data=data, k=dimReductionNeighborsNum, n=200)
	hidden, reconstructionError, recon_rate, regu1_rate, regu2_rate, total_rate = nnae3.NNAEDimReduction(data_nn, layersNum, compressionRate, iteration, learning_rate, regu1, regu2)
	_, anomalyScore1 = ad.iforest(hidden)
	anomalyScore2 = ad.iforest_normalization2(hidden, reconstructionError)
	timeDuration = time.time() - startTime
	AUC1 = CalAUC(groundTruth, anomalyScore1)
	AUC2 = CalAUC(groundTruth, anomalyScore2)
	print("nnaeif: ", AUC1, AUC2)
	return timeDuration, AUC1, AUC2, recon_rate



def GNNAEKNN(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum, self_weight):
	iteration = get_iteration(data.shape[0])
	regulationCoefficient = 100
	startTime = time.time()
	data_nn = gnn.generate_gnn_mix_mean(data, data, dimReductionNeighborsNum, self_weight)
	hidden, reconstructionError = nnae.NNAEDimReduction(data_nn, layersNum, compressionRate, iteration, learning_rate, regulationCoefficient)
	_, anomalyScore = ad.knn(hidden, contamination=0.15, n_neighbors=anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC = CalAUC(groundTruth, anomalyScore)
	return timeDuration, AUC

def ASNNAEKNN(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum, select):
	assert select in ['normal', 'vague', 'abnormal']
	iteration = get_iteration(data.shape[0])
	regulationCoefficient = 100
	startTime = time.time()
	if select == 'normal':
		data_nn = nn.find_nearest_neighbors_normal(data, k=dimReductionNeighborsNum)
	elif select == 'vague':
		data_nn = nn.find_nearest_neighbors_vague(data, k=dimReductionNeighborsNum)
	elif select == 'abnormal':
		data_nn = nn.find_nearest_neighbors_abnormal(data, k=dimReductionNeighborsNum)
	hidden, reconstructionError = nnae.NNAEDimReduction(data_nn, layersNum, compressionRate, iteration, learning_rate, regulationCoefficient)
	_, anomalyScore = ad.knn(hidden, contamination=0.15, n_neighbors=anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC = CalAUC(groundTruth, anomalyScore)
	return timeDuration, AUC

# find_nearest_neighbors_with_ae_anomaly_score_3_partitions_sampling
def ASGNNAEKNN(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum, select, self_weight):
	assert select in ['normal', 'vague', 'abnormal']
	iteration = get_iteration(data.shape[0])
	regulationCoefficient = 100
	startTime = time.time()
	if select == 'normal':
		data_nn, ratio = gnn.find_mix_mean_neighbors_with_normal(data, dimReductionNeighborsNum, self_weight)
	elif select == 'vague':
		data_nn, ratio = gnn.find_mix_mean_neighbors_with_vague(data, dimReductionNeighborsNum, self_weight)
	elif select == 'abnormal':
		data_nn, ratio = gnn.find_mix_mean_neighbors_with_abnormal(data, dimReductionNeighborsNum, self_weight)
	else:
		return 0,0,[0,0,0]
	hidden, reconstructionError = nnae.NNAEDimReduction(data_nn, layersNum, compressionRate, iteration, learning_rate, regulationCoefficient)
	_, anomalyScore = ad.knn(hidden, contamination=0.15, n_neighbors=anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC = CalAUC(groundTruth, anomalyScore)
	ratio = np.array(ratio)
	ratio = np.round(ratio, 2)
	ratio = list(ratio)
	return timeDuration, AUC, ratio

def SortingAndRecording(datasetResults, datasetName):
	print(datasetResults)
	original = sys.stdout
	f = open('logs.'+str(datasetName)+'.txt', 'a+')
	sys.stdout = f
	print("Dataset name: ", datasetName)
	print("Learning rate: ", learning_rate)
	firstLine = ['layers', 'cr', 'nn', 'if', 'nnaeif1', 'nnaeif2', 'RDR']
	print('({:>7s},{:>6s},{:>5s}){:>10s} {:>10s} {:>10s} {:>10s}'.format(*firstLine))
	sortedResults = sorted(datasetResults, reverse=True, key = lambda x: x['nnaeifauc2'])
	for i in range(len(datasetResults)):
		line = []
		for key in ['layersNum', 'compressionRate', 'dimReductionNeighborsNum', 'ifauc', 'nnaeifauc1', 'nnaeifauc2', 'recon_rate']:
			if key == 'prob_list' or key == 'ratio':
				line.append(str(sortedResults[i][key]))
			else:
				line.append(sortedResults[i][key])
		print('({:>7.0f},{:>6.2f},{:>5.0f}) {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}'.format(*line))
	print('\n\n')
	sys.stdout = original
	f.close()

imageDatasetsList = ['mnist', 'fashion_mnist', 'USPS', 'STL10']
verbalDatasetsList = ['cardio', 'ecoli', 'kddcup99_sampled', '20newsgroups', 'reuters2000']
other20newsgroupsList = list('other20newsgroups'+str(i) for i in range(20))

AvailableDatasets = ['20newsgroups', 'fashion_mnist', 'lymphography', 'reuters2000', 'USPS', 'cardio', 'kddcup99', 'mnist', 'shuttle', 'waveform', 'ecoli', 'kddcup99_sampled', 'STL10', 'waveform_noise', 'other20newsgroups']
AvailableDatasets += other20newsgroupsList
AvailableDatasets += ['fonts', 'fonts-ad']
AvailableDatasets += ['kddcup99_2', 'kddcup99_2_sampled']



datasetsList = ['cardio', 'ecoli', 'fashion_mnist', 'USPS', 'kddcup99_sampled', 'STL10', 'kddcup99']

# global anomalyNeighborsNum, layersNum, compressionRate, dimReductionNeighborsNum, bandsWidth, bandsNum, prob_list, learning_rate
global anomalyNeighborsNum, layersNum, compressionRate, dimReductionNeighborsNum, regu1, regu2


def getNetStructureParams(dataset_name):
	if dataset_name == 'fashion_mnist':
		return 6,0.66
	elif dataset_name == 'USPS':
		return 6,0.5
	elif dataset_name == 'STL10':
		return 5,0.66
	elif dataset_name == 'cardio':
		return 4,0.8
	elif dataset_name == 'ecoli':
		return 3,0.5
	elif dataset_name == 'kddcup99_sampled':
		return 6,0.66
	elif dataset_name == 'kddcup99':
		return 3,0.8
	elif dataset_name == 'fonts':
		return 6,0.5
	else:
		return 6,0.66

# def ParametersIterator_Settings(dataset_name):
# 	for anomalyNeighborsNum in [25,50,100,200]:
# 		for dimReductionNeighborsNum in [2,3]:
# 			for regu1 in [200]:
# 				for regu2 in [10]:
# 					layersNum, compressionRate = getNetStructureParams(dataset_name)
# 					yield {
# 									'layersNum': layersNum,
# 									'compressionRate': compressionRate, 
# 									'dimReductionNeighborsNum': dimReductionNeighborsNum,
# 									'anomalyNeighborsNum': anomalyNeighborsNum,
# 									'regu1': regu1,
# 									'regu2': regu2,
# 									}

def ParametersIterator_Settings(dataset_name):
	# for layersNum in [3,5,7]:
	# 	for compressionRate in [0.5,0.66,0.8]:
	for layersNum in [3]:
		for compressionRate in [0.5]:
			for dimReductionNeighborsNum in [1]:
				for regu1 in [200]:
					for regu2 in [10]:
						# layersNum, compressionRate = getNetStructureParams(dataset_name)
						yield {
										'layersNum': layersNum,
										'compressionRate': compressionRate, 
										'dimReductionNeighborsNum': dimReductionNeighborsNum,
										'regu1': regu1,
										'regu2': regu2,
										}

if __name__ == '__main__':
	if len(sys.argv) != 2:
		raise "Error. Accept exactly one arg."
	elif sys.argv[1] == "image":
		datasetsList = imageDatasetsList
	elif sys.argv[1] == "verbal":
		datasetsList = verbalDatasetsList
	elif sys.argv[1] == "6datasets":
		datasetsList = datasetsList
		print(datasetsList)
	elif sys.argv[1] in AvailableDatasets:
		datasetsList = [sys.argv[1]]
	else:
		raise "Error. Unknown."

	for dataset in datasetsList:
		itr = ParametersIterator_Settings(dataset)
		datasetResults = []
		data, groundTruth = loading.get_data(dataset)
		print(dataset, data.shape)
		while True:
			try:
				globals().update(next(itr))
				result = {}
				ifknnt, ifauc = IForest(data, groundTruth)
				nnaeift, nnaeifauc1, nnaeifauc2, recon_rate = [], [], [], []
				for i in range(repeat_times):

					out = NNAEIForest(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, regu1, regu2)
					nnaeift.append(out[0])
					nnaeifauc1.append(out[1])
					nnaeifauc2.append(out[2])
					recon_rate.append(out[3])

				nnaeift = sum(nnaeift)/len(nnaeift)
				nnaeifauc1 = sum(nnaeifauc1)/len(nnaeifauc1)
				nnaeifauc2 = sum(nnaeifauc2)/len(nnaeifauc2)
				recon_rate = sum(recon_rate)/len(recon_rate)
				result['layersNum'] = layersNum
				result['compressionRate'] = round(compressionRate, 2)
				result['dimReductionNeighborsNum'] = dimReductionNeighborsNum
				result['regu1'] = round(regu1)
				result['regu2'] = round(regu2)
				result['ift'], result['ifauc'] = round(ifknnt, 4), round(ifauc, 4)
				result['nnaeift'] = round(nnaeift, 4)
				result['nnaeifauc1'] = round(nnaeifauc1, 4)
				result['nnaeifauc2'] = round(nnaeifauc2, 4)
				result['recon_rate'] = round(recon_rate, 4)
				processFile = open("./process.txt", 'a+')
				original = sys.stdout
				sys.stdout = processFile
				print(dataset, result)
				sys.stdout = original
				datasetResults.append(result.copy())

			except StopIteration:
				break
		SortingAndRecording(datasetResults, dataset)

	# for dataset in datasetsList:
	# 	itr = ParametersIterator_Settings(dataset)
	# 	datasetResults = []
	# 	data, groundTruth = loading.get_data(dataset)
	# 	print(dataset, data.shape)
	# 	knn_time, knn_auc = KNN(data, groundTruth, anomalyNeighborsNum)
	# 	while True:
	# 		try:
	# 			globals().update(next(itr))
	# 			result = {}
	# 			aeknnt, aeknnauc, nnaeknnt, nnaeknnauc, gnnaeknnt, gnnaeknnauc, asnnaeknnt, asnnaeknnauc, asgnnaeknnt, asgnnaeknnauc = [], [], [], [], [], [], [], [], [], []
	# 			for i in range(repeat_times):

	# 				out = AEKNN(data, groundTruth, layersNum, compressionRate, anomalyNeighborsNum)
	# 				aeknnt.append(out[0])
	# 				aeknnauc.append(out[1])

	# 				out = NNAEKNN(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum)
	# 				nnaeknnt.append(out[0])
	# 				nnaeknnauc.append(out[1])

	# 				out = GNNAEKNN(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum, self_weight)
	# 				gnnaeknnt.append(out[0])
	# 				gnnaeknnauc.append(out[1])

	# 				out = ASNNAEKNN(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum, select)
	# 				asnnaeknnt.append(out[0])
	# 				asnnaeknnauc.append(out[1])

	# 				out = ASGNNAEKNN(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum, select, self_weight)
	# 				asgnnaeknnt.append(out[0])
	# 				asgnnaeknnauc.append(out[1])

	# 			aeknnt = sum(aeknnt)/len(aeknnt)
	# 			aeknnauc = sum(aeknnauc)/len(aeknnauc)
	# 			nnaeknnt = sum(nnaeknnt)/len(nnaeknnt)
	# 			nnaeknnauc = sum(nnaeknnauc)/len(nnaeknnauc)
	# 			gnnaeknnt = sum(gnnaeknnt)/len(gnnaeknnt)
	# 			gnnaeknnauc = sum(gnnaeknnauc)/len(gnnaeknnauc)
	# 			asnnaeknnt = sum(asnnaeknnt)/len(asnnaeknnt)
	# 			asnnaeknnauc = sum(asnnaeknnauc)/len(asnnaeknnauc)
	# 			asgnnaeknnt = sum(asgnnaeknnt)/len(asgnnaeknnt)
	# 			asgnnaeknnauc = sum(asgnnaeknnauc)/len(asgnnaeknnauc)
	# 			result['anomalyNeighborsNum'] = anomalyNeighborsNum
	# 			result['layersNum'] = layersNum
	# 			result['compressionRate'] = compressionRate
	# 			result['dimReductionNeighborsNum'] = dimReductionNeighborsNum
	# 			result['select'] = select
	# 			result['self_weight'] = self_weight
	# 			result['KNNTimeDuration'], result['KNNAUC'] = round(knn_time, 4), round(knn_auc, 4)
	# 			result['AEKNNTimeDuration'] = round(aeknnt, 4)
	# 			result['AEKNNAUC'] = round(aeknnauc, 4)
	# 			result['NNAEKNNTimeDuration'] = round(nnaeknnt, 4)
	# 			result['NNAEKNNAUC'] = round(nnaeknnauc, 4)
	# 			result['GNNAEKNNTimeDuration'] = round(gnnaeknnt, 4)
	# 			result['GNNAEKNNAUC'] = round(gnnaeknnauc, 4)
	# 			result['ASNNAEKNNTimeDuration'] = round(asnnaeknnt, 4)
	# 			result['ASNNAEKNNAUC'] = round(asnnaeknnauc, 4)
	# 			result['ASGNNAEKNNTimeDuration'] = round(asgnnaeknnt, 4)
	# 			result['ASGNNAEKNNAUC'] = round(asgnnaeknnauc, 4)
	# 			# result['ratio'] = np.round(ratio.mean(axis=0),2)
	# 			processFile = open("./process.txt", 'a+')
	# 			original = sys.stdout
	# 			sys.stdout = processFile
	# 			print(dataset, result)
	# 			sys.stdout = original
	# 			datasetResults.append(result.copy())

	# 		except StopIteration:
	# 			break
	# 	SortingAndRecording(datasetResults, dataset)

