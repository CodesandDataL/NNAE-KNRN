import ae
import nnae
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
import sys

MAX_N = 200
learning_rate = 0.002
iteration_list = [80,200,300]
repeat_times = 3
## #SBATCH -t 2-00:00

def ParametersIterator_Settings(dataset_name, anomalyNeighborsNumList, dimReductionNeighborsNumList):
	for anomalyNeighborsNum in anomalyNeighborsNumList:
		for layersNum in [3,5,7,9]:
			for compressionRate in [0.2,0.4,0.6,0.8]:
				for dimReductionNeighborsNum in dimReductionNeighborsNumList:
					for regu1 in [200]:
						for regu2 in [10]:
							# layersNum, compressionRate = getNetStructureParams(dataset_name)
							yield {
											'layersNum': layersNum,
											'compressionRate': compressionRate, 
											'dimReductionNeighborsNum': int(dimReductionNeighborsNum),
											'anomalyNeighborsNum': int(anomalyNeighborsNum),
											'regu1': regu1,
											'regu2': regu2,
											}

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
	print("KNN_AUC: ", AUC)
	return timeDuration, AUC


def AEKNN(data, groundTruth, layersNum, compressionRate, anomalyNeighborsNum):
	iteration = get_iteration(data.shape[0])
	regulationCoefficient = 100
	startTime = time.time()
	hidden, reconstructionError = ae.AutoEncoderDimReduction(data, layersNum, compressionRate, iteration, learning_rate, regulationCoefficient)
	_, anomalyScore2 = ad.knn(hidden, contamination=0.15, n_neighbors=anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC1 = CalAUC(groundTruth, reconstructionError)
	AUC2 = CalAUC(groundTruth, anomalyScore2)
	return timeDuration, AUC1, AUC2

def NNAEKNN(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum, regu):
	iteration = get_iteration(data.shape[0])
	startTime = time.time()
	data_nn = nn.find_nearest_neighbors(data=data, k=dimReductionNeighborsNum)
	hidden, reconstructionError = nnae.NNAEDimReduction(data_nn, layersNum, compressionRate, iteration, learning_rate, regu)
	_, anomalyScore = ad.knn(hidden, contamination=0.15, n_neighbors=anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC = CalAUC(groundTruth, anomalyScore)
	return timeDuration, AUC

# all kinds of normalization
def NNAEKNN_7(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum, regu1, regu2):
	iteration = get_iteration(data.shape[0])
	startTime = time.time()
	data_nn, data_rank = nn.find_nearest_neighbors2(data=data, k=dimReductionNeighborsNum, n=anomalyNeighborsNum)
	hidden, reconstructionError, recon_0, regu1_0, regu2_0, recon_1, regu1_1, regu2_1 = nnae3.NNAEDimReduction(data_nn, layersNum, compressionRate, iteration, learning_rate, regu1, regu2)
	# anomalyScore1 = ad.krnn_normalization1(hidden, reconstructionError, anomalyNeighborsNum, dimReductionNeighborsNum)
	_, anomalyScore1 = ad.knn(hidden, contamination=0.15, n_neighbors=anomalyNeighborsNum)
	anomalyScore2, hidden_rank = ad.krnn_normalization2(hidden, reconstructionError, anomalyNeighborsNum)
	# anomalyScore3 = ad.krnn_normalization3(hidden, reconstructionError, anomalyNeighborsNum)
	timeDuration = time.time() - startTime
	AUC1 = CalAUC(groundTruth, anomalyScore1)
	AUC2 = CalAUC(groundTruth, anomalyScore2)
	print("AUC: ", AUC1, AUC2)
	# rank_diff = np.sum(np.abs(data_rank - hidden_rank))
	lower_half_sorted_reconstruction_error = np.sort(reconstructionError)[:int(data.shape[0]*0.5)]
	top_percents_sorted_reconstruction_error = np.sort(reconstructionError)[int(data.shape[0]*0.95):]
	return timeDuration, AUC1, AUC2, np.mean(lower_half_sorted_reconstruction_error) / np.mean(top_percents_sorted_reconstruction_error)

def SortingAndRecording(datasetResults, datasetName):
	print(datasetResults)
	original = sys.stdout
	f = open('logs.'+str(datasetName)+'.txt', 'a+')
	sys.stdout = f
	print("Dataset name: ", datasetName)
	print("Learning rate: ", learning_rate)
	firstLine = ['layers', 'cr', 'nn', 'NN', 'AE', 'KNN', 'AEKNN', 'NNAEKNN', 'NNAEKRNN', 'indicator']
	print('({:>7s},{:>6s},{:>5s},{:>5s}) {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(*firstLine))
	sortedResults = sorted(datasetResults, reverse=True, key = lambda x: x['nnaekrnnauc'])
	# vs = []
	# rd = []
	# for i in range(len(datasetResults)):
	# 	vs.append(float(datasetResults[i]['recon_v']))
	# 	rd.append(float(datasetResults[i]['rank_d']))
	for i in range(len(datasetResults)):
		line = []
		for key in ['layersNum', 'compressionRate', 'dimReductionNeighborsNum', 'anomalyNeighborsNum', 'knnauc', 'aeauc', 'aeknnauc', 'nnaeknnauc', 'nnaekrnnauc', 'indicator']:
			line.append(sortedResults[i][key])
		print('({:>7.0f},{:>6.2f},{:>5.0f},{:>5.0f}) {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.4g}'.format(*line))
	indicator_ = []
	nnaekrnnauc_ = []
	for i in range(len(datasetResults)):
		indicator_.append(sortedResults[i]['indicator'])
		nnaekrnnauc_.append(sortedResults[i]['nnaekrnnauc'])
	print(list(np.array(nnaekrnnauc_)[np.argsort(np.array(indicator_))]))
	print('\n\n')
	sys.stdout = original
	f.close()

imageDatasetsList = ['mnist', 'fashion_mnist', 'USPS', 'STL10']
verbalDatasetsList = ['cardio', 'ecoli', 'kddcup99_sampled', '20newsgroups', 'reuters2000']
AvailableDatasets = ['20newsgroups', 'fashion_mnist', 'lymphography', 'reuters2000', 'USPS', 'cardio', 'kddcup99', 'mnist', 'shuttle', 'waveform', 'ecoli', 'kddcup99_sampled', 'STL10', 'waveform_noise', 'other20newsgroups']
AvailableDatasets += ['fonts', 'fonts-ad']
AvailableDatasets += ['kddcup99_2', 'kddcup99_2_sampled']

datasetsList = ['cardio', 'ecoli', 'fashion_mnist', 'USPS', 'kddcup99_sampled', 'STL10', 'kddcup99']
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

if __name__ == '__main__':
	if len(sys.argv) != 4:
		raise "Argv number error."
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
	anomalyNeighborsNumList = sys.argv[2].split(",")
	dimReductionNeighborsNumList = sys.argv[3].split(",")

	for dataset in datasetsList:
		itr = ParametersIterator_Settings(dataset, anomalyNeighborsNumList, dimReductionNeighborsNumList)
		datasetResults = []
		data, groundTruth = loading.get_data(dataset)
		print(dataset, data.shape)
		while True:
			try:
				globals().update(next(itr))
				result = {}
				knnt, knnauc = KNN(data, groundTruth, anomalyNeighborsNum)
				aet, aeauc, aeknnauc, nnaeknnt, nnaeknnauc, nnaekrnnauc, indicator = [], [], [], [], [], [], []
				for i in range(repeat_times):
					out = AEKNN(data, groundTruth, layersNum, compressionRate, anomalyNeighborsNum)
					aet.append(out[0])
					aeauc.append(out[1])
					aeknnauc.append(out[2])

					out = NNAEKNN_7(data, groundTruth, layersNum, compressionRate, dimReductionNeighborsNum, anomalyNeighborsNum, regu1, regu2)
					nnaeknnt.append(out[0])
					nnaeknnauc.append(out[1])
					nnaekrnnauc.append(out[2])
					indicator.append(out[3])

				aet = sum(aet)/len(aet)
				aeauc = sum(aeauc)/len(aeauc)
				aeknnauc = sum(aeknnauc)/len(aeknnauc)
				nnaeknnt = sum(nnaeknnt)/len(nnaeknnt)
				nnaeknnauc = sum(nnaeknnauc)/len(nnaeknnauc)
				nnaekrnnauc = sum(nnaekrnnauc)/len(nnaekrnnauc)
				indicator = sum(indicator)/len(indicator)

				result['anomalyNeighborsNum'] = anomalyNeighborsNum
				result['layersNum'] = layersNum
				result['compressionRate'] = round(compressionRate, 2)
				result['dimReductionNeighborsNum'] = dimReductionNeighborsNum
				result['regu1'] = round(regu1)
				result['regu2'] = round(regu2)
				result['knnt'], result['knnauc'] = round(knnt, 4), round(knnauc, 4)

				result['aet'] = round(aet, 4)
				result['aeauc'] = round(aeauc, 4)
				result['aeknnauc'] = round(aeknnauc, 4)
				result['nnaeknnt'] = round(nnaeknnt, 4)
				result['nnaeknnauc'] = round(nnaeknnauc, 4)
				result['nnaekrnnauc'] = round(nnaekrnnauc, 4)
				result['indicator'] = round(indicator, 8)

				processFile = open("./process.txt", 'a+')
				original = sys.stdout
				sys.stdout = processFile
				print(dataset, result)
				sys.stdout = original
				datasetResults.append(result.copy())
			except StopIteration:
				break
		SortingAndRecording(datasetResults, dataset)

# def call_shape(dataset):
# 	dict_ = {
# 		"USPS": (16, 16),
# 		"mnist": (28, 28),
# 		"fashion_mnist": (28, 28),
# 	}
# 	return dict_[dataset]

# if __name__ == "__main__":
# 	import PIL.Image as Image
# 	from util import ImShow as I
# 	dataset = "fashion_mnist"
# 	data, groundTruth = loading.get_data(dataset)
# 	Image.fromarray(I.tile_raster_images(X=data[:25],img_shape=(call_shape(dataset)), tile_shape=(5, 5),tile_spacing=(1, 1))).save(r"original.png")
