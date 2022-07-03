from time import time
import os
import anomaly_detection as ad
import ae
import numpy as np
from sklearn.neighbors import NearestNeighbors

def get_n_jobs():
	if os.path.isdir("H:/"):
		return 4
	else:
		return -1

# def generate_nn(data, sampled_data, k):
# 	assert len(data.shape) == 2
# 	assert len(sampled_data.shape) == 2
# 	assert data.shape[-1] == sampled_data.shape[-1]
# 	assert k < sampled_data.shape[0]

# 	data_nn = np.zeros((data.shape[0], k+1, data.shape[-1]))
# 	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=-1).fit(sampled_data)
# 	_, indices = nbrs.kneighbors(data)
# 	# print(indices)
# 	for i in range(data.shape[0]):
# 		one_data_neighbors = np.zeros((k+1, data.shape[1]))
# 		for j in range(k+1):
# 			if j == 0:
# 				one_data_neighbors[j] = data[i]
# 				continue
# 			indx = indices[i][j]
# 			one_data_neighbors[j] = sampled_data[indx]
# 		data_nn[i] = one_data_neighbors
# 	return data_nn

def mix_mean(data, indices, i, k, self_weight, neighbors_weight):
	# tmp_ = np.zeros((k+1, data.shape[1]))
	self_ = data[i]
	neighbors_ = data[indices[i][1:]]
	neighbors_mean = neighbors_.mean(axis=0)
	return self_*self_weight + neighbors_mean*neighbors_weight

# def expanse_mean(data, indices, i, k):
# 	tmp_ = np.zeros((data.shape[1]*2, ))
# 	tmp_[:data.shape[1]] = data[i]
# 	tmp_[data.shape[1]:] = data[indices[i][1:]].mean(axis=0)
# 	return tmp_

# def convolution_with_neighbors_mean(data, indices, i, k):
# 	self_ = data[i]
# 	nei_ = data[indices[i][1:]]
# 	nei_mean = nei_.mean(axis=0)
# 	conv = np.convolve(self_, nei_mean)
# 	return conv

# def expanse_coordianates_wise_max(data, indices, i, k):
# 	tmp_ = np.zeros((data.shape[1]*2, ))
# 	tmp_[:data.shape[1]] = data[i]
# 	tmp_[data.shape[1]:] = data[indices[i][1:]].max(axis=0)
# 	return tmp_

# def mix_coordianates_wise_max(data, indices, i, k):
# 	tmp_ = np.zeros((k+1, data.shape[1]))
# 	tmp_[0] = data[i]
# 	tmp_[1:] = data[indices[i][1:]]
# 	return tmp_.max(axis=0)

# def generate_gnn_expanse_mean(data, sampled_data, k):
# 	self_weight = 0.5
# 	neighbors_weight = 0.5/k
# 	assert len(data.shape) == 2
# 	assert len(sampled_data.shape) == 2
# 	assert data.shape[-1] == sampled_data.shape[-1]
# 	assert k < sampled_data.shape[0]
# 	data_nn = np.zeros((data.shape[0], k+1, data.shape[-1]*2))
# 	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=-1).fit(sampled_data)
# 	_, indices = nbrs.kneighbors(data)
# 	for i in range(data.shape[0]):
# 		one_data_neighbors = np.zeros((k+1, data.shape[1]*2))
# 		for ii in range(k+1):
# 			if ii == 0:
# 				one_data_neighbors[ii] = data[i]
# 			else:
# 				one_data_neighbors[ii] = expanse_mean(data, indices, indices[i][ii], k)
# 		data_nn[i] = one_data_neighbors
# 	return data_nn


# def generate_gnn_expanse_coordianates_wise_max(data, sampled_data, k):
# 	self_weight = 0.5
# 	neighbors_weight = 0.5/k
# 	assert len(data.shape) == 2
# 	assert len(sampled_data.shape) == 2
# 	assert data.shape[-1] == sampled_data.shape[-1]
# 	assert k < sampled_data.shape[0]
# 	data_nn = np.zeros((data.shape[0], k+1, data.shape[-1]))
# 	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=-1).fit(sampled_data)
# 	_, indices = nbrs.kneighbors(data)
# 	for i in range(data.shape[0]):
# 		one_data_neighbors = np.zeros((k+1, data.shape[1]))
# 		for ii in range(k+1):
# 			if ii == 0:
# 				one_data_neighbors[ii] = data[i]
# 			else:
# 				one_data_neighbors[ii] = expanse_coordianates_wise_max(data, indices, indices[i][ii], k)
# 		data_nn[i] = one_data_neighbors
# 	return data_nn

# def generate_gnn_mix_coordianates_wise_max(data, sampled_data, k):
# 	self_weight = 0.5
# 	neighbors_weight = 0.5/k
# 	assert len(data.shape) == 2
# 	assert len(sampled_data.shape) == 2
# 	assert data.shape[-1] == sampled_data.shape[-1]
# 	assert k < sampled_data.shape[0]
# 	data_nn = np.zeros((data.shape[0], k+1, data.shape[-1]))
# 	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=-1).fit(sampled_data)
# 	_, indices = nbrs.kneighbors(data)
# 	for i in range(data.shape[0]):
# 		one_data_neighbors = np.zeros((k+1, data.shape[1]))
# 		for ii in range(k+1):
# 			if ii == 0:
# 				one_data_neighbors[ii] = data[i]
# 			else:
# 				one_data_neighbors[ii] = mix_coordianates_wise_max(data, indices, indices[i][ii], k)
# 		data_nn[i] = one_data_neighbors
# 	return data_nn

def generate_gnn_mix_mean(data, sampled_data, k, self_weight):
	neighbors_weight = (1 - self_weight)/k
	assert len(data.shape) == 2
	assert len(sampled_data.shape) == 2
	assert data.shape[-1] == sampled_data.shape[-1]
	assert k < sampled_data.shape[0]
	data_nn = np.zeros((data.shape[0], k+1, data.shape[-1]))
	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=-1).fit(sampled_data)
	_, indices = nbrs.kneighbors(data)
	for i in range(data.shape[0]):
		one_data_neighbors = np.zeros((k+1, data.shape[1]))
		for ii in range(k+1):
			if ii == 0:
				one_data_neighbors[ii] = data[i]
			else:
				one_data_neighbors[ii] = mix_mean(data, indices, indices[i][ii], k, self_weight, neighbors_weight)
		data_nn[i] = one_data_neighbors
	return data_nn

def generate_gnn_mix_mean_normal(data, ae_anomaly_score, criteria, k, self_weight):
	neighbors_weight = (1 - self_weight)/k
	assert len(data.shape) == 2
	assert data.shape[0] == ae_anomaly_score.shape[0]
	data_nn = np.zeros((data.shape[0], k+1, data.shape[-1]))
	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=-1).fit(data)
	_, indices = nbrs.kneighbors(data)
	for i in range(data.shape[0]):
		one_data_neighbors = np.zeros((k+1, data.shape[1]))
		if ae_anomaly_score[i] < criteria:
			for ii in range(k+1):
				if ii == 0:
					one_data_neighbors[ii] = data[i]
				else:
					one_data_neighbors[ii] = mix_mean(data, indices, indices[i][ii], k, self_weight, neighbors_weight)
			data_nn[i] = one_data_neighbors
		else:
			for ii in range(k+1):
				one_data_neighbors[ii] = data[i]
				data_nn[i] = one_data_neighbors
	return data_nn

def generate_gnn_mix_mean_vague(data, ae_anomaly_score, lower, higher, k, self_weight):
	neighbors_weight = (1 - self_weight)/k
	assert len(data.shape) == 2
	assert data.shape[0] == ae_anomaly_score.shape[0]
	data_nn = np.zeros((data.shape[0], k+1, data.shape[-1]))
	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=-1).fit(data)
	_, indices = nbrs.kneighbors(data)
	for i in range(data.shape[0]):
		one_data_neighbors = np.zeros((k+1, data.shape[1]))
		if ae_anomaly_score[i] > lower and ae_anomaly_score[i] < higher:
			for ii in range(k+1):
				if ii == 0:
					one_data_neighbors[ii] = data[i]
				else:
					one_data_neighbors[ii] = mix_mean(data, indices, indices[i][ii], k, self_weight, neighbors_weight)
			data_nn[i] = one_data_neighbors
		else:
			for ii in range(k+1):
				one_data_neighbors[ii] = data[i]
				data_nn[i] = one_data_neighbors
	return data_nn

def generate_gnn_mix_mean_abnormal(data, ae_anomaly_score, criteria, k, self_weight):
	neighbors_weight = (1 - self_weight)/k
	assert len(data.shape) == 2
	assert data.shape[0] == ae_anomaly_score.shape[0]
	data_nn = np.zeros((data.shape[0], k+1, data.shape[-1]))
	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=-1).fit(data)
	_, indices = nbrs.kneighbors(data)
	for i in range(data.shape[0]):
		one_data_neighbors = np.zeros((k+1, data.shape[1]))
		if ae_anomaly_score[i] > criteria:
			for ii in range(k+1):
				if ii == 0:
					one_data_neighbors[ii] = data[i]
				else:
					one_data_neighbors[ii] = mix_mean(data, indices, indices[i][ii], k, self_weight, neighbors_weight)
			data_nn[i] = one_data_neighbors
		else:
			for ii in range(k+1):
				one_data_neighbors[ii] = data[i]
				data_nn[i] = one_data_neighbors
	return data_nn

# def generate_gnn_convolution_with_neighbors_mean(data, sampled_data, k):
# 	self_weight = 0.5
# 	neighbors_weight = 0.5/k
# 	assert len(data.shape) == 2
# 	assert len(sampled_data.shape) == 2
# 	assert data.shape[-1] == sampled_data.shape[-1]
# 	assert k < sampled_data.shape[0]
# 	data_nn = np.zeros((data.shape[0], k+1, 2*data.shape[-1]-1))
# 	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=-1).fit(sampled_data)
# 	_, indices = nbrs.kneighbors(data)

# 	for i in range(data.shape[0]):
# 		one_data_neighbors = np.zeros((k+1, data.shape[1]*2-1))
# 		for ii in range(k+1):
# 			if ii == 0:
# 				one_data_neighbors[ii] = data[i]
# 			else:
# 				one_data_neighbors[ii] = convolution_with_neighbors_mean(data, indices, indices[i][ii], k)
# 		data_nn[i] = one_data_neighbors
# 	return data_nn

def find_mix_mean_neighbors_with_normal(data, k, self_weight):
	import numpy as np
	from sklearn.neighbors import NearestNeighbors
	n_jobs = get_n_jobs()
	_, ae_anomaly_score = ae.AutoEncoderDimReduction(data, 4, 0.66, 100, 0.0001, 1)
	lower_criteria = ae_anomaly_score.mean()
	higher_criteria = 1.5 * ae_anomaly_score.mean()
	partition_size_list = []
	partition_size_list.append(ae_anomaly_score[ae_anomaly_score<lower_criteria].shape[0]/data.shape[0])
	tmp = ae_anomaly_score[ae_anomaly_score>lower_criteria]
	partition_size_list.append(tmp[tmp<higher_criteria].shape[0]/data.shape[0])
	partition_size_list.append(ae_anomaly_score[ae_anomaly_score>higher_criteria].shape[0]/data.shape[0])
	return generate_gnn_mix_mean_normal(data, ae_anomaly_score, lower_criteria, k, self_weight), partition_size_list

def find_mix_mean_neighbors_with_vague(data, k, self_weight):
	import numpy as np
	from sklearn.neighbors import NearestNeighbors
	n_jobs = get_n_jobs()
	_, ae_anomaly_score = ae.AutoEncoderDimReduction(data, 4, 0.66, 100, 0.0001, 1)
	lower_criteria = ae_anomaly_score.mean()
	higher_criteria = 1.5 * ae_anomaly_score.mean()
	partition_size_list = []
	partition_size_list.append(ae_anomaly_score[ae_anomaly_score<lower_criteria].shape[0]/data.shape[0])
	tmp = ae_anomaly_score[ae_anomaly_score>lower_criteria]
	partition_size_list.append(tmp[tmp<higher_criteria].shape[0]/data.shape[0])
	partition_size_list.append(ae_anomaly_score[ae_anomaly_score>higher_criteria].shape[0]/data.shape[0])
	return generate_gnn_mix_mean_vague(data, ae_anomaly_score, lower_criteria, higher_criteria, k, self_weight), partition_size_list

def find_mix_mean_neighbors_with_abnormal(data, k, self_weight):
	import numpy as np
	from sklearn.neighbors import NearestNeighbors
	n_jobs = get_n_jobs()
	_, ae_anomaly_score = ae.AutoEncoderDimReduction(data, 4, 0.66, 100, 0.0001, 1)
	lower_criteria = ae_anomaly_score.mean()
	higher_criteria = 1.5 * ae_anomaly_score.mean()
	partition_size_list = []
	partition_size_list.append(ae_anomaly_score[ae_anomaly_score<lower_criteria].shape[0]/data.shape[0])
	tmp = ae_anomaly_score[ae_anomaly_score>lower_criteria]
	partition_size_list.append(tmp[tmp<higher_criteria].shape[0]/data.shape[0])
	partition_size_list.append(ae_anomaly_score[ae_anomaly_score>higher_criteria].shape[0]/data.shape[0])
	return generate_gnn_mix_mean_abnormal(data, ae_anomaly_score, higher_criteria, k, self_weight), partition_size_list

# def asgnn_expanse_mean_neighbors(data, k, prob_list):
# 	import numpy as np
# 	from sklearn.neighbors import NearestNeighbors
# 	n_jobs = get_n_jobs()
# 	_, ae_anomaly_score = ae.AutoEncoderDimReduction(data, 4, 0.66, 100, 0.0001, 1)
# 	lower_criteria = ae_anomaly_score.mean()
# 	higher_criteria = 1.5 * ae_anomaly_score.mean()
# 	partition_size_list = []
# 	partition_size_list.append(ae_anomaly_score[ae_anomaly_score<lower_criteria].shape[0]/data.shape[0])
# 	tmp = ae_anomaly_score[ae_anomaly_score>lower_criteria]
# 	partition_size_list.append(tmp[tmp<higher_criteria].shape[0]/data.shape[0])
# 	partition_size_list.append(ae_anomaly_score[ae_anomaly_score>higher_criteria].shape[0]/data.shape[0])
# 	sampled_data = sampling_3_partitions_by_prob(data, ae_anomaly_score, partition_size_list, prob_list)
# 	return generate_gnn_expanse_mean(data, sampled_data, k), partition_size_list

# def sampling_uniformly_by_size(data, anomaly_score, size_of_out):
# 	size_of_in = data.shape[0]
# 	sampled_data = np.zeros(data[:size_of_out].shape)
# 	sorted_data = data[np.argsort(anomaly_score)]
# 	for i in range(size_of_out):
# 		sampled_data[i] = sorted_data[int(i*size_of_in/size_of_out)]
# 	return sampled_data

# def sampling_3_partitions_by_prob(data, anomaly_score, partition_size_list, prob_list):
# 	size_of_in = data.shape[0]
# 	# sampled_data = np.zeros(data[:size_of_out].shape)
# 	sampled_data = []
# 	sorted_data = data[np.argsort(anomaly_score)]
# 	# for i in range(size_of_out):
# 	# 	sampled_data[i] = sorted_data[int(i*size_of_in/size_of_out)]
# 	for i in range(len(partition_size_list)):
# 		start_index = int(data.shape[0]*sum(partition_size_list[:i]))
# 		end_index = int(data.shape[0]*sum(partition_size_list[:i+1]))
# 		for one_data in data[start_index:end_index]:
# 			if np.random.uniform() <= prob_list[i]:
# 				sampled_data.append(one_data)
# 	return np.array(sampled_data)


# # (5000, k+1, 784)
# def find_nearest_neighbors_with_ae_anomaly_score_uniform_sampling(data, k, sampling_size, ae_layersNum, ae_compressionRate, iteration):
# 	n_jobs = get_n_jobs()
# 	_, ae_anomaly_score = ae.AutoEncoderDimReduction(data, ae_layersNum, ae_compressionRate, iteration, 0.0001, 1)
# 	sampled_data = sampling_uniformly_by_size(data, ae_anomaly_score, sampling_size)
# 	return generate_nn(data, sampled_data, k)

# def find_nearest_neighbors_with_ae_anomaly_score_3_partitions_sampling(data, k, ae_layersNum, ae_compressionRate, iteration, partition_size_list=[0.6,0.2,0.2], prob_list=[0.1,0.5,1]):
# 	n_jobs = get_n_jobs()
# 	_, ae_anomaly_score = ae.AutoEncoderDimReduction(data, ae_layersNum, ae_compressionRate, iteration, 0.0001, 1)
# 	sampled_data = sampling_3_partitions_by_prob(data, ae_anomaly_score, partition_size_list, prob_list)
# 	return generate_nn(data, sampled_data, k)

# def find_nearest_neighbors_with_anomaly_score_criteria_sampling(data, k, prob_list):
# 	n_jobs = get_n_jobs()
# 	_, ae_anomaly_score = ae.AutoEncoderDimReduction(data, 4, 0.66, 100, 0.0001, 1)
# 	lower_criteria = ae_anomaly_score.mean()
# 	higher_criteria = 1.5 * ae_anomaly_score.mean()

# 	partition_size_list = []
# 	partition_size_list.append(ae_anomaly_score[ae_anomaly_score<lower_criteria].shape[0]/data.shape[0])
# 	tmp = ae_anomaly_score[ae_anomaly_score>lower_criteria]
# 	partition_size_list.append(tmp[tmp<higher_criteria].shape[0]/data.shape[0])
# 	partition_size_list.append(ae_anomaly_score[ae_anomaly_score>higher_criteria].shape[0]/data.shape[0])
# 	sampled_data = sampling_3_partitions_by_prob(data, ae_anomaly_score, partition_size_list, prob_list)
# 	return generate_nn(data, sampled_data, k)


if __name__ == '__main__':
	from util import imagedataLoading
	data,groundTruth = imagedataLoading.get_data('mnist', 5000, [0])
	t = time()
	data_nn = find_nearest_neighbors(data, 5)
	print(time() - t)
