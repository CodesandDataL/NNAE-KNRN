from time import time
import os
import anomaly_detection as ad
import ae
import numpy as np

def get_n_jobs():
	if os.path.isdir("H:/"):
		return 4
	else:
		return -1

## (5000, 784) to (5000, k+1, 784)
def find_nearest_neighbors(data, k):
	from sklearn.neighbors import NearestNeighbors
	n_jobs = get_n_jobs()
	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=n_jobs).fit(data)
	distances, indices = nbrs.kneighbors(data)
	data_nn = np.zeros((data.shape[0], k+1, data.shape[1]), dtype=np.float32)
	for i in range(data.shape[0]):
		one_data_neighbors = np.zeros((k+1, data.shape[1]))
		for j in range(k+1):
			indx = indices[i][j]
			one_data_neighbors[j] = data[indx]
		data_nn[i] = one_data_neighbors
	return data_nn

## (5000, 784) to (5000, 2k+1, 784)
def find_nearest_neighbors2(data, k, n):
	from sklearn.neighbors import NearestNeighbors
	n_jobs = get_n_jobs()
	nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree', n_jobs=n_jobs).fit(data)
	distances, indices = nbrs.kneighbors(data)
	data_nn = np.zeros((data.shape[0], 2*k+1, data.shape[1]), dtype=np.float32)
	for i in range(data.shape[0]):
		one_data_neighbors = np.zeros((2*k+1, data.shape[1]))
		for j in range(k+1):
			indx = indices[i][j]
			one_data_neighbors[j] = data[indx]
		for j in range(k):
			indx = indices[i][n-1-j]
			one_data_neighbors[k+1+j] = data[indx]
		data_nn[i] = one_data_neighbors
	rank = distances[:, -1].reshape((-1,)).argsort().argsort()
	return data_nn, rank

def find_nearest_neighbors_index(data, k):
	from sklearn.neighbors import NearestNeighbors
	import numpy as np
	n_jobs = get_n_jobs()

	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=n_jobs).fit(data)
	distances, indices = nbrs.kneighbors(data)
	return indices

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

def generate_nn(data, sampled_data, k):
	assert len(data.shape) == 2
	assert len(sampled_data.shape) == 2
	assert data.shape[-1] == sampled_data.shape[-1]
	assert k < sampled_data.shape[0]
	from sklearn.neighbors import NearestNeighbors
	data_nn = np.zeros((data.shape[0], k+1, data.shape[-1]))
	nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1).fit(sampled_data)
	_, indices = nbrs.kneighbors(data)
	# print(indices)
	for i in range(data.shape[0]):
		one_data_neighbors = np.zeros((k+1, data.shape[1]))
		for j in range(k+1):
			if j == 0:
				one_data_neighbors[j] = data[i]
				continue
			indx = indices[i][j-1]
			one_data_neighbors[j] = sampled_data[indx]
		data_nn[i] = one_data_neighbors
	return data_nn

# (5000, k+1, 784)
# def find_nearest_neighbors_with_ae_anomaly_score_uniform_sampling(data, k, sampling_size, ae_layersNum, ae_compressionRate, iteration):
# 	import numpy as np
# 	from sklearn.neighbors import NearestNeighbors
# 	n_jobs = get_n_jobs()
# 	_, ae_anomaly_score = ae.AutoEncoderDimReduction(data, ae_layersNum, ae_compressionRate, iteration, 0.0001, 1)
# 	sampled_data = sampling_uniformly_by_size(data, ae_anomaly_score, sampling_size)
# 	return generate_nn(data, sampled_data, k)

# def find_nearest_neighbors_with_ae_anomaly_score_3_partitions_sampling(data, k, ae_layersNum, ae_compressionRate, iteration, partition_size_list=[0.6,0.2,0.2], prob_list=[0.1,0.5,1]):
# 	import numpy as np
# 	from sklearn.neighbors import NearestNeighbors
# 	n_jobs = get_n_jobs()
# 	_, ae_anomaly_score = ae.AutoEncoderDimReduction(data, ae_layersNum, ae_compressionRate, iteration, 0.0001, 1)
# 	sampled_data = sampling_3_partitions_by_prob(data, ae_anomaly_score, partition_size_list, prob_list)
# 	return generate_nn(data, sampled_data, k)

# def find_nearest_neighbors_with_anomaly_score_criteria_sampling(data, k, prob_list):
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
# 	return generate_nn(data, sampled_data, k), partition_size_list

def find_nearest_neighbors_normal(data, k):
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

	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=n_jobs).fit(data)
	distances, indices = nbrs.kneighbors(data)
	data_nn = np.zeros((data.shape[0], k+1, data.shape[1]), dtype=np.float32)
	for i in range(data.shape[0]):
		one_data_neighbors = np.zeros((k+1, data.shape[1]))
		if ae_anomaly_score[i] < lower_criteria:
			for j in range(k+1):
				indx = indices[i][j]
				one_data_neighbors[j] = data[indx]
		else:
			for j in range(k+1):
					indx = indices[i][0]
					one_data_neighbors[j] = data[indx]
		data_nn[i] = one_data_neighbors
	return data_nn

def find_nearest_neighbors_vague(data, k):
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

	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=n_jobs).fit(data)
	distances, indices = nbrs.kneighbors(data)
	data_nn = np.zeros((data.shape[0], k+1, data.shape[1]), dtype=np.float32)
	for i in range(data.shape[0]):
		one_data_neighbors = np.zeros((k+1, data.shape[1]))
		if ae_anomaly_score[i] > lower_criteria and ae_anomaly_score[i] < higher_criteria:
			for j in range(k+1):
				indx = indices[i][j]
				one_data_neighbors[j] = data[indx]
		else:
			for j in range(k+1):
					indx = indices[i][0]
					one_data_neighbors[j] = data[indx]
		data_nn[i] = one_data_neighbors
	return data_nn

def find_nearest_neighbors_abnormal(data, k):
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

	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=n_jobs).fit(data)
	distances, indices = nbrs.kneighbors(data)
	data_nn = np.zeros((data.shape[0], k+1, data.shape[1]), dtype=np.float32)
	for i in range(data.shape[0]):
		one_data_neighbors = np.zeros((k+1, data.shape[1]))
		if ae_anomaly_score[i] > higher_criteria:
			for j in range(k+1):
				indx = indices[i][j]
				one_data_neighbors[j] = data[indx]
		else:
			for j in range(k+1):
					indx = indices[i][0]
					one_data_neighbors[j] = data[indx]
		data_nn[i] = one_data_neighbors
	return data_nn

# def find_nearest_neighbors_with_ae_anomaly_score_3_partitions_sampling_ft(data, k, ae_layersNum, ae_compressionRate, iteration, partition_size_list=[0.6,0.2,0.2], prob_list=[0.1,0.5,1]):
# 	import numpy as np
# 	from sklearn.neighbors import NearestNeighbors
# 	import nnae
# 	n_jobs = get_n_jobs()
# 	print(np.expand_dims(data, axis=1).shape)
# 	_, ae_anomaly_score, path = nnae.AEDimReduction(np.expand_dims(data, axis=1), ae_layersNum, ae_compressionRate, iteration, 0.0001, 1)
# 	sampled_data = sampling_3_partitions_by_prob(data, ae_anomaly_score, partition_size_list, prob_list)
# 	return generate_nn(data, sampled_data, k), path

# def find_nearest_neighbors_using_LSH(data, k, r, b):
# 	def bitdiff(a,b):
# 		diff = 0
# 		c = a ^ b
# 		while c:
# 			diff += (c & 1)
# 			c >>= 1
# 		# print("bitdiff: ", a, " ",b, " ", diff)
# 		# print("bitdiff: ", bin(a), " ",bin(b), " ", diff)
# 		return diff
# 	print("Getting bands of data of LSH.")
# 	t = time()
# 	from util import LocalitySensitiveHashing
# 	from sklearn.neighbors import NearestNeighbors
# 	import numpy as np
# 	if os.path.isdir("H:/"):
# 		n_jobs = 4
# 	else:
# 		n_jobs = -1
# 	lsh = LocalitySensitiveHashing.LocalitySensitiveHashing(dim=data.shape[-1], r=r, b=b)
# 	lsh.get_data_from_nparray(data)
# 	lsh.initialize_hash_store()
# 	lsh.hash_all_data_tf(data)
# 	lsh_bands, sample_hash_key = lsh.lsh_basic_for_nearest_neighbors()
# 	# print(sample_hash_key)
# 	# exit(0)

# 	# print(lsh.band_hash)
# 	# print(len(lsh.band_hash))
# 	# exit(0)


# 	length = []
# 	for item in lsh_bands:
# 		length.append(len(lsh_bands[item]))
# 	mean_length = np.mean(length)
# 	# assert mean_length != 0
# 	LSHRate = mean_length/data.shape[0]
# 	print("Average partial rate of band data: ", LSHRate)
# 	data_nn = np.zeros((data.shape[0], k+1, data.shape[1]))
# 	## bands does not contain the query entry itself, so n_neighbors=k, instead of k+1.
# 	nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=n_jobs)
# 	for i in range(data.shape[0]):
# 		one_data_neighbors = np.zeros((k+1, data.shape[1]))
# 		index_list_band_group = list(map(int, lsh_bands[str(i)]))
# 		# print(len(index_list_band_group))
# 		d = 1
# 		while len(index_list_band_group) < k:
# 			for key in lsh.band_hash:
# 				if key.split(" ")[1] == sample_hash_key[str(i)].split(" ")[1]:
# 					continue
# 				diff = bitdiff(int(key.split(" ")[1],2), int(sample_hash_key[str(i)].split(" ")[1]))
# 				if diff == d:
# 					# print("Not enough points in Bucket: "+ str(sample_hash_key[str(i)].split(" ")[1])+". Adding Bucket: ", key.split(" ")[1])
# 					# print("Adding: ", lsh.band_hash[key])
# 					index_list_band_group += list(map(int, lsh.band_hash[key]))
# 					if len(index_list_band_group) < k:
# 						continue
# 					else:
# 						break
# 			d += 1
# 		band_data = data[index_list_band_group]
# 		# print(band_data)
# 		# print(band_data.shape)
# 		nbrs.fit(band_data)
# 		distances, indices = nbrs.kneighbors(data[i].reshape((1,-1)))
# 		# print(indices)
# 		one_data_neighbors[0] = data[i]
# 		for j in range(1, k+1):
# 			indx = indices[0][j-1]
# 			one_data_neighbors[j] = data[indx]
# 		data_nn[i] = one_data_neighbors
# 	# print(data_nn)
# 	# print(data_nn.shape)
# 	print("LSH done, time:", time() - t)
# 	return data_nn, LSHRate

# def find_nearest_neighbors_using_FLSH(data, k, r, b):
# 	def bitdiff(a,b):
# 		diff = 0
# 		c = a ^ b
# 		while c:
# 			diff += (c & 1)
# 			c >>= 1
# 		return diff
# 	print("Getting bands of data of LSH.")
# 	t = time()
# 	import FastLocalitySensitiveHashing
# 	from sklearn.neighbors import NearestNeighbors
# 	import numpy as np
# 	if os.path.isdir("H:/"):
# 		n_jobs = 4
# 	else:
# 		n_jobs = -1
# 	lsh = FastLocalitySensitiveHashing.FastLocalitySensitiveHashing(data=data, bandsWidth=r, bandsNum=b)
# 	lsh_bands, sample_hash_key = lsh.similarity_neighborhoods, lsh.sample_name_to_hash_key
# 	length = []
# 	for item in lsh_bands:
# 		length.append(len(lsh_bands[item]))
# 	mean_length = np.mean(length)
# 	# assert mean_length != 0
# 	LSHRate = mean_length/data.shape[0]
# 	print("Average partial rate of band data: ", LSHRate)
# 	data_nn = np.zeros((data.shape[0], k+1, data.shape[1]))
# 	## bands does not contain the query entry itself, so n_neighbors=k, instead of k+1.
# 	nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=n_jobs)
# 	for i in range(data.shape[0]):
# 		one_data_neighbors = np.zeros((k+1, data.shape[1]))
# 		index_list_band_group = list(map(int, lsh_bands[i]))
# 		# print(len(index_list_band_group))
# 		d = 1
# 		while len(index_list_band_group) < k:
# 			for key in lsh.band_hash:
# 				if key.split(" ")[1] == sample_hash_key[i].split(" ")[1]:
# 					continue
# 				diff = bitdiff(int(key.split(" ")[1],2), int(sample_hash_key[i].split(" ")[1]))
# 				if diff == d:
# 					# print("Not enough points in Bucket: "+ str(sample_hash_key[i].split(" ")[1])+". Adding Bucket: ", key.split(" ")[1])
# 					# print("Adding: ", lsh.band_hash[key])
# 					index_list_band_group += list(map(int, lsh.band_hash[key]))
# 					if len(index_list_band_group) < k:
# 						continue
# 					else:
# 						break
# 			d += 1
# 		band_data = data[index_list_band_group]
# 		# print(band_data)
# 		# print(band_data.shape)
# 		nbrs.fit(band_data)
# 		distances, indices = nbrs.kneighbors(data[i].reshape((1,-1)))
# 		# print(indices)
# 		one_data_neighbors[0] = data[i]
# 		for j in range(1, k+1):
# 			indx = indices[0][j-1]
# 			one_data_neighbors[j] = data[indx]
# 		data_nn[i] = one_data_neighbors
# 	# print(data_nn)
# 	# print(data_nn.shape)
# 	print("LSH done, time:", time() - t)
# 	return data_nn, LSHRate

if __name__ == '__main__':
	from util import imagedataLoading
	data,groundTruth = imagedataLoading.get_data('mnist', 5000, [0])
	t = time()
	data_nn = find_nearest_neighbors(data, 5)
	print(time() - t)
