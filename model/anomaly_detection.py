import numpy as np
import os
import time
import sys
from collections import Counter
from time import localtime,strftime


def knn(features, contamination=0.15, n_neighbors=25):
	print('knn..')
	if os.path.isdir("H:/"):
		n_jobs = 4
	else:
		n_jobs = -1
	from pyod.models.knn import KNN
	clf = KNN(n_neighbors=n_neighbors, contamination=contamination, n_jobs=n_jobs)
	clf.fit(features)
	return clf.labels_, clf.decision_scores_

# 本身recon_err*第k距离
def krnn(features, recon_error, n_neighbors):
	from sklearn.neighbors import NearestNeighbors
	nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1).fit(features)
	distance, _ = nbrs.kneighbors(features)
	k_distance = distance[:,-1]
	print(k_distance.shape)
	return k_distance*recon_error

# 平均recon_error*第k距离
def krann(features, recon_error, n_neighbors):
	from sklearn.neighbors import NearestNeighbors
	nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1).fit(features)
	distance, indices = nbrs.kneighbors(features)
	all_recon_error = np.zeros((recon_error.shape[0], n_neighbors))
	for i in range(recon_error.shape[0]):
		all_recon_error[i] = recon_error[indices[i]]
	mean = np.mean(all_recon_error, axis=1)
	print(mean.shape, distance[:,-1].shape)
	return mean*distance[:,-1]

def knn_recon_error_rate(features, a, recon_error, n_neighbors):
	threshold = np.mean(recon_error) * a
	def cal_rate(re, re_array):
		return re_array[re_array>re].shape[0] / re_array.shape[0]
	from sklearn.neighbors import NearestNeighbors
	nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1).fit(features)
	distance, indices = nbrs.kneighbors(features)
	anomaly_score = np.zeros((recon_error.shape[0],))
	for i in range(recon_error.shape[0]):
		anomaly_score[i] = recon_error[i] * cal_rate(recon_error[i], recon_error[indices[i,:]])
	print(anomaly_score)
	print(anomaly_score.shape)
	return anomaly_score

def knn_recon_error_distance_vector_product(features, recon_error, n_neighbors):
	from sklearn.neighbors import NearestNeighbors
	nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1).fit(features)
	distance, indices = nbrs.kneighbors(features)
	anomaly_score = np.zeros((recon_error.shape[0],))
	for i in range(recon_error.shape[0]):
		anomaly_score[i] = np.dot(distance[i], recon_error[indices[i]])
	return anomaly_score

def knn_k_recon_error_k_distance_product(features, recon_error, n_neighbors):
	from sklearn.neighbors import NearestNeighbors
	nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1).fit(features)
	distance, indices = nbrs.kneighbors(features)
	anomaly_score = np.zeros((recon_error.shape[0],))
	for i in range(recon_error.shape[0]):
		anomaly_score[i] = distance[i][-1] * recon_error[indices[i][-1]]
	return anomaly_score

# e^((r-max)/(max-min))
def krnn_normalization1(features, recon_error, n_neighbors, k):
	from sklearn.neighbors import NearestNeighbors
	nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1).fit(features)
	distance, indices = nbrs.kneighbors(features)
	k_distance = distance[:,-1]
	print(k_distance.shape)
	re_max = np.max(recon_error)
	re_min = np.min(recon_error)
	new_recon_error = np.zeros((recon_error.shape))
	for i in range(new_recon_error.shape[0]):
		new_recon_error[i] = np.mean(recon_error[indices[i][:k+1]])
	recon_error = np.exp((recon_error - re_min)/(re_max - re_min))
	return k_distance*recon_error

# e^((r-min)/(max-min))
def krnn_normalization2(features, recon_error, n_neighbors):
	from sklearn.neighbors import NearestNeighbors
	nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1).fit(features)
	distance, _ = nbrs.kneighbors(features)
	k_distance = distance[:,-1] 
	print(k_distance.shape)
	re_max = np.max(recon_error)
	re_min = np.min(recon_error)
	recon_error = np.exp((recon_error - re_min)/(re_max - re_min))
	features_rank = distance[:,-1].reshape((-1,)).argsort().argsort()
	return k_distance*recon_error, features_rank

def krnn_normalization3(features, recon_error, n_neighbors):
	from sklearn.neighbors import NearestNeighbors
	nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1).fit(features)
	distance, indices = nbrs.kneighbors(features)
	k_distance = distance[:,-1] 
	print(k_distance.shape)
	re_max = np.max(recon_error)
	re_min = np.min(recon_error)
	recon_error = np.exp((recon_error - re_min)/(re_max - re_min))
	anomaly_score = np.zeros((recon_error.shape[0],))
	for i in range(recon_error.shape[0]):
		anomaly_score[i] = np.dot(distance[i], recon_error[indices[i]])
	return anomaly_score

def cal_anomaly_score_LSH(i, lsh_index, data, n_neighbors):
	lsh_index = list(lsh_index)
	band_data = data[lsh_index]
	print(band_data.shape)
	print('calling eu distance')
	eu_dis = euclidean_distances(band_data, [data[i]])
	print(eu_dis.shape)
	anomaly_score = eu_dis[min(n_neighbors, len(lsh_index)-1)]
	return anomaly_score

# def lshknn(data, n_neighbors=25, r=10, b=20):
# 	from util import LocalitySensitiveHashing
# 	from sklearn.metrics.pairwise import euclidean_distances
# 	t = time.time()
# 	lsh = LocalitySensitiveHashing.LocalitySensitiveHashing(dim = data.shape[-1], r = r, b = b)
# 	lsh.get_data_from_nparray(data)
# 	print("Reading data time: ", time.time() - t)
# 	t = time.time()
# 	# lsh.show_data_for_lsh()
# 	lsh.initialize_hash_store()
# 	print("Initailizing hashing time: ", time.time() - t)
# 	t = time.time()
# 	# print('hashing')
# 	# lsh.hash_all_data_tf(data)
# 	lsh.hash_all_data()
# 	print("Hashing time: ", time.time() - t)
# 	t = time.time()
# 	# print('getting neighbors')
# 	lsh_neighbors = lsh.lsh_basic_for_nearest_neighbors()
# 	print("Finding hashing neigbors time: ", time.time() - t)
# 	t = time.time()
# 	print("lsh_neighbors: ", len(lsh_neighbors))
# 	anomaly_score = []
# 	# print('calling auc')
# 	for i, sample_name in enumerate(lsh_neighbors):
# 		lsh_index = []
# 		for item in lsh_neighbors[sample_name]:
# 			lsh_index.append(int(item))
# 		band_data = data[lsh_index]
# 		reasonable_k = min(n_neighbors, len(lsh_index)-1)
# 		# print('1')
# 		eu_dis = euclidean_distances(band_data, data[i].reshape(1,-1))
# 		# print(eu_dis.shape)
# 		ind = eu_dis.argsort()[reasonable_k]
# 		one_score = eu_dis[ind]
# 		anomaly_score.append(one_score)
# 		# print("One score time: ", time.time() - t)
# 		# t = time.time()
# 		# _ = input()
# 	anomaly_score = np.array(anomaly_score).reshape(-1,1)
# 	return anomaly_score

def lshknn(data, n_neighbors=25, r=10, b=20):
	from util import LocalitySensitiveHashing
	from sklearn.metrics.pairwise import euclidean_distances
	lsh = LocalitySensitiveHashing.LocalitySensitiveHashing(dim = data.shape[-1], r = r, b = b)
	lsh.get_data_from_nparray(data)
	lsh.initialize_hash_store()
	lsh.hash_all_data_tf(data)
	lsh_neighbors = lsh.lsh_basic_for_nearest_neighbors()
	anomaly_score = []
	for i, sample_name in enumerate(lsh_neighbors):
		lsh_index = []
		for item in lsh_neighbors[sample_name]:
			lsh_index.append(int(item))
		band_data = data[lsh_index]
		reasonable_k = min(n_neighbors, len(lsh_index)-1)
		eu_dis = euclidean_distances(band_data, data[i].reshape(1,-1))
		ind = eu_dis.argsort()[reasonable_k]
		one_score = eu_dis[ind]
		anomaly_score.append(one_score)
	anomaly_score = np.array(anomaly_score).reshape(-1,1)
	return anomaly_score

def abod(features, contamination=0.15, n_neighbors=10):
	from pyod.models.abod import ABOD
	clf = ABOD(n_neighbors=n_neighbors, contamination=contamination)
	clf.fit(features)
	return clf.labels_, clf.decision_scores_

def iforest(features, contamination=0.15):
	from pyod.models.iforest import IForest
	clf = IForest(contamination=contamination)
	clf.fit(features)
	return clf.labels_, clf.decision_scores_

def iforest_normalization2(features, recon_error):
	from pyod.models.iforest import IForest
	clf = IForest()
	clf.fit(features)
	re_max = np.max(recon_error)
	re_min = np.min(recon_error)
	recon_error = np.exp((recon_error - re_min)/(re_max - re_min))
	return clf.decision_scores_*recon_error

def lof(features, contamination=0.15, n_neighbors=25, metric="euclidean"):
	## Using euclidean distance for default
	## Bigger scores correspond to inliers, so returning its opposite value.
	from pyod.models.lof import LOF
	clf = LOF(contamination=contamination, n_neighbors=n_neighbors, metric=metric)
	clf.fit(features)
	return clf.labels_, clf.decision_scores_

def loci(features, contamination=0.15, alpha=0.5, k=3):
 #    alpha : int, default = 0.5
 #    The neighbourhood parameter measures how large of a neighbourhood
 #    should be considered "local".

	# k: int, default = 3
 #    An outlier cutoff threshold for determine whether or not a point 
 #    should be considered an outlier.
	from pyod.models.loci import LOCI
	clf = LOCI(contamination=contamination, alpha=alpha, k=k)
	clf.fit(features)
	return clf.labels_

def pca(data, dimension):
	from sklearn.decomposition import PCA
	pca = PCA(n_components=dimension)
	pca.fit(data)
	## PCA(copy=True, iterated_power='auto', n_components=2, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
	return pca.transform(data)

def aedetector(features=None, contamination=0.15):
	recon = np.load("./recon_error.npy")
	recon_tmp = recon.copy()
	recon_tmp.sort()
	threshold = recon_tmp[int((1-contamination)*recon.shape[0])]
	return (recon >= threshold).astype('int').ravel(), recon


if __name__ == '__main__':
	from util import imagedataLoading
	data, groundTruth = imagedataLoading.get_data("mnist", 5000, [0])
	pred_label, anomaly_score = lof(data)
	print(anomaly_score[anomaly_score<0].shape)
	print(anomaly_score[anomaly_score>0][anomaly_score<1].shape)
	print(anomaly_score[anomaly_score>1].shape)
	print(anomaly_score[pred_label == 1])

