import numpy as np
import os

dataset_list = ['cardio', 'ecoli', 'kddcup99', 'lymphography', 'waveform', 'waveform_noise', 'kddcup99_sampled']
dataset_list += ['mnist', 'fashion_mnist', 'USPS', 'STL10', '20newsgroups', 'reuters2000', 'combonewsgroups']
dataset_list += ['other20newsgroups0', 'other20newsgroups1', 'other20newsgroups2', 'other20newsgroups3', 'other20newsgroups4', 'other20newsgroups5', 'other20newsgroups6', 'other20newsgroups7', 'other20newsgroups8', 'other20newsgroups9', 'other20newsgroups10', 'other20newsgroups11', 'other20newsgroups12', 'other20newsgroups13', 'other20newsgroups14', 'other20newsgroups15', 'other20newsgroups16', 'other20newsgroups17', 'other20newsgroups18', 'other20newsgroups19']
dataset_list += ['p53', 'fonts', 'fonts-ad']
dataset_list += ['kddcup99_2', 'kddcup99_2_sampled']

# data_path = "H:/data/" if os.path.isdir("H:/") else "/Users/lappe-rutgers/data/"
# data_path = "H:/data/" if os.path.isdir("H:/") else "/home/LAB/liusz/data/"
if os.path.isdir("H:/"):
	data_path = "H:/data/"
elif os.path.isdir("/Users/lappe-rutgers/"):
	data_path = "/Users/lappe-rutgers/data/"
elif os.path.isdir("/home/LAB"):
	data_path = "/home/LAB/liusz/data/"
else:
	print("Unknown system environment, set up dataloading path in './util/generatedDataLoading.py'")

def get_data(dataset_name):
	print("dataset_name: "+dataset_name)
	assert dataset_name in dataset_list
	return np.load(data_path+str(dataset_name)+'/data.npy'), np.load(data_path+str(dataset_name)+'/gt.npy')

if __name__ == '__main__':
	for name in dataset_list:
		data,gt = get_data(name)
		print(data.shape)
		print(gt.shape)
