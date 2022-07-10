# NNAE + KNRN
## Directories
model
- codes of NNAE+KNRN

dataset 
- cardio: a cardiotocography data set from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php).
- waveform_noise: a waveform database generator data set from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php).
- [fashion_mnist](https://arxiv.org/abs/1708.07747): a 28x28 grayscale image dataset from Xiao H, Rasul K, Vollgraf R. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. CoRR, 2017, abs/1708.07747.
- [USPS](https://www.kaggle.com/datasets/bistaumanga/usps-dataset): a handwritten digits USPS dataset from Kaggle.
- [STL10](https://proceedings.mlr.press/v15/coates11a.html): an image dataset from Coates A, Ng A Y, Lee H. An analysis of single-layer networks in unsupervised feature learning. In AISTATS, 2011.


## Usage

1. ./dataset/$dataset_name/data.npy is used to fit the model and to generate labels.
2. ./dataset/$dataset_name/gt.npy is the groundtruth of the dataset.
3. These environments are required:
```
Python >= 3.6
TensorFlow >= 1.8
Numpy
Sklearn
```
3. Models can be used according to the following instructions. 

```
cd ./model
python ./ExprimentsFrame.py fashion_mnist 6 0.66
python ./ExprimentsFrame.py USPS 6 0.5
python ./ExprimentsFrame.py STL10 5 0.66
python ./ExprimentsFrame.py cardio 4 0.8
```
The meaning of these args can be found in the following file, which is the entrance of the code.
```
./model/ExprimentsFrame.py
```