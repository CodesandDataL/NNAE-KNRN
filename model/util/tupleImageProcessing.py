import numpy as np
import PIL.Image as Image
from util import ImShow as I


def X_to_tuple(X, shape):
    if len(shape) == 3:
        X_ = X.reshape(-1,shape[0],shape[1],shape[2])
        X_ = np.transpose(X_, (3,0,1,2))
        X_tuple = (X_[0].reshape(-1,shape[0]*shape[1]),X_[1].reshape(-1,shape[0]*shape[1]),X_[2].reshape(-1,shape[0]*shape[1]), None)
        return X_tuple
    else:
        return X

def image_saving(data, dataset_name, save_to_file):
    if dataset_name == 'mnist' or dataset_name == 'fashion_mnist':
        Image.fromarray(I.tile_raster_images(X=np.array(data).reshape((-1,784)),img_shape=(28, 28), tile_shape=(20, 20),tile_spacing=(1, 1))).save(save_to_file)
    if dataset_name == 'USPS':
        Image.fromarray(I.tile_raster_images(X=np.array(data).reshape((-1,256)),img_shape=(16, 16), tile_shape=(20, 20),tile_spacing=(1, 1))).save(save_to_file)
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        data = np.array(data, dtype=np.float64)
        images_tuple = X_to_tuple(data,[32,32,3])
        Image.fromarray(I.tile_raster_images(X=images_tuple,img_shape=(32, 32), tile_shape=(20, 20),tile_spacing=(1, 1))).save(save_to_file)
