import numpy as np
gt = np.load("gt.npy")
print(np.load("data.npy")[gt==0].shape)
