import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

def read_arrays_by_file_name(file_name):
	return np.load("./arrays/"+file_name), np.load("./arrays/nn"+file_name), np.load("./arrays/gt_"+file_name.split("ae_")[1])

def sampling(ae,nnae,gt,rate):
	sampled_ae = np.zeros((ae.shape[0]*rate, ae.shape[1]))
	sampled_nnae = np.zeros((ae.shape[0]*rate, ae.shape[1]))
	sampled_gt = np.zeros((ae.shape[0]*rate, ae.shape[1]))
	shape_focused_points = gt[gt>0].shape[0]
	sampled_ae[:shape_focused_points] = ae[gt>0]
	sampled_nnae[:shape_focused_points] = nnae[gt>0]
	sampled_gt[:shape_focused_points] = gt[gt>0]
	left_lines = sampled_ae[shape_focused_points:].shape[0]
	sampled_ae[shape_focused_points:] = ae[gt==0][:left_lines]
	sampled_nnae[shape_focused_points:] = nnae[gt==0][:left_lines]
	sampled_gt[shape_focused_points:] = gt[gt==0][:left_lines]
	return sampled_ae, sampled_nnae, sampled_gt

file_name = "ae_20newsgroups_Fri-04-Sep-2020-193417_hidden.npy"
rate = 0.2
ae_hidden, nnae_hidden, gt = read_arrays_by_file_name(file_name)
ae_hidden, nnae_hidden, gt = sampling(ae_hidden, nnae_hidden, gt,rate)
gt[gt>5] = 0
print(Counter(gt))
df = pd.DataFrame()
df['AutoEnocder_hidden_x'] = ae_hidden[:,0][gt>0]
df['AutoEnocder_hidden_y'] = ae_hidden[:,1][gt>0]
df['NNAE_hidden_x'] = nnae_hidden[:,0][gt>0]
df['NNAE_hidden_y'] = nnae_hidden[:,1][gt>0]
df['labels'] = gt[gt>0]
# df['labels'] = df['labels'].astype(np.int64)
# markers = {0: "X", 1: "s", 2: "s", 3: "s", 4: "s", 5: "s"}
# colors = {0: "#000000", 1: "#555555", 2: "#aaaaaa", 3: "#eeeeee", 4: "#eeeeee", 5: "#eeeeee"}
# colors = ["#000000", "#555555", "#aaaaaa", "#eeeeee"]
# sns.set_palette(sns.color_palette(colors)) 
# ax = sns.scatterplot(x="AutoEnocder_hidden_x", y="AutoEnocder_hidden_y", hue="labels", markers=markers, c=colors, data=df)
# ax = sns.scatterplot(x="NNAE_hidden_x", y="AutoEnocder_hidden_y", hue="labels", markers=markers, data=df)

# cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
# ax = sns.scatterplot(x="AutoEnocder_hidden_x", y="AutoEnocder_hidden_y",
#                      hue="labels", size="labels",
#                      palette="Set2",
#                      data=df)

# df = pandas.DataFrame(ae_hidden, columns=["x", "y"])
def colors(x):
	dic_ = {
		0: "yellow",
		1: "blue",
		2: "green",
		3: "red",
		4: "cyan",
		5: "black",
	}
	return dic_[x]

# import seaborn as sns
# sns.relplot(x="AutoEnocder_hidden_x", y="AutoEnocder_hidden_y", hue="labels", size="labels",
#             sizes=(20, 80),
#             height=6, data=df)

sns.set(style='dark',)

# sns.scatterplot(x = "AutoEnocder_hidden_x", y = "AutoEnocder_hidden_y", data=df, hue='labels', style='labels', markers=['*', 'o', 'o', 'o', 'o', 'o'], palette='Spectral')
sns.scatterplot(x = "AutoEnocder_hidden_x", y = "AutoEnocder_hidden_y", data=df, hue='labels', style='labels', markers=['o','o','o','o','o'], palette='Spectral')
# sns.scatterplot(x = "NNAE_hidden_x", y = "NNAE_hidden_y", data=df, hue='labels', style='labels', markers=['o','o','o','o','o'], palette='Spectral')


# sns.set(style="white")

plt.show()
