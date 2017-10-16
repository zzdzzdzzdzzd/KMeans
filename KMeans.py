# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:53:47 2017

@author: zzd
"""

import numpy as np
from Recommand_Lib import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataSet=file2matrix('data/4k2_far.txt','\t')
dataMat=np.mat(dataSet[:,1:])

#执行KMeans算法
kmeans=KMeans(init='k-means++',n_clusters=4)
kmeans.fit(dataMat)

#绘制计算结果
drawScatter(plt,np.array(dataMat),size=20,color='b',mrkr='.')
drawScatter(plt,kmeans.cluster_centers_,size=60,color='red',mrkr='D')
plt.show()




