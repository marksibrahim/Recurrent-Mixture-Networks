# -*- coding: utf-8 -*-

from __future__ import print_function
from time import time
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import math, random
from numpy import linalg as LA
import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation



# Implemeting LDA on order level

PATH = "~"
userIDprodName_df = pd.read_csv(PATH + "DataRareProdMerg.csv", encoding = "ISO-8859-1") # in this dataset, rare products are replaced with their aisle names
newProdName=pd.factorize(userIDprodName_df['product_name'])
prodID = newProdName[0]
prodIDindex = newProdName[1]
userIDprodNameprodID_df = pd.concat([userIDprodName_df.reset_index(drop=True), pd.DataFrame(prodID)], axis=1)
userIDprodNameprodID_df.columns = ['n',
 'order_id',
 'product_id',
 'user_id',
 'product_name',
 'aisle',
 'product_id_tsfm',
 'nn']


ndocs = userIDprodNameprodID_df['order_id'].max()
nwords = userIDprodNameprodID_df['product_id_tsfm'].max()
print("number of documents (orders) and words (products) are:")
print( ndocs,nwords)


countOrder_series = userIDprodNameprodID_df.groupby(['user_id','order_id', 'product_id_tsfm']).size()
new_df = countOrder_series.to_frame(name = 'size').reset_index()


newdf_sparsemat=csr_matrix((new_df['size'], ( new_df['user_id'],new_df['order_id'])))

newdf_sparsemat_orderProd=csr_matrix((new_df['size'], ( new_df['order_id'], new_df['product_id_tsfm'])))


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += "-- ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    


lda = LatentDirichletAllocation(n_components=25, max_iter=100,
                                learning_method='online',
                                batch_size = 1000,
                                learning_offset=100. ,
                                random_state=0,
                                learning_decay = 0.6)
#   
t0 = time()
lda.fit(newdf_sparsemat_orderProd)
print("done in %0.3fs." % (time() - t0))


print_top_words(lda, prodIDindex, 7)

doc_word_dist = lda.transform(newdf_sparsemat_orderProd)



fig=plt.figure(figsize=(12,8))
columns = 7
rows = 2
ind_plot = random.sample(range(1, userIDprodNameprodID_df['user_id'].max()), 14)
cmt = 0
for i in ind_plot:
    cmt = cmt +1
    tmp = np.nonzero(newdf_sparsemat[i].todense())[1]
    tmpprod = np.transpose(doc_word_dist[tmp])
    img = tmpprod
    fig.add_subplot(rows, columns, cmt)
    plt.imshow(img,aspect='auto')
    plt.title('user ' + str(i))

plt.show()

