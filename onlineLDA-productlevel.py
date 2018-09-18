from __future__ import print_function
from time import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import gamma
from scipy.stats import multivariate_normal
import pandas as pd

import math, random
from numpy import linalg as LA
import matplotlib.pyplot as plt
import copy

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


userIDprodName_df = pd.read_csv("DataRareProdMergOrderTime.csv", encoding = "ISO-8859-1") # in this dataset, rare products are replaced with their aisle names
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
 'days_since_prior_order',
 'order_number',
 'nn']

user_idx = userIDprodNameprodID_df['user_id'].unique()
ndocs = userIDprodNameprodID_df['user_id'].max()
nwords = userIDprodNameprodID_df['product_id_tsfm'].max()
print("number of documents (users) and words (products) are:")
print( ndocs,nwords)


countOrder_series = userIDprodNameprodID_df.groupby(['user_id','order_id', 'product_id_tsfm','order_number','days_since_prior_order']).size()
new_df = countOrder_series.to_frame(name = 'size').reset_index()

newdf_sparsemat=csr_matrix((new_df['size'], ( new_df['user_id'],new_df['order_id'])))

newdf_sparsemat_orderProd=csr_matrix((new_df['size'], ( new_df['order_id'], new_df['product_id_tsfm'])))

n_components = 25
n_top_words = 20

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += "-- ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
####
print('data preparation completed...')


lda = LatentDirichletAllocation(n_topics=n_components, max_iter=100,
                                learning_method='online',
                                batch_size = 1000,
                                learning_offset=100. ,
                                random_state=0,
                                learning_decay = 0.6)
print('lda initialized')
#   
t0 = time()
lda.fit(newdf_sparsemat_orderProd)
print("done in %0.3fs." % (time() - t0))


#print_top_words(lda, prodIDindex, 7)

doc_word_dist = lda.transform(newdf_sparsemat_orderProd)

topicdist = lda.components_

topicdist=pd.DataFrame(topicdist)
doc_word_dist = pd.DataFrame(doc_word_dist)

topicdist.to_csv('topicdist.csv')
doc_word_dist.to_csv('doc_word_dist.csv')
