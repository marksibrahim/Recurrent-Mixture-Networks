# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:45:37 2018

@author: Ghazal
"""

from __future__ import print_function
from time import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import gamma
from scipy.stats import multivariate_normal
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import math, random
from numpy import linalg as LA
import matplotlib.pyplot as plt
import copy

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


PATH = "~"
userIDprodName_df = pd.read_csv(PATH + "DataRareProdMergOrderTime.csv", encoding = "ISO-8859-1")
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
nwords = userIDprodNameprodID_df['aisle'].max()
print("number of documents (users) and words (products) are:")
print( ndocs,nwords)


countOrder_series = userIDprodNameprodID_df.groupby(['user_id','order_id', 'aisle','order_number','days_since_prior_order']).size()
new_df = countOrder_series.to_frame(name = 'size').reset_index()
#new_df_new = countOrder_series.to_frame(name = 'size').reset_index()
###############
################
# mapping order ids and order number
dictOrderIDOrdernum = dict(zip(new_df['order_id'],new_df['order_number']))
dictOrderIDDayS = dict(zip(new_df['order_id'],new_df['days_since_prior_order']))
################


newdf_sparsemat=csr_matrix((new_df['size'], ( new_df['user_id'],new_df['order_id'])))

newdf_sparsemat_orderAisle=csr_matrix((new_df['size'], ( new_df['order_id'], new_df['aisle'])))


#################
n_users = len(new_df['user_id'].unique())
####################################################################################################

n_epochs = 50
n_iters = n_users
hidden_size = 10
n_layers = 1
#inputsize = len(new_df['product_id_tsfm'].unique()) # 49689
#inputsize = doc_word_dist.shape[1]
inputsize = 134






class VaeRNN(nn.Module):
    def __init__(self, hidden_size, inputsize):
        super(VaeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = inputsize
        self.output_size = inputsize
        self.inp = nn.Linear(inputsize, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)
        self.out = nn.Linear(hidden_size, inputsize)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.linenc1 = nn.Linear(inputsize,inputsize)
        self.linenc2 = nn.Linear(inputsize,inputsize)
        self.linenc22 = nn.Linear(inputsize,inputsize)
        self.linenc3 = nn.Linear(inputsize,inputsize)
        self.linenc4 = nn.Linear(inputsize,inputsize)
        #self.relu = nn.ReLU()
        
        
    def encode(self, x):
        h1 = F.relu(self.linenc1(x))
        return self.linenc2(h1), self.linenc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
          
        
    def decode(self, z):
        h3 = F.relu(self.linenc3(z))
        return self.sigmoid(self.linenc4(h3))

        
    def step(self, input, hidden):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        
        #output = self.sigmoid(output)
        return output, hidden

    def forward(self, inputs, user_g, rhot,hidden=None, force=True,steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, inputsize,1))
        decodeVec = Variable(torch.zeros(steps, inputsize,1))
        for i in range(steps):
            if force or i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)
            mu, logvar = self.encode(user_g.view(-1,inputsize))
            z = self.reparameterize(mu, logvar)
            outputsTmp = rhot[i]*output.t() + (1.-rhot[i])*(self.decode(z).t())
            outputs[i] = self.softmax(outputsTmp) # for when expetimenting using aisles
            decodeVec[i]  = self.decode(z).t()
           
        return outputs, hidden, mu, logvar, decodeVec


def lossFun_vae(recon_x, x, mu, logvar,inputsize,outputs,targets):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    MSE = torch.mean((outputs - targets).pow(2))

    return  0.1*BCE + 0.1*KLD + MSE


model = VaeRNN(hidden_size,inputsize)
#criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
losses = np.zeros(n_epochs)


mu = np.random.multivariate_normal(np.transpose(np.zeros(inputsize)), np.identity(inputsize), n_iters)
logvar = []
for iin in range(n_iters):
    logvar.append(np.zeros(inputsize))

for epoch in range(n_epochs):

    for iter in range(n_iters-1):
        tmp = np.nonzero(newdf_sparsemat[iter+1].todense())[1]
        tmpprod = newdf_sparsemat_orderAisle[tmp].todense()
        
        row_sums = tmpprod.sum(axis=1)
        new_matrix = tmpprod / row_sums
        
        
        _inputs = new_matrix
        if _inputs.shape[0]>2:
            inputs = Variable(torch.from_numpy(_inputs[:-2]).float()).unsqueeze(2) # use all orders excpet the last one for each user
            targets = Variable(torch.from_numpy(_inputs[1:-1]).float().unsqueeze(2)) # use shifted input as output
            daysS = np.asarray([dictOrderIDDayS[x] for x in tmp])[:-1]
            
            user_g = torch.from_numpy(multivariate_normal.rvs(mean = mu[iter],cov=np.diag(np.exp(np.asarray(logvar[iter]))))).float()
            rhot = Variable(torch.from_numpy(np.ones(len(daysS))), requires_grad = False).float()
            
            # Use teacher forcing 50% of the time
            force = random.random() < 0.5
            outputs, hidden, mutmp, logvartmp, recon_batch = model(inputs, user_g, rhot,None, force)
            mu[iter] = mutmp.detach().numpy()
            logvar[iter] = logvartmp.detach().numpy()[0]
    
            optimizer.zero_grad()
            loss = lossFun_vae(recon_batch,targets,mutmp,logvartmp,inputsize,outputs,targets)
            loss.backward()
            optimizer.step()
    
            losses[epoch] += loss.data[0]

    if epoch > -1:
        print((epoch), loss.data[0])
        




plt.plot(losses)




##########################################################
##### Assesing Accuracy

lastAcc = np.zeros(n_iters)
aisleAcc = np.zeros(n_iters)
highestTopicLastTarget = np.zeros(n_iters)
highestTopicLastPred = np.zeros(n_iters)

for j in range(n_iters-1):
    tmp = np.nonzero(newdf_sparsemat[j+1].todense())[1]
    tmpprod = newdf_sparsemat_orderAisle[tmp].todense()
    
    row_sums = tmpprod.sum(axis=1)
    new_matrix = tmpprod / row_sums
        
        
    _inputs = new_matrix
    if _inputs.shape[0]>2:
        inputs = Variable(torch.from_numpy(_inputs[:-1]).float()).unsqueeze(2) # use all orders excpet the last one for each user
        targets = Variable(torch.from_numpy(_inputs[1:]).float().unsqueeze(2)) # use shifted input as output
        daysS = np.asarray([dictOrderIDDayS[x] for x in tmp])
        
        user_g = torch.from_numpy(multivariate_normal.rvs(mean = mu[j],cov=np.diag(np.exp(np.asarray(logvar[j]))))).float()
        #user_g = copy.copy(mu[iter])
        rhot = Variable(torch.from_numpy(np.ones(len(daysS))), requires_grad = False).float()
        
        # Use teacher forcing 50% of the time
        outputstest, hiddentest, mutmptest, logvartmptest, recon_batchtest = model(inputs, user_g, rhot,None)
        
        
        outputstest = outputstest.squeeze(2)
        accuracy = np.zeros(targets.size()[0])
        for q in range(targets.size()[0]):
            accuracy[q] = LA.norm(np.subtract(targets[q,:],outputstest[q,:].detach().numpy()),2)
        lastAcc[j] = accuracy[-1] # accuracy for prediction of the last order topic level
        highestTopicLastTarget[j] = np.argmax(targets[q,:])
        highestTopicLastPred[j] = np.argmax(outputstest[q,:].detach().numpy())
        if j % 10000 == 0:
            print(j)











