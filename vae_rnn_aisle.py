
# coding: utf-8

# # VAE RNN AISLE

# In[ ]:


#! pip3 install torch torchvision


# In[3]:


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
import copy
import sys

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


# In[4]:

print("===========Beginning Data Processing===============")


PATH = "~"
userIDprodName_df = pd.read_csv("data/DataRareProdMergOrderTime.csv", encoding = "ISO-8859-1")


# In[5]:


list(userIDprodName_df)


# In[12]:


newProdName=pd.factorize(userIDprodName_df['product_name'])
prodID = newProdName[0]
prodIDindex = newProdName[1]
userIDprodNameprodID_df = pd.concat([userIDprodName_df.reset_index(drop=True), pd.DataFrame(prodID)], axis=1)


userIDprodNameprodID_df["aisle"] = userIDprodNameprodID_df["aisle"].replace(userIDprodNameprodID_df['aisle'].unique(),
                                                                            list(range(len(userIDprodNameprodID_df['aisle'].unique()))))


# In[14]:


userIDprodNameprodID_df['aisle'].unique()


# In[9]:


list(userIDprodNameprodID_df)


# In[16]:


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
nwords = len(userIDprodNameprodID_df['aisle'].unique())
print("number of documents (users) and words (products) are:")
print( ndocs,nwords)


# In[17]:




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



hidden_size = 10
n_layers = 2
#inputsize = len(new_df['product_id_tsfm'].unique()) # 49689
#inputsize = doc_word_dist.shape[1]
inputsize = 134

print("===========Finished Data Processing===============")


# In[2]:


list(new_df)


# ## VAE Class
# 

# In[52]:



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


# In[51]:


def lossFun_vae(recon_x, targets, mu, logvar,inputsize,outputs):
    BCE = F.binary_cross_entropy(recon_x, targets, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    MSE = torch.mean((outputs - targets).pow(2))

    return  0.01*BCE + 0.01*KLD + MSE


# In[55]:

print("===========Beginning VAERNN Training===============")

n_epochs = 1
n_iters = n_users
model = VaeRNN(hidden_size,inputsize)
#criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
losses = np.zeros(n_epochs)


mu = np.random.multivariate_normal(np.transpose(np.zeros(inputsize)), np.identity(inputsize), n_iters)
logvar = []
for iin in range(n_iters):
    logvar.append(np.zeros(inputsize))


# In[56]:


mu.shape




t0= int(sys.argv[1])
kappa = float(sys.argv[2])
print("t0 {}, kappa {}".format(t0, kappa))

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
            rhot = Variable(torch.from_numpy(pow((t0+daysS),-kappa)), requires_grad = False).float()
            
            # Use teacher forcing 50% of the time
            force = random.random() < 0.5
            outputs, hidden, mutmp, logvartmp, recon_batch = model(inputs, user_g, rhot,None, force)
            mu[iter] = mutmp.detach().numpy()
            logvar[iter] = logvartmp.detach().numpy()[0]
    
            optimizer.zero_grad()
            loss = lossFun_vae(recon_batch,targets,mutmp,logvartmp,inputsize,outputs)
            loss.backward()
            optimizer.step()
    
            losses[epoch] += loss.data[0]
            if (iter%1000 == 0):
                print(iter)

    if epoch > -1:
        print((epoch), losses[epoch])
        


print("===========Finished VAERNN Training===============")
# In[41]:

print("===========Beginning Error Calculations===============")

l2errorLastOrder = np.zeros(n_iters)
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
            accuracy[q] = LA.norm(np.subtract(targets[q,:].view(1,-1)[0],outputstest[q,:].detach().numpy().transpose()),2)/inputsize
        l2errorLastOrder[j] = accuracy[-1] # 1-accuracy for prediction of the last order topic level
        highestTopicLastTarget[j] = np.argmax(targets[q,:])
        highestTopicLastPred[j] = np.argmax(outputstest[q,:].detach().numpy())
        if j % 10000 == 0:
            print(j)


# # Save Results

# In[46]:


np.save("data/t0_{}_k_{}_l2error.npy".format(t0, kappa), l2errorLastOrder)

print("SAVED "+ "t0_{}_k_{}_l2error.npy".format(t0, kappa))
print("===========DONE!===============")

