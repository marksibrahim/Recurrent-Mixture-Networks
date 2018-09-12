# -*- coding: utf-8 -*-
"""
@author: Ghazal

Creating new dataset where products that are appeared less than 20 times get replaced with their aisle name
"""

import pandas as pd
import numpy as np

transactions_df = pd.read_csv("transactions.csv", encoding = "ISO-8859-1")

countProductID = transactions_df.groupby('product_name').product_name.count()
countProdSorted = countProductID.sort_values(ascending=True)
lessthan20Index = countProdSorted[countProdSorted < 20].index
transactions_df['product_name'] = np.where(transactions_df['product_name'].isin(lessthan20Index),
                transactions_df['aisle'] , transactions_df['product_name'])

newProdName=pd.factorize(transactions_df['product_name'])
prodID = newProdName[0]
transactions_df = pd.concat([transactions_df.reset_index(drop=True), pd.DataFrame(prodID)], axis=1)

transactions_df.columns = ['Unnamed: 0',
 'order_id',
 'product_id',
 'add_to_cart_order',
 'reordered',
 'user_id',
 'eval_set',
 'order_number',
 'order_dow',
 'order_hour_of_day',
 'days_since_prior_order',
 'product_name',
 'aisle_id',
 'department_id',
 'aisle',
 'department',
 'product_id_tsfm']


transactions_df[['order_id','product_id','user_id','product_name','aisle','product_id_tsfm']].to_csv("DataRareProdMerg.csv")

newProdNameNew = pd.DataFrame(newProdName[1])
newProdNameNew.to_csv('mergedProdKey.csv')
