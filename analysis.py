# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 10:20:04 2016

@author: Issac
"""

import os
import sys
import pandas as pd
from model import model_KM
from preprocess import mulExtractFeature, preprocess_feature, save_result
from sklearn.cluster import KMeans

cpu=3
dataRoot_cache = os.getcwd()+os.sep+'data'+os.sep+'cache'

data_train = []
data_pred = []
for i in range(cpu):
    data_train.append(pd.read_csv(os.path.join(dataRoot_cache,'train_origin_feature%s.csv'%i)))
    data_pred.append(pd.read_csv(os.path.join(dataRoot_cache,'test_origin_feature%s.csv'%i)))
data_train = pd.concat(data_train)
data_pred = pd.concat(data_pred)
data_train = data_train.reset_index(drop=True)
data_pred = data_pred.reset_index(drop=True)

data_train = preprocess_feature(data_train)
data_pred = preprocess_feature(data_pred)

# 训练模型    
m,n = data_train.shape
X = data_train[range(1,n-1)]
X1 = X[range(22)]
y= data_train['overdue']
for i in range(4,30,1):
    km = KMeans(n_clusters=i,random_state=2,n_jobs=3)
    km.fit(X1)
    sys.stderr('i=%d,inertia=%f'%(i,km.inertia_))

#==============================================================================
# traindataroot = os.getcwd()+os.sep+'data'+os.sep+'train'
# billDetailPath = os.path.join(traindataroot,'bill_detail_train.csv')
# overduePath = os.path.join(traindataroot,'overdue_train.csv')
# loanTimePath = os.path.join(traindataroot,'loan_time_train.csv')
# 
# 
# data_billDetail = pd.read_csv(billDetailPath,names=['user_id','time','bank_id','last_bill_amount','last_return_amount',\
#                                                     'credit_line','current_balance','current_min_return','consume_count',\
#                                                     'current_bill_amount','adjust_amount','interest','available_balance','borrow_amount','status'])
# data_overdue = pd.read_csv(overduePath,names=['user_id','overdue'])
# data_loadTime = pd.read_csv(loanTimePath,names=['user_id','loan_time'])
# 
# result = pd.merge(data_billDetail,data_overdue,on='user_id')
# 
# a=result[(result['loan_time']-result['time'])<32000000]
# a=a[(a['loan_time']-a['time'])>0]
# a_p = a[a['overdue']==1]
# a_n = a[a['overdue']==0]
# 
# a_p = a_p[a_p['wage']==1]
# a_p=a_p[['user_id','amount']]
# a_pp = a_p.groupby('user_id').mean()
# 
# a_n = a_n[a_n['wage']==1]
# a_n=a_n[['user_id','amount']]
# a_nn = a_n.groupby('user_id').mean()
# t = result[(result['amount']*result['type'])>0]
# t1=t[t['overdue']==1]
# t2=t[t['overdue']==0]
# 
# t=pd.DataFrame(t)
#==============================================================================
