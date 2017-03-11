# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 22:00:44 2016

@author: Issac
"""

import os
import pandas as pd
import numpy as np
from model import model_GR, model_RF
from multiprocessing import Process

#==============================================================================
# #数据库
# from sqlalchemy import create_engine
# bankInfoPath = os.path.join(traindataroot,'bank_detail_train.csv')
# data_bankInfo = pd.read_csv(bankInfoPath,names=['user_id','time','type','amount','wage'])
# engine = create_engine('mysql://root:hkx921023@localhost:3306/credir_risk?charset=utf8')
# sql = pd.read_sql('SELECT action_data FROM browse_history_train WHERE user_id= %d'%row[0],engine)
#==============================================================================

def deal_sex(x):
    if x==2:
        return 0
    else:
        return 1


def extractFeature(data_userInfo,data_browseHistory,data_loanTime,data_bankDetail,data_billDetail,data_overdue=None,process_no=''):
    dataRoot_cache = os.getcwd()+os.sep+'data'+os.sep+'cache'
    data = data_userInfo
    # browse
    temp_feature_browse = []
    temp_total_browse = []
    temp_browse_type = []
    # bank
    temp_feature_bank = []
    temp_banknum = []
    temp_wage = []
    temp_income = []
    temp_expense = []
    # bill
    temp_feature_bill = []
    temp_billnum = []
    temp_notreturn = []
    temp_notreturn_std = []
    temp_totalconsume = []
    temp_interest = []
    temp_adjust = []
    temp_currentbill = []
    temp_currentbill_std = []
    temp_creditline = []
    temp_creditline_std = []
    temp_card = []
    for row in data.itertuples(index=False):
        # browse
        user_action_data = data_browseHistory[data_browseHistory['user_id']==row[0]]['action_data']
        if user_action_data.shape[0] == 0: 
            temp_feature_browse.append(0)
            temp_total_browse.append(None)
            temp_browse_type.append(None)
        else:
            temp_feature_browse.append(1)
            temp_total_browse.append(user_action_data.shape[0])
            temp_browse_type.append(user_action_data.value_counts().shape[0])
        # bank
        user_loan_time = data_loanTime[data_loanTime['user_id']==row[0]].iloc[0,1]
        user_bank_data = data_bankDetail[data_bankDetail['user_id']==row[0]]
        user_bank_data = user_bank_data[(user_bank_data['time']>user_loan_time-32000000)&(user_bank_data['time']<user_loan_time)]
        banknum = user_bank_data.shape[0]
        if banknum == 0:
            temp_feature_bank.append(0)
            temp_banknum.append(None)
            temp_wage.append(None)
            temp_income.append(None)
            temp_expense.append(None)
        else:
            temp_feature_bank.append(1)
            temp_banknum.append(banknum)
            income = user_bank_data[user_bank_data['type']==1]['amount']
            expense = user_bank_data[user_bank_data['type']==0]['amount']
            temp_income.append(income.sum())
            temp_expense.append(expense.sum())           
            wage = user_bank_data[user_bank_data['wage']==1]        
            if wage.shape[0] == 0:
                temp_wage.append(0)
            else:
                temp_wage.append(wage['amount'].median())
        
        # bill
        user_bill_data = data_billDetail[data_billDetail['user_id']==row[0]]
        billnum = user_bill_data.shape[0]
        if billnum == 0:
            temp_feature_bill.append(0)
            temp_billnum.append(None)
            temp_notreturn.append(None)
            temp_notreturn_std.append(None)
            temp_totalconsume.append(None)
            temp_interest.append(None)
            temp_adjust.append(None)
            temp_currentbill.append(None)
            temp_currentbill_std.append(None)
            temp_creditline_std.append(None)
            temp_creditline.append(None)
            temp_card.append(None)
        else:
            if billnum==1:
                temp_notreturn_std.append(0)
                temp_currentbill_std.append(0)
                temp_creditline_std.append(0)
            else:
                temp_notreturn_std.append( (user_bill_data['last_bill_amount']-user_bill_data['last_return_amount']).std())
                temp_currentbill_std.append( (user_bill_data['current_balance']-user_bill_data['current_bill_amount']).std() )
                temp_creditline_std.append( user_bill_data['credit_line'].std() )
            temp_feature_bill.append(1)
            temp_billnum.append(billnum)
            temp_notreturn.append( (user_bill_data['last_bill_amount']-user_bill_data['last_return_amount']).mean())
            temp_totalconsume.append( user_bill_data['consume_count'].mean())
            temp_interest.append( user_bill_data['interest'].mean())
            temp_adjust.append( user_bill_data['adjust_amount'].mean())
            temp_currentbill.append( (user_bill_data['current_balance']-user_bill_data['current_bill_amount']).mean() )
            temp_creditline.append( user_bill_data['credit_line'].mean() )
            temp_card.append( user_bill_data['bank_id'].value_counts().count() )
        
    data['feature_browse'] = temp_feature_browse
    data['total_browse_num'] = temp_total_browse
    data['browse_type_num'] = temp_browse_type
    data['feature_bank'] = temp_feature_bank
    data['banknum'] = temp_banknum
    data['wage'] = temp_wage
    data['income'] = temp_income
    data['expense'] = temp_expense
    data['feature_bill'] = temp_feature_bill
    data['billnum'] = temp_billnum
    data['notreturn'] = temp_notreturn
    data['notreturn_std'] = temp_notreturn_std
    data['totalconsume'] = temp_totalconsume
    data['interest'] = temp_interest
    data['adjust'] = temp_adjust
    data['currentbill'] = temp_currentbill
    data['currentbill_std'] = temp_currentbill_std
    data['creditline_std'] = temp_creditline_std
    data['creditline'] = temp_creditline
    data['card'] = temp_card
       
    if data_overdue is not None:
        data = pd.merge(data_userInfo,data_overdue,on='user_id')
        data.to_csv(os.path.join(dataRoot_cache,'train_origin_feature%s.csv'%process_no),index=False)
    else:
        data.to_csv(os.path.join(dataRoot_cache,'test_origin_feature%s.csv'%process_no),index=False)

 
def mulExtractFeature(data_userInfo,data_browseHistory,data_loanTime,data_bankDetail,data_billDetail,data_overdue=None,cpu=1):
    if cpu == 1:
        extractFeature(data_userInfo,data_browseHistory,data_loanTime,data_bankDetail,data_billDetail,data_overdue)
    else:       
        d = int(data_userInfo.shape[0]/cpu)
        processes = []
        for i in range(cpu):
            if i == (cpu-1):
                p = Process(target=extractFeature,\
                            args=(data_userInfo[d*i:],data_browseHistory,data_loanTime,data_bankDetail,data_billDetail,data_overdue,i))
            else:
                p = Process(target=extractFeature,\
                            args=(data_userInfo[d*i:d*(i+1)],data_browseHistory,data_loanTime,data_bankDetail,data_billDetail,data_overdue,i))
            processes.append(p)
        for proc in processes:
            proc.start()
        for proc in processes:
            proc.join()
        
        
# normalize
def preprocess_feature1(data_train,data_pred):    
    data_train['sex'] = data_train['sex'].map(deal_sex)
    data_pred['sex'] = data_pred['sex'].map(deal_sex)
    data_train = oneHot(data_train)
    data_pred = oneHot(data_pred)
    
    # 删除训练集中量小缺失多的值    
    tdata_ = getData(data_train)
    data_train = pd.concat([tdata_['001'],tdata_['110'],tdata_['101'],tdata_['011'],tdata_['111']])
    tdata_111 = tdata_['111']
   

    return tdata_111,data_train,data_pred
    

def preprocess_feature2(data):
    dataRoot_cache = os.getcwd()+os.sep+'data'+os.sep+'cache' 
    # 清洗数据
    data=washData(data)
    
    # 数据规约
    deal_col2 = ['total_browse_num','browse_type_num','banknum','wage','income','expense','billnum','notreturn','notreturn_std','totalconsume','interest','adjust','currentbill','currentbill_std','creditline','creditline_std','card']
    data[deal_col2] = (data[deal_col2]-data[deal_col2].mean())/data[deal_col2].std()
    
    if 'overdue' in data.columns:
        data.to_csv(os.path.join(dataRoot_cache,'train_processed_feature.csv'))
    else:
        data.to_csv(os.path.join(dataRoot_cache,'test_processed_feature.csv'))
    return data
    
def washData(data,columns=\
             ['total_browse_num','browse_type_num',\
             'banknum','wage','income','expense',\
             'billnum','notreturn','notreturn_std','totalconsume','currentbill','currentbill_std','creditline','creditline_std']):        
    des = data.describe()
    des = des[columns]
    U = des.loc['75%'] + 1.5*(des.loc['75%'] - des.loc['25%'])
    L =  des.loc['25%'] - 1.5*(des.loc['75%'] - des.loc['25%'])
    for column in columns:
        temp = data[column]
        data.loc[temp[ (temp>U[column]*1.1) ].loc[:].index,column] = des[column]['mean']
        data.loc[temp[ (temp<L[column]) ].loc[:].index,column] = des[column]['mean']
    return data
    
    
def save_result(predict_y): 
    dataRoot_test = os.getcwd()+os.sep+'data'+os.sep+'test'
    usersIDPath = os.path.join(dataRoot_test,'usersID_test.csv')
    data_usersID = pd.read_csv(usersIDPath,header=None,names=['user_id'])
    predict_y = pd.DataFrame(predict_y,index=data_usersID['user_id'])
    predict_y = predict_y[1]
    predict_y=predict_y.reset_index()
    predict_y.columns = ['userid','probability']
    predict_y.to_csv('output.csv',index=False)
   
 
def getData(data):
    data_000 = data[(data['banknum'].isnull()) & (data['total_browse_num'].isnull()) & (data['billnum'].isnull())]
    data_100 = data[(data['banknum'].notnull()) & (data['total_browse_num'].isnull()) & (data['billnum'].isnull())]    
    data_010 = data[(data['banknum'].isnull()) & (data['total_browse_num'].notnull()) & (data['billnum'].isnull())]
    data_001 = data[(data['banknum'].isnull()) & (data['total_browse_num'].isnull()) & (data['billnum'].notnull())]
    data_110 = data[(data['banknum'].notnull()) & (data['total_browse_num'].notnull()) & (data['billnum'].isnull())]
    data_101 = data[(data['banknum'].notnull()) & (data['total_browse_num'].isnull()) & (data['billnum'].notnull())]
    data_011 = data[(data['banknum'].isnull()) & (data['total_browse_num'].notnull()) & (data['billnum'].notnull())]
    data_111 = data[(data['banknum'].notnull()) & (data['total_browse_num'].notnull()) & (data['billnum'].notnull())]
    data = {'000':data_000,
            '100':data_100,
            '010':data_010,
            '001':data_001,
            '110':data_110,
            '101':data_101,
            '011':data_011,
            '111':data_111}
    return data
    

def oneHot(originData):
    data = pd.DataFrame(index=originData.index)
    deal_col = ['job','education','marriage','residence_type']
    for col, col_data in originData.iteritems():
        if col in deal_col:
            col_data = pd.get_dummies(col_data,prefix=col)
        data = data.join(col_data)
    return data
    

def clcDistance(vec1,vec2,xxx):
    index = {'000':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
             '100':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,26,27,28,29],
             '010':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24],
             '001':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,31,32,33,34,35,36,37,38,39,40,41],
             '110':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,26,27,28,29],
             '101':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,26,27,28,29,31,32,33,34,35,36,37,38,39,40,41],
             '011':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,31,32,33,34,35,36,37,38,39,40,41]}
    vec1 = vec1[index[xxx]]
    vec2 = vec2[index[xxx]]
    dis = np.linalg.norm(vec1 - vec2)
    return dis

    
def nearestData(data_111,vec,xxx,num=100):
    dis = []
    for i in range(data_111.shape[0]):
        dis.append( clcDistance(data_111.iloc[i],vec,xxx) )
    dis = pd.DataFrame(dis,columns=['distance'])
    dis = dis.sort_values('distance')
    data_nearest = data_111.iloc[dis.index[0:num]]
    return data_nearest.describe().loc['mean']
    
  
def nearestPolation(tdata_111,data,process_no):
    dataRoot_cache = os.getcwd()+os.sep+'data'+os.sep+'cache' 
    data_ = getData(data)
    label = ['000','100','010','001','110','101','011']
    for la in label:
        for i in range(data_[la].shape[0]):
            mean = nearestData(tdata_111,data_[la].iloc[i],la)
            data_[la].iloc[i] = data_[la].iloc[i].fillna(mean)
    data = pd.concat([data_['000'],data_['100'],data_['010'],data_['001'],data_['110'],data_['101'],data_['011'],data_['111']]) 
    if ('overdue' in data.columns):
        data.to_csv(os.path.join(dataRoot_cache,'train_polation_feature%s.csv'%process_no))
    else:
        data.to_csv(os.path.join(dataRoot_cache,'test_polation_feature%s.csv'%process_no))

        
def mulNearestPolation(tdata_111,data):
    cpu = 3
    d = int(data.shape[0]/cpu)
    processes = []
    for i in range(cpu):
        if i == (cpu-1):
            p = Process(target=nearestPolation,args=(tdata_111,data[d*i:],i))
        else:
            p = Process(target=nearestPolation,args=(tdata_111,data[d*i:d*(i+1)],i))
        processes.append(p)
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()
        
        
        
        
        
       
        
        
        
        
        
        
        
        
def getFirstResult(data_train,data_pred,des):
    data_train = oneHot(data_train)
    data_pred = oneHot(data_pred)
    data_train['sex'] = data_train['sex'].map(deal_sex)
    data_pred['sex'] = data_pred['sex'].map(deal_sex)
    data_train.fillna(des.loc['50%'],inplace=True)
    data_pred.fillna(des.loc['50%'],inplace=True)
    m,n = data_train.shape   
    rf = model_RF(data_train[range(1,n-1)],data_train['overdue'])
    data_pred.sort_index(inplace=True)
    result = rf.predict(data_pred[range(1,n-1)])
    result = pd.DataFrame(result,index=data_pred.index)
    dataRoot_cache = os.getcwd()+os.sep+'data'+os.sep+'cache'
    result.to_csv(os.path.join(dataRoot_cache,'firstPredictedResult.csv'),index=False)
    
    
def interpolationFunc(originData):
    dataRoot_cache = os.getcwd()+os.sep+'data'+os.sep+'cache'   
    originData['sex'] = originData['sex'].map(deal_sex)
    data = oneHot(originData)   
    data = washData(data,originData.describe())
    data.drop(['interest','adjust'],1,inplace=True)
    
    data_000,data_100,data_010,data_001,data_110,data_101,data_011,data_111 = getData(data)

    # 填补浏览数据
    bank_columns = ['banknum','wage','income','expense']
    train_x11 = pd.concat([data_011,data_111]).drop(bank_columns,1)
    pred_x01 = pd.concat([data_001,data_101]).drop(bank_columns,1) 
    browse_columns = ['total_browse_num','browse_type_num']
    browse_model = []
    for col in browse_columns:
        model = model_GR(train_x11.drop(browse_columns,1),train_x11[col])
        result = model.predict(pred_x01.drop(browse_columns,1))
        result = pd.DataFrame(result,index=pred_x01.index,columns=[col])
        data.fillna(result,inplace=True)
        browse_model.append(model)
    data_000,data_100,data_010,data_001,data_110,data_101,data_011,data_111 = getData(data)
    
    # 填补信用卡数据
    train_111 = data_111
    pred_110 = data_110
    bill_columns = ['billnum','notreturn','notreturn_std','totalconsume','currentbill','currentbill_std','creditline_std','creditline','card']
    bill_model = []
    for col in bill_columns:
        model = model_GR(train_111.drop(bill_columns,1),train_111[col])
        result = model.predict(pred_110.drop(bill_columns,1))
        result = pd.DataFrame(result,index=pred_110.index,columns=[col])
        data.fillna(result,inplace=True)
        bill_model.append(model)
    data_000,data_100,data_010,data_001,data_110,data_101,data_011,data_111 = getData(data)
    
    # 填补银行数据
    train_111 = data_111
    pred_011 = data_011
    bank_columns = ['banknum','wage','income','expense']
    bank_model = []
    for col in bank_columns:
        model = model_GR(train_111.drop(bank_columns,1),train_111[col])
        result = model.predict(pred_011.drop(bank_columns,1))
        result = pd.DataFrame(result,index=pred_011.index,columns=[col])
        data.fillna(result,inplace=True)
        bank_model.append(model)
    
    data.sort_index(inplace=True)    
    data.to_csv(os.path.join(dataRoot_cache,'train_interpolation_feature.csv'))
    
    return data,browse_model,bill_model,bank_model
    
    
def interpolationTest(testData,browse_model,bill_model,bank_model):
    dataRoot_cache = os.getcwd()+os.sep+'data'+os.sep+'cache'   
    
    testData.sort_index(inplace=True)    
    testData['sex'] = testData['sex'].map(deal_sex)
    data = pd.DataFrame(index=testData.index)
    deal_col = ['job','education','marriage','residence_type']
    for col, col_data in testData.iteritems():
        if col in deal_col:
            col_data = pd.get_dummies(col_data,prefix=col)
        data = data.join(col_data)
    data.drop(['interest','adjust'],1,inplace=True)
        
    firstResult = pd.read_csv(os.path.join(dataRoot_cache,'firstPredictedResult.csv'))
    firstResult.index = data.index
    data['overdue'] = firstResult['0']
    data_000,data_100,data_010,data_001,data_110,data_101,data_011,data_111 = getData(data)
    
    # 填补浏览数据
    bank_columns = ['banknum','wage','income','expense']
    pred_x01 = pd.concat([data_001,data_101]).drop(bank_columns,1) 
    browse_columns = ['total_browse_num','browse_type_num']
    for i,col in enumerate(browse_columns):
        model = browse_model[i]
        result = model.predict(pred_x01.drop(browse_columns,1))
        result = pd.DataFrame(result,index=pred_x01.index,columns=[col])
        data.fillna(result,inplace=True)
    data_000,data_100,data_010,data_001,data_110,data_101,data_011,data_111 = getData(data)
    
    # 填补信用卡数据
    pred_110 = data_110
    bill_columns = ['billnum','notreturn','notreturn_std','totalconsume','currentbill','currentbill_std','creditline_std','creditline','card']
    for i,col in enumerate(bill_columns):
        model = bill_model[i]
        result = model.predict(pred_110.drop(bill_columns,1))
        result = pd.DataFrame(result,index=pred_110.index,columns=[col])
        data.fillna(result,inplace=True)
    data_000,data_100,data_010,data_001,data_110,data_101,data_011,data_111 = getData(data)
    
    # 填补银行数据
    pred_011 = data_011
    bank_columns = ['banknum','wage','income','expense']
    for i,col in enumerate(bank_columns):
        model = bank_model[i]
        result = model.predict(pred_011.drop(bank_columns,1))
        result = pd.DataFrame(result,index=pred_011.index,columns=[col])
        data.fillna(result,inplace=True)
    
    data.drop(['overdue'],1,inplace=True)
    data.sort_index(inplace=True)    
    data.to_csv(os.path.join(dataRoot_cache,'test_interpolation_feature.csv'))
    
    return data
    
    