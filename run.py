# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 21:01:14 2016

@author: Issac
"""
import os
import pandas as pd
from time import time
from model import ks_score, evaluate, model_RF, model_KM, model_GB, stacking_model, stacking_get_X_layer2
from send_email import sendResultByEmail
from preprocess import mulExtractFeature, mulNearestPolation, preprocess_feature1, preprocess_feature2, save_result

def main(cpu=3):
    # 数据根目录
    dataRoot_train = os.getcwd()+os.sep+'data'+os.sep+'train'
    dataRoot_test = os.getcwd()+os.sep+'data'+os.sep+'test'
    dataRoot_cache = os.getcwd()+os.sep+'data'+os.sep+'cache'
    
#==============================================================================
#     # 数据路径
#     userInfoPath_train = os.path.join(dataRoot_train,'user_info_train.csv')
#     browseHistoryPath_train = os.path.join(dataRoot_train,'browse_history_train.csv')
#     overduePath_train = os.path.join(dataRoot_train,'overdue_train.csv')
#     bankDetailPath_train = os.path.join(dataRoot_train,'bank_detail_train.csv')
#     loanTimePath_train = os.path.join(dataRoot_train,'loan_time_train.csv')
#     billDetailPath_train = os.path.join(dataRoot_train,'bill_detail_train.csv')
#     
#     userInfoPath_test = os.path.join(dataRoot_test,'user_info_test.csv')
#     browseHistoryPath_test = os.path.join(dataRoot_test,'browse_history_test.csv')
#     bankDetailPath_test = os.path.join(dataRoot_test,'bank_detail_test.csv')
#     loanTimePath_test = os.path.join(dataRoot_test,'loan_time_test.csv')
#     billDetailPath_test = os.path.join(dataRoot_test,'bill_detail_test.csv')
# 
#     # 读数据提取特征
#     data_userInfo_train = pd.read_csv(userInfoPath_train,header=None,names=['user_id','sex','job','education','marriage','residence_type'])
#     data_browseHistory_train = pd.read_csv(browseHistoryPath_train,header=None,names=['user_id','time','action_data','action_id'])
#     data_overdue_train = pd.read_csv(overduePath_train,header=None,names=['user_id','overdue'])
#     data_bankDetail_train = pd.read_csv(bankDetailPath_train,names=['user_id','time','type','amount','wage'])
#     data_loanTime_train = pd.read_csv(loanTimePath_train,names=['user_id','loan_time'])
#     data_billDetail_train = pd.read_csv(billDetailPath_train,names=['user_id','time','bank_id','last_bill_amount','last_return_amount',\
#                                                     'credit_line','current_balance','current_min_return','consume_count',\
#                                                     'current_bill_amount','adjust_amount','interest','available_balance','borrow_amount','status'])
#     mulExtractFeature(data_userInfo_train,data_browseHistory_train,data_loanTime_train,data_bankDetail_train,data_billDetail_train,data_overdue_train,cpu)
#     del data_userInfo_train,data_browseHistory_train,data_loanTime_train,data_bankDetail_train,data_overdue_train,data_billDetail_train
# 
#      
#     data_userInfo_test = pd.read_csv(userInfoPath_test,header=None,names=['user_id','sex','job','education','marriage','residence_type'])
#     data_browseHistory_test = pd.read_csv(browseHistoryPath_test,header=None,names=['user_id','time','action_data','action_id'])
#     data_bankDetail_test = pd.read_csv(bankDetailPath_test,names=['user_id','time','type','amount','wage'])
#     data_loadTime_test = pd.read_csv(loanTimePath_test,names=['user_id','loan_time'])
#     data_billDetail_test = pd.read_csv(billDetailPath_test,names=['user_id','time','bank_id','last_bill_amount','last_return_amount',\
#                                                     'credit_line','current_balance','current_min_return','consume_count',\
#                                                     'current_bill_amount','adjust_amount','interest','available_balance','borrow_amount','status'])
#     mulExtractFeature(data_userInfo_test,data_browseHistory_test,data_loadTime_test,data_bankDetail_test,data_billDetail_test,cpu=cpu)
#     del data_userInfo_test,data_browseHistory_test,data_loadTime_test,data_bankDetail_test,data_billDetail_test   
#     del dataRoot_train,dataRoot_test
#     del userInfoPath_train,browseHistoryPath_train,overduePath_train,bankDetailPath_train,loanTimePath_train
#     del userInfoPath_test,browseHistoryPath_test,bankDetailPath_test,loanTimePath_test
#     
#==============================================================================
    # 读取数据
    if cpu==1:
        data_train = pd.read_csv(os.path.join(dataRoot_cache,'train_origin_feature.csv'))
        data_pred = pd.read_csv(os.path.join(dataRoot_cache,'test_origin_feature.csv'))
    else:
        data_train = []
        data_pred = []
        for i in range(cpu):
            data_train.append(pd.read_csv(os.path.join(dataRoot_cache,'train_origin_feature%s.csv'%i)))
            data_pred.append(pd.read_csv(os.path.join(dataRoot_cache,'test_origin_feature%s.csv'%i)))
        data_train = pd.concat(data_train)
        data_pred = pd.concat(data_pred)
    data_train = data_train.set_index('user_id')
    data_pred = data_pred.set_index('user_id')
    data_train.sort_index(inplace=True)
    data_pred.sort_index(inplace=True)
    
#==============================================================================
#     dt = data_train[['feature_browse','feature_bank','feature_bill']]
#     dp = data_pred[['feature_browse','feature_bank','feature_bill']]
#==============================================================================
    
    # 预处理数据    
    tdata_111,data_train,data_pred = preprocess_feature1(data_train,data_pred)
    
    # 最邻近3个数据均值填补缺失值
    mulNearestPolation(tdata_111,data_pred)
    mulNearestPolation(tdata_111,data_train)
    
    
    
    # 读取填补后的数据
    data_train = []
    data_pred = []
    for i in range(3):
        data_train.append(pd.read_csv(os.path.join(dataRoot_cache,'train_polation_feature%s.csv'%i)))
        data_pred.append(pd.read_csv(os.path.join(dataRoot_cache,'test_polation_feature%s.csv'%i)))
    data_train = pd.concat(data_train)
    data_pred = pd.concat(data_pred)
    data_train = data_train.set_index('user_id')
    data_pred = data_pred.set_index('user_id')
    data_train.sort_index(inplace=True)
    data_pred.sort_index(inplace=True)
    
    data_train = preprocess_feature2(data_train)
    data_pred = preprocess_feature2(data_pred)
    
    # 训练模型    
    m,n = data_train.shape    
    X = data_train[range(n-1)]
    y = data_train['overdue']
    X_pred = data_pred[range(n-1)]
                       
#==============================================================================
#     X = X.join(dt)
#     X_pred=X_pred.join(dp)
#==============================================================================
       
    rf,gb,et,lr = stacking_model(X,y)      
    X_layer2 = stacking_get_X_layer2(rf,gb,et,X)
    model = lr
    X = X_layer2
    X_pred = stacking_get_X_layer2(rf,gb,et,X_pred) 

#    features = [u'sex', u'job_0', u'job_1', u'job_2', u'job_3', u'job_4',
#       u'education_0', u'education_1', u'education_2', u'education_3',
#       u'education_4', u'marriage_0', u'marriage_1', u'marriage_2',
#       u'marriage_3', u'marriage_4', u'marriage_5', u'residence_type_0',
#       u'residence_type_1', u'residence_type_2', u'residence_type_3',
#       u'residence_type_4', u'total_browse_num', u'browse_type_num',
#       u'banknum', u'wage', u'income', u'expense', u'billnum', u'notreturn',
#       u'totalconsume',u'currentbill', u'creditline']
       
#    X = X[features]
#    X_pred = X_pred[features]
             
#==============================================================================
#     rf = model_RF(X,y) 
#     model=rf
#==============================================================================
    
    y_train_pred = model.predict_proba(X)
    KS_train = ks_score(y,y_train_pred)
    P_train = model.score(X,y)
    F1_train = evaluate(model,X,y)
    result = model.predict_proba(X_pred)
    save_result(result)
    return KS_train,P_train,F1_train,model.get_params(),model
    
if __name__ == '__main__':
    start = time()
    KS_train,P_train,F1_train,params,model = main(cpu=3)
    end = time()
    delta = end - start
    m, s = divmod(delta, 60)
    h, m = divmod(m, 60)
    content = '''
    程序运行时间:%02d小时%02d分%02d秒.\n
    KS value for train set: {:.4f}.\n
    P value for train set: {:.4f}.\n
    F1 value for train set: {:.4f}.\n  
    best params:\n%s.
    '''.format(KS_train,P_train,F1_train) % (h,m,s,params)
    print content
    att_paths = [os.getcwd() + os.sep + 'output.csv']
    sendResultByEmail(content,att_paths,from_who='sever')
#    os.system('shutdown -s -t 60')    #关机  
