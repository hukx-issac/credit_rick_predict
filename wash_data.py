# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:09:37 2016

@author: Issac
"""

import matplotlib.pyplot as plt

def checkData(data): 
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    p = data.boxplot(return_type='dict') 
    x = p['fliers'][0].get_xdata() 
    y = p['fliers'][0].get_ydata()
    y.sort() #从小到大排序
        
    for i in range(len(x)): 
        if i>0:
            plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.05 -0.8/(y[i]-y[i-1]),y[i]))
        else:
            plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.08,y[i]))
                    
    plt.show() #展示箱线图
    
    
def washData(data,columns=\
             ['total_browse_num','browse_type_num',\
             'banknum','wage','income','expense',\
             'billnum','notreturn','notreturn_std','totalconsume','interest','adjust','currentbill','currentbill_std','creditline','creditline_std','card']):        
    data_1 = data[data['overdue']==1]
    data_0 = data[data['overdue']==0]
    data_1 = data_1[columns];data_0 = data_0[columns]
    for column in columns:
        temp_1 = data_1[column][data_1[column]>0]
        temp_0 = data_0[column][data_0[column]>0]
        des_1 = temp_1.describe()
        des_0 = temp_0.describe()
        U_1 = des_1['75%'] + 1.5*(des_1['75%'] - des_1['25%'])
        L_1 =  des_1['25%'] - 1.5*(des_1['75%'] - des_1['25%'])
        U_0 = des_0['75%'] + 1.5*(des_0['75%'] - des_0['25%'])
        L_0 =  des_0['25%'] - 1.5*(des_0['75%'] - des_0['25%'])
        data.loc[temp_1[temp_1 > U_1*1.1].loc[:].index,column] = des_1['50%']
        data.loc[temp_1[temp_1 < L_1].loc[:].index,column] = des_1['50%']
        data.loc[temp_0[temp_0 > U_0*1.1].loc[:].index,column]= des_0['50%']
        data.loc[temp_0[temp_0< L_0].loc[:].index,column] = des_0['50%']
    return data