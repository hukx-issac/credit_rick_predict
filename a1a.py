# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:31:43 2016

@author: Issac
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF

from scipy.optimize import fmin_powell

def func_RF(array,*args):
    array = map(lambda i: int(i) if i > 2 else 2,array)
    rf = RF(max_depth=array[0],min_samples_split=array[1],min_samples_leaf=array[2],\
            n_estimators=100,class_weight='balanced',n_jobs=3,max_features="auto")
    X = args[0]; y = args[1]
    rf.fit(X,y)
    y_pred = pd.DataFrame(rf.predict_proba(X),index=X.index)[1]
    return 1.0/ks_score(y,y_pred)

def callback(xk):
    print xk
    
def opt_RF(X,y):
    print fmin_powell(func_RF,x0=[10,2,1],args=(X,y),callback=callback,disp=True)
    
def ks_score(y_true, y_pred, pos_label=1):
    difference = []
    pos = y_pred[y_true[y_true==pos_label].index]
    neg = y_pred[y_true[y_true!=pos_label].index]
    pos_total = float(len(pos))
    neg_total = float(len(neg))
    for k in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        difference.append(abs( len(pos[pos<=k])/pos_total - len(neg[neg<=k])/neg_total ))
    return max(difference)
    
if __name__ == '__main__'  :
    opt_RF(X,y)