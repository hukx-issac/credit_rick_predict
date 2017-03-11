# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 17:46:45 2016

@author: Issac
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit, KFold
from scipy.optimize import fmin_powell
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.ensemble import GradientBoostingRegressor as GR
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import fbeta_score, make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.cluster import KMeans

 
def func_ET(parameters,*args):
    parameters = map(lambda i: int(i) if i > 2 else 2,parameters)
    et = ET(max_depth=parameters[0],min_samples_split=parameters[1],min_samples_leaf=parameters[2],\
            n_estimators=100,n_jobs=3,class_weight='balanced')
    X = args[0]; y = args[1]
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    kf = KFold(n_splits=5,random_state=0, shuffle=True)
    score = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        et.fit(X_train,y_train)
        y_pred = pd.DataFrame(et.predict_proba(X_test),index=X_test.index)[1]
        score.append( ks_score(y_test,y_pred) )
    score = np.array(score)
    return np.mean(score)

def func_GB(parameters,*args):
    parameters = map(lambda i: int(i) if i > 2 else 2,parameters)
    gb = GB(max_depth=parameters[0],min_samples_split=parameters[1],min_samples_leaf=parameters[2])
    X = args[0]; y = args[1]
    gb.fit(X,y)
    y_pred = pd.DataFrame(gb.predict_proba(X),index=X.index)[1]
    return ks_score(y,y_pred)

def func_RF(parameters,*args):
    parameters = map(lambda i: int(i) if i > 2 else 2,parameters)
    rf = RF(max_depth=parameters[0],min_samples_split=parameters[1],min_samples_leaf=parameters[2],\
            n_estimators=100,class_weight='balanced',n_jobs=3,max_features="auto")
    X = args[0]; y = args[1]
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    kf = KFold(n_splits=5,random_state=0, shuffle=True)
    score = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        rf.fit(X_train,y_train)
        y_pred = pd.DataFrame(rf.predict_proba(X_test),index=X_test.index)[1]
        score.append( ks_score(y_test,y_pred) )
    score = np.array(score)
    return np.mean(score)

def func_LR(parameters,*args):
    C = parameters[0] if parameters[0]>0 else 1
    lr = LR(C = C)
    X = args[0]; y = args[1]
    kf = KFold(n_splits=5,random_state=0, shuffle=True)
    score =[]
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        lr.fit(X_train,y_train)
        y_test_pred = pd.DataFrame(lr.predict_proba(X_test),index=X_test.index)[1]
        score.append(ks_score(y_test,y_test_pred))
    return float(len(score))/sum(score)

def callback(xk):
    print xk
    
def opt_RF(X,y): 
    return fmin_powell(func_RF,x0=[10,2,1],args=(X,y),callback=callback,disp=True)

def opt_GB(X,y): 
    return fmin_powell(func_GB,x0=[10,2,1],args=(X,y),callback=callback,disp=True)

def opt_ET(X,y): 
    return fmin_powell(func_ET,x0=[10,2,1],args=(X,y),callback=callback,disp=True)

def opt_LR(X,y): 
    return fmin_powell(func_LR,x0=[5],args=(X,y),callback=callback,disp=True)
    
def opt_model_RF(X,y):
    parameters = opt_RF(X,y)
    parameters = map(lambda i: int(i) if i > 2 else 2,parameters)
    rf = RF(max_depth=parameters[0],min_samples_split=parameters[1],min_samples_leaf=parameters[2],\
            n_estimators=100,class_weight='balanced',n_jobs=3,max_features="auto",oob_score=True)
    rf.fit(X,y)
    return rf

def opt_model_GB(X,y):
    parameters = opt_GB(X,y)
    parameters = map(lambda i: int(i) if i > 2 else 2,parameters)
    gb = GB(max_depth=parameters[0],min_samples_split=parameters[1],min_samples_leaf=parameters[2])
    gb.fit(X,y)
    return gb

def opt_model_ET(X,y):
    parameters = opt_ET(X,y)
    parameters = map(lambda i: int(i) if i > 2 else 2,parameters)
    et = ET(max_depth=parameters[0],min_samples_split=parameters[1],min_samples_leaf=parameters[2],\
            n_estimators=100,n_jobs=3,class_weight='balanced')
    et.fit(X,y)
    return et

def stacking_model(X,y):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    kf = KFold(n_splits=3,random_state=0, shuffle=True)
    train_index, test_index = kf.split(X).next()
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    print 'In RF'
    rf = opt_model_RF(X_train,y_train)
    y_test_pred_rf = pd.DataFrame(rf.predict_proba(X_test),index=y_test.index,columns=['rf_0','rf_1'])['rf_1']
    print 'In GB'
    gb = opt_model_GB(X_train,y_train)
    y_test_pred_gb = pd.DataFrame(gb.predict_proba(X_test),index=y_test.index,columns=['gb_0','gb_1'])['gb_1']
    print 'In ET'
    et = opt_model_ET(X_train,y_train)
    y_test_pred_et = pd.DataFrame(et.predict_proba(X_test),index=y_test.index,columns=['et_0','et_1'])['et_1']
    X_layer2 = pd.concat([y_test_pred_rf,y_test_pred_gb,y_test_pred_et], axis=1, join_axes=[y_test.index])
    X_layer2.to_csv('X_layer2.csv')
    print 'In LR'
    opt_C = opt_LR(X_layer2,y_test).mean()
    lr = LR(C=opt_C)
    lr.fit(X_layer2,y_test)
    return rf,gb,et,lr
    
def stacking_get_X_layer2(rf,gb,et,X):
    y_rf = pd.DataFrame(rf.predict_proba(X),index=X.index,columns=['rf_0','rf_1'])['rf_1']
    y_gb = pd.DataFrame(gb.predict_proba(X),index=X.index,columns=['gb_0','gb_1'])['gb_1']
    y_et = pd.DataFrame(et.predict_proba(X),index=X.index,columns=['et_0','et_1'])['et_1']
    X_layer2 = pd.concat([y_rf,y_gb,y_et], axis=1, join_axes=[y_rf.index])
    return X_layer2
    
    
def ks_score(y_true, y_pred, pos_label=1):
    difference = []
    y_pred = pd.DataFrame(y_pred,index=y_true.index)[1]  
    pos = y_pred[y_true[y_true==pos_label].index]
    neg = y_pred[y_true[y_true!=pos_label].index]
    pos_total = float(len(pos))
    neg_total = float(len(neg))
    for k in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        difference.append(abs( len(pos[pos<=k])/pos_total - len(neg[neg<=k])/neg_total ))
    return max(difference)
    
        
def evaluate(model,X,y):
    y_pred = model.predict(X)
    return f1_score(y.values, y_pred, pos_label=1,average='weighted')


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,scoring=None,\
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

    
def model_RF(X,y):
    parameters = {'n_estimators':[250],
                  'max_features':['sqrt'],
                  'max_depth':[10,25,35],
                  'min_samples_split':[40],
                  'min_samples_leaf':[225],
                  'class_weight':['balanced'],
                  'n_jobs':[3],
                  'oob_score':[True]}
    f1_scorer = make_scorer(fbeta_score,beta=1,pos_label=1)
    ks_scorer = make_scorer(ks_score,needs_proba=True)
    rf = RF(class_weight='balanced',n_jobs=3,max_features='sqrt',n_estimators=250,max_depth=10)
    rf.fit(X,y)
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=2)
    grid_obj = GridSearchCV(rf,param_grid=parameters,scoring=ks_scorer,n_jobs=3,cv=cv) 
    grid_obj.fit(X,y)
    rf = grid_obj.best_estimator_
    
    title = "Learning Curves (Random Forest)"
    # Cross validation with 10 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    plot_learning_curve(rf, title, X, y, cv=3,scoring=f1_scorer, n_jobs=3,train_sizes=np.linspace(.1, 1.0, 10))
    plt.show()
    plt.savefig('learning_curve.png')
    return rf
    
def model_GB(X,y):
    parameters = {'n_estimators':[250],
                  'max_features':['sqrt'],
                  'max_depth':[25],
                  'min_samples_split':[40],
                  'min_samples_leaf':[225]
                  }
    f1_scorer = make_scorer(fbeta_score,beta=0.5,pos_label=1)
    ks_scorer = make_scorer(ks_score,needs_proba=True)
    gb = GB()
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=2)
    grid_obj = GridSearchCV(gb,param_grid=parameters,scoring=ks_scorer,n_jobs=3,cv=cv) 
    grid_obj.fit(X,y)
    gb = grid_obj.best_estimator_
    
    title = "Learning Curves (GB)"
    # Cross validation with 10 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    plot_learning_curve(gb, title, X, y, cv=3, n_jobs=3,train_sizes=np.linspace(.1, 1.0, 10))
    plt.show()
    plt.savefig('learning_curve.png')
    return gb


def model_GR(X,y):
    gr = GR()
    gr.fit(X,y)
    return gr
    
    
def model_KM(X,n_clusters=5):
    km = KMeans(n_clusters=n_clusters,random_state=2,n_jobs=3)
    km.fit(X)
    return km

    

    