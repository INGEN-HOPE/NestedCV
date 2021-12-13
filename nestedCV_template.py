'''
Import statements
'''
#data libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#models
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

import xgboost
from xgboost import XGBClassifier
xgboost.set_config(verbosity=0)
from sklearn.model_selection import train_test_split
from sklearn import metrics

import sklearn
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix

from scipy import interp


'''
Template code demonstrating the Nested CV Pipeline as described in manuscript. For usage, see .ipynb
'''

#Nested CV Pipeline
def nestedcv(pipeline,param_grid, X, Y):

    #storing evluation metrics
    f1 = [0]*7
    prec = [0]*7
    rec = [0]*7
    acc = [0]*7
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    cv_outer = KFold(n_splits=7, shuffle=True)
    i = 0

    #train test split - 6 folds for train and 1 for test
    for train_ix, test_ix in cv_outer.split(X):

        X_train = X.iloc[train_ix]
        X_test =  X.iloc[test_ix]
        y_train = Y.iloc[train_ix]
        y_test =  Y.iloc[test_ix]

        #minmax normalization
        scaler = MinMaxScaler()
        model = ExtraTreesClassifier()
        model.fit(scaler.fit_transform(X_train),y_train)

        #feature importances using ExtraTreesClassifier
        feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
        x = feat_importances.nlargest(5)
        features = np.array(x.index)
        print(*features+',')


        #gridSearchCV for hyperparameter tuning
        gs = GridSearchCV(estimator=pipeline, param_grid = param_grid, cv = 6, scoring = 'f1', n_jobs = -1, refit = True)
        
        result = gs.fit(X_train[features],y_train)
        print(result.best_params_)
        best_model = result.best_estimator_
        
        #training best model
        best_model.fit(X_train[features],y_train)

        #testing
        y_hat = best_model.predict(X_test[features])

        #evaluation metrics for test set
        f1[i] = metrics.f1_score(y_test, y_hat)
        prec[i] = metrics.precision_score(y_test, y_hat)
        rec[i] = metrics.recall_score(y_test,y_hat)
        acc[i] = metrics.accuracy_score(y_test,y_hat)
        
        #roc curve
        viz = metrics.plot_roc_curve(best_model, X_test[features], y_test, name='ROC fold {}'.format(i+1), alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        i+=1
        print()

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
         
    arr = [np.mean(acc), np.mean(f1), mean_auc, np.mean(prec), np.mean(rec)]
    return arr


'''
Sample Usage

Let X be the 2D dataframe with multiple features as columns and Y be the outcome column dataframe.

# Pipeline created using Logistic Regression and MinMaxScaler
pipeline = make_pipeline(MinMaxScaler(),LogisticRegression(max_iter=10000))

param_grid = {'logisticregression__solver' : ['newton-cg', 'lbfgs', 'liblinear','sag','saga'],
'logisticregression__penalty' : ['l2'],
'logisticregression__C' : [300, 100, 30, 10, 3, 1.0, 0.3, 0.1, 0.03, 0.01]} 

lg = nestedcv(pipeline,param_grid,X,Y)

'''