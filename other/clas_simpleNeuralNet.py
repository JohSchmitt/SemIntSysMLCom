#--------------------------------------------
# Imports
#--------------------------------------------
import pandas as pd
import numpy as np
import math
#import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler

import MyDataHandler as dat

from sklearn.neural_network import MLPClassifier

#--------------------------------------------
# Custom Functions
#--------------------------------------------

#RemoveOutlierInstancesByAverage
def ROIByAvg (X, y, timesAVG = 20, features='all'):
    
    # Calculate mean for every feature
    means = X.mean(axis=0)
       
    # Select Feature(s)
    if(features == 'all'):
        f = X.T
    else:
        f = X.T[features]
        
    # Boolean Array for instances to keep
    itk = [] 
    
    # For every feature create mask_array
    for index, feature in enumerate(f):
        mask_array = feature[:] < timesAVG * means[index]
        itk.append(mask_array)
    
    # Merge mask_array
    itk = np.all(itk, axis=0)
    
    # Apply mask_array
    X = X[itk]
    y = y[itk]
    
    return X, y


def squareroot (X, y, feature=0):
    
    # For every element
    for index, element in enumerate(X[feature]):
        X[feature][index] = math.sqrt(element)
    
    return X, y

#--------------------------------------------
# Pipeline Steps
#--------------------------------------------
rForest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None, min_samples_split=2,
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt',
                                   max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
                                   oob_score=False, n_jobs=None, random_state=3, verbose=0,
                                   warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=10000)

sc = StandardScaler()
neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 64, 32, 16, 8), random_state=1, max_iter = 1000)

#--------------------------------------------
# Pipeline
#--------------------------------------------
pipe = Pipeline(steps=[('Outlier_removal_f0', FunctionSampler(func=ROIByAvg, kw_args={'timesAVG': 22, 'features': [0]})),
                       #('Outlier_removal_f6', FunctionSampler(func=ROIByAvg, kw_args={'timesAVG': 22, 'features': [6]})),
                       #('SquareRoot', FunctionSampler(func=squareroot, kw_args={'feature': [0]})),
                       ("sc", sc),
                       ("NeuralN", neural)])

#--------------------------------------------
# Grid Search
#--------------------------------------------
param_grid = dict(Outlier_removal_f0 = [#'passthrough',
                                        #FunctionSampler(func=ROIByAvg, kw_args={'timesAVG': 23, 'features': [0]}),
                                        #FunctionSampler(func=ROIByAvg, kw_args={'timesAVG': 24, 'features': [0]}),
                                        #FunctionSampler(func=ROIByAvg, kw_args={'timesAVG': 25, 'features': [0]}),
                                        FunctionSampler(func=ROIByAvg, kw_args={'timesAVG': 26, 'features': [0]})
                                        #FunctionSampler(func=ROIByAvg, kw_args={'timesAVG': 27, 'features': [0]})
                                       ]
                  #Outlier_removal_f6 = ['passthrough'
                  #                      #FunctionSampler(func=ROIByAvg, kw_args={'timesAVG': 20, 'features': [6]})
                  #                     ],
                  #SquareRoot = ['passthrough',
                  #                      FunctionSampler(func=squareroot, kw_args={'features': [0]})
                  #                     ],
                  #"forest__n_estimators": [100],
                  #"forest__criterion": ['gini', 'entropy'],
                  #forest = [RandomForestClassifier(n_estimators=2000, criterion='entropy',bootstrap=True, random_state=0)]
                 )

search = GridSearchCV(pipe, param_grid, n_jobs=6)

X_train, y_train = dat.getData("clas", "train")

search.fit(X_train, y_train)

#--------------------------------------------
# Evaluation
#--------------------------------------------
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#--------------------------------------------
# Export to kaggle
#--------------------------------------------

# Open Test Data
X_test = dat.getData("clas", "test")

# Predict y
y_test = search.predict(X_test)

# Convert prediction
k_export = pd.DataFrame(y_test, columns = ['Predicted'])
k_export['Id'] = range(0, len(k_export))
k_export = k_export.reindex(columns=['Id','Predicted'])

# Get date and time
now = datetime.now()

# Save prediction
k_export.to_csv('pred_clas_neuralNetwork_'+now.strftime("%d.%m.%Y_%H.%M.%S")+'_'+str(search.best_score_)+'.csv', index=False)