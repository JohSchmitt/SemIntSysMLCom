import numpy as np
import math
import pandas as pd

#RemoveOutlierInstancesByAverage
def ROIByAvg (X, y, timesAVG = 20, features='all'):
    
    # Calculate mean for every feature
    means = X.mean(axis=0)

    print(means)

    print(X)

    # Select Feature(s)
    if(features == 'all'):
        f = X
    else:
        f = X.iloc[:, features]

    itk = [] 
    
    mask_array = f[:] < timesAVG * means[features]
    itk.append(mask_array)

    # Boolean Array for instances to keep
    
    # For every feature create mask_array
    #for index, feature in enumerate(f):
    #    print(feature)
    #    mask_array = feature[:] < timesAVG * means[index]
    #    itk.append(mask_array)
    
    # Merge mask_array
    itk = np.all(itk, axis=0)
    
    # Apply mask_array
    X = X[itk]
    y = y[itk]
    
    return X, y

#SquareRoot feature
def squareroot (X, y, feature=0):
    
    # For every element
    for index, element in enumerate(X[feature]):
        X[feature][index] = math.sqrt(element)
    
    return X, y