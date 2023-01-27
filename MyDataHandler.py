import pandas as pd

# Define main data path
data_path = "./_data/"

# Define challenge sub paths
classification_path = "seminar-isg-ml-competition-ws22-classification/"
regression_path     = "seminar-isg-ml-competition-ws22-regression/"

# Define data file names
train_features_name = "train_features.csv"
train_labels_name   = "train_label.csv"
test_features_name  = "test_features.csv"

def getData(challenge, type, table=False):

    path = data_path

    if(challenge == "clas"):
        path += classification_path
    elif(challenge == "regr"):
        path += regression_path

    if(type == "train"):
        # Open Data
        features = pd.read_csv(path + train_features_name)
        labels   = pd.read_csv(path + train_labels_name)

        # Drop ID Columns
        features = features.drop(labels='Id', axis=1)
        labels = labels.drop(labels='Id', axis=1)
        #labels = labels.values.ravel()
        
        if(table):
            train_data = features
            train_data['label'] = labels
            return train_data

        else:
            return [features, labels]

    elif(type == "test"):
        # Open Data
        features = pd.read_csv(path + test_features_name)

        # Drop ID Columns
        features = features.drop(labels='Id', axis=1)
        
        return features