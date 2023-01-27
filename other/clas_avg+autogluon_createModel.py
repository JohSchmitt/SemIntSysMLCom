from autogluon.tabular import TabularDataset, TabularPredictor
import MyDataHandler as dat
import MyFunctions as fun

# Load train data
#train = dat.getData("clas", "train", True)

X_train, y_train = dat.getData("clas", "train", False)

data, label = fun.ROIByAvg(X_train, y_train, timesAVG = 20, features=[0])

train = data
train['label'] = label

# Print loaded train data
# print(train)

# Set train data
train_data = TabularDataset(train)

# Print train_data
#print(train_data)

# Define vars
labelcolumn = "label"
modelpath   = "./clas_model2/"
metric      = "log_loss"

# Train predictor
predictor = TabularPredictor(label=labelcolumn, path=modelpath, eval_metric=metric, problem_type="multiclass").fit(train_data=train_data, presets='best_quality')