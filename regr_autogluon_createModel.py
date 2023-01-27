from autogluon.tabular import TabularDataset, TabularPredictor
import MyDataHandler as dat

# Load train data
train = dat.getData("regr", "train", True)

# Print loaded train data
# print(train)

# Set train data
train_data = TabularDataset(train)

# Print train_data
#print(train_data)

# Define vars
labelcolumn = "label"
modelpath   = "./regr_model/"
#metric      = "median_absolute_error"     # alternative: "median_absolute_error" or "mean_absolute_error"
time_limit  = None   # seconds             # alternative: "None"

# Train predictor
predictor = TabularPredictor(label=labelcolumn, path=modelpath, problem_type="regression").fit(train_data=train_data, time_limit=time_limit, presets='best_quality')