from autogluon.tabular import TabularDataset, TabularPredictor
import MyDataHandler as dat

# Load train data
train = dat.getData("clas", "train", True)

# Print loaded train data
# print(train)

# Set train data
train_data = TabularDataset(train)

# Print train_data
#print(train_data)

# Define vars
labelcolumn = "label"
modelpath   = "./clas_model/"
metric      = "log_loss"
time_limit  = None   # seconds             # alternative: "None"

# Train predictor
predictor = TabularPredictor(label=labelcolumn, path=modelpath, eval_metric=metric, problem_type="multiclass").fit(train_data=train_data, time_limit=time_limit, presets='best_quality')