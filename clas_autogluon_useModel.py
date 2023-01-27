from autogluon.tabular import TabularDataset, TabularPredictor
from datetime import datetime
import pandas as pd
import MyDataHandler as dat

# Load predictor
predictor = TabularPredictor.load("./clas_model/")

# Load test data
X_test = dat.getData("clas", "test")

# Predict y with predictor
y_pred = predictor.predict(X_test)

# Print predictions
#print("Predictions:  \n", y_pred)

# Get date and time
now = datetime.now()

# Prepare export to kaggle
k_export = pd.DataFrame(y_pred)
k_export['Id'] = range(0, len(y_pred))
k_export = k_export.rename(columns={"label": "Predicted"})
k_export = k_export[["Id", "Predicted"]]

# Print predictions
print("Predictions:  \n", k_export)

# Save predictions as .csv
k_export.to_csv('./_predictions/pre_clas_autogl_'+now.strftime("%d.%m.%Y_%H.%M.%S")+'.csv', index=False)