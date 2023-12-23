import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from phishing_site_detection_model import *

pd.set_option("display.max_columns", None)

# Data Ingestion
df = pd.read_parquet("data/phishing_site_data.parquet.gzip")
#df = df[0:1000]

# Split train/test
df_train, df_test = train_test_split(df, test_size=0.3, random_state=1)
del df

# Training
# -----------------------
# Model training pipeline
print ("Training...")
pipe = model_training_pipeline(df_train)

# PMML artifact
path = "artifacts/phishing_site_detection_model.pmml"
#pipe_to_pmml(pipe, path)

# Pkl artifact
path = "artifacts/phishing_site_detection_model.pkl"
pipe_to_pkl(pipe, path)

print("Training Step Succeeded")

# Prediction on test set
# -----------------------
# Scoring
y_pred_proba = pipe.predict_proba(df_test['contents'])[:, 1]
df_test["phishing_site_score"] = y_pred_proba

# Test metrics
y_test = df_test["result"].values
y_pred = y_pred_proba > 0.5
metrics = classification_metrics(y_pred_proba, y_pred, y_test)

print("Prediction on Test Step Succeeded")