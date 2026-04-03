#main.py
#External imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#Own repo imports
from src.ML.config import RANDOM_STATE, TEST_SIZE, FULL_FEATURES
from src.ML.pipelines import create_LR_pipeline, create_RF_pipeline, create_SVR_pipeline
from src.ML.evaluation import train_and_evaluate
"""
Since i dont have the features yet, this function will generate random data for testing the pipeline and evaluation functions.
It will be removed once we have the real data.
"""
def make_random_data(n_samples=200):
    X=pd.DataFrame(
        np.random.randn(n_samples,len(FULL_FEATURES)),
                        columns=FULL_FEATURES
    )
    y=np.random.uniform(0,10,n_samples)
    return X,y
######################################################
#MAIN
X,y=make_random_data()
d=
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE)

pipeline=create_LR_pipeline()
metrics=train_and_evaluate(pipeline,X_train,y_train,X_test,y_test)

print("Results:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")