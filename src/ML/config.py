##CONFIG FILE
import pandas as pd
import os
#Constants
RANDOM_STATE=42
TEST_SIZE=0.2
CV_FOLDS=5

TARGET="covas_max"
#Template features, NOT FINAL
FULL_FEATURES = [ 
    "bvp_mean", "bvp_std",
    "eda_mean", "eda_std",
    "temp_mean", "temp_slope",
    "resp_mean", "resp_std",
    "ecg_hr", "ecg_hrv",
    "emg_rms", "emg_energy",
]
#to be deremined
EXCLUDED=[ "subject_idx",   "window_idx","t_start_s","t_end_s","covas_mean","covas_max","covas_p90","covas_max","covas_min","covas_diff"]
WEARABLE_FEATURES=[] #to be deremined
csv_address= os.path.abspath(os.path.join(os.getcwd(), "..", "Features", "all_subjects_features.csv"))
def extract_wearable_features(df,signals):
    csv_address= os.path.abspath(os.path.join(os.getcwd(), "..", "Features", "all_subjects_features.csv"))
    columns= [col for col in pd.read_csv(csv_address).columns if any(signal in col for signal in signals)]
    return columns

WEARABLE_FEATURES= extract_wearable_features(pd.read_csv(csv_address), ["bvp","eda_e4","temp","ecg","subject_idx"])

"""
For full set we have the following best parameters:
BEST_PARAMS_RF = {
    'model__n_estimators': ?,
    'model__max_depth': ?,
    'model__min_samples_split': ?,
    'model__min_samples_leaf': ?
}
BEST_PARAMS_SVR = {
    'model__C': ?,
    'model__gamma': ?,
    'model__kernel': ?
}
"""

"""
For wearable set we have the following best parameters:
BEST_PARAMS_RF = {
    'model__n_estimators': ?,
    'model__max_depth': ?,
    'model__min_samples_split': ?,
    'model__min_samples_leaf': ?
}
BEST_PARAMS_SVR = {
    'model__C': ?,
    'model__gamma': ?,
    'model__kernel': ?
}
"""
