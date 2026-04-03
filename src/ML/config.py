##CONFIG FILE

#Constants
RANDOM_STATE=42
TEST_SIZE=0.2
CV_FOLDS=5

TARGET="COVAS"
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
WEARABLE_FEATURES=[] #to be deremined
