import math
import pathlib
import joblib
import numpy as np


# Access path where dataset is stored
PATH = pathlib.Path(pathlib.Path(__file__).parent.resolve()).parent.resolve()
directory = str(PATH) + r'\data\python_data'
V_rescaled_path = directory + r'\V_rescaled.joblib' 
I_rescaled_path = directory + r'\I_rescaled.joblib'
I_rescaled_noDC_path = directory+r'\Ι_rescaled_noDC.joblib'
V_rescaled_noDC_path = directory+r'\V_rescaled_noDC.joblib'

V_rescaled = joblib.load(V_rescaled_path)
#I_rescaled = joblib.load(I_rescaled_path)
#V_rescaled_noDC = joblib.load(V_rescaled_noDC_path)

for idx in range(V_rescaled.shape[0]):
    for branch in range(V_rescaled.shape[1]):
        for meter in range(V_rescaled.shape[3]):
            dc = V_rescaled[idx,branch,0,meter]
            for time in range(V_rescaled.shape[2]):
                V_rescaled[idx,branch,time,meter] = abs(V_rescaled[idx,branch,time,meter] - dc)
        
joblib.dump(V_rescaled, V_rescaled_noDC_path)

#%%

#V_rescaled = joblib.load(V_rescaled_path)
I_rescaled = joblib.load(I_rescaled_path)
#V_rescaled_noDC = joblib.load(V_rescaled_noDC_path)

for idx in range(I_rescaled.shape[0]):
    for meter in range(I_rescaled.shape[2]):
        dc = I_rescaled[idx,0,meter]
        for time in range(I_rescaled.shape[1]):
            I_rescaled[idx,time,meter] = abs(I_rescaled[idx,time,meter] - dc)
        
joblib.dump(I_rescaled, I_rescaled_noDC_path)