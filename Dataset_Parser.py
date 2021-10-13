import os
import pathlib
import joblib
import mat73
import numpy as np


def create_dataset():
    # Load data
    PATH = pathlib.Path(pathlib.Path(__file__).parent.resolve()).parent.resolve()
    directory = str(PATH) + r'\data\python_data'
    
    print("Directory is ",directory)
    
    
    #value  = Data_i['Output'][#4,#743,#15][0,0:Voltage or 1:Current][0,0][#Timestep]
    #class  = Data_i['Class'][#4,#743,#15][0]
    
    # scenario_length = datapoints for each scenario
    scenario_length = [3715, 9535, 9535, 9535, 1022, 1022, 1022, 2384, 2384, 2384, 817, 1225]
    
    Dataset = []
    Class   = []
    
    # Open  dataset file
    
    filename = directory + r'\Dataset.mat'
    print("Data location is ",filename)
    if os.path.exists(filename):
        Data_i = mat73.loadmat(filename) #, variable_names=['Output','Class'])
        # For the number of instances of each scenario add the mat files data in a new Dataset file optimized for ML 
        for counter in range(sum(scenario_length)):
            instance = np.empty([250,108], dtype=float)
            for meter in range(108):
                # Check if any value is Voltage measurement
                if meter<99:
                    val = Data_i['Output'][counter][0][0][:,meter]
                    # Check if any value if non-valid and change it accordingly
                    for time in range(250):
                        check_nan = val[time]
                        if np.isnan(check_nan): 
                            if time == 0: val[time] = 0
                            else: val[time] = val[time-1]
                # Check if any value is Current measurement
                else:
                    val = Data_i['Output'][counter][1][0][:,meter-99]
                    # Check if any value if non-valid and change it accordingly
                    for time in range(250):
                        check_nan = val[time]
                        if np.isnan(check_nan): 
                            if time == 0: val[time] = 0
                            else: val[time] = val[time-1]
                instance[:,meter] = val
            Dataset.append(instance)
        Class = Data_i['Class'][:]
    
    joblib.dump(Dataset, directory + r'\Dataset.joblib')
    joblib.dump(Class, directory + r'\Class.joblib')
