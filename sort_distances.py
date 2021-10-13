import pathlib
import joblib
import numpy as np

# Load data
PATH = pathlib.Path(pathlib.Path(__file__).parent.resolve()).parent.resolve()
directory = str(PATH) + r'\data\python_data'
Distance_Class_path = directory + r'\Distance_Output.joblib'

Distance_Class = joblib.load(Distance_Class_path)

def distributer(dist_class):
    new_class = []
    for i in range(dist_class.shape[0]):
       if np.any(dist_class[i]) == True:
           new_class.append(dist_class[i])
    
    return new_class

Distance_Class_sorted = distributer(Distance_Class)

joblib.dump(Distance_Class_sorted, directory + r'\Distance_Class_sorted.joblib')
