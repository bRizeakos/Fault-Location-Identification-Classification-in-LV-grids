import os
import pathlib
import joblib
import numpy as np


def rescaler(choose_data, rescale):
    # Access path where dataset is stored
    PATH = pathlib.Path(pathlib.Path(__file__).parent.resolve()).parent.resolve()
    directory = str(PATH) + r'\data\python_data'
    Iabc_path = directory + r'\Ι_rescaled.joblib'
    voltage_int_path = directory + r'\V_rescaled.joblib'
    V_rescaled_norm_path = directory + r'\V_rescaled_norm.joblib' 
    I_rescaled_norm_path = directory + r'\I_rescaled_norm.joblib'
    
    # For Voltage rescaling    
    if not os.path.exists(V_rescaled_norm_path):
        
        V_int = joblib.load(voltage_int_path)
    
        if rescale == True:
            x_max = V_int.max()  
            x_min = V_int.min()   
            d_range = x_max - x_min
            
            # transform data
            V_rescaled = np.empty(V_int.shape, dtype=float)
            for scenario in range(len(V_int)):
                for node in range(len(V_int[0])):
                    for time in range(len(V_int[0,0])):
                        for meter in range(len(V_int[0,0,0])):
                            V_rescaled[scenario, node, time, meter] = (V_int[scenario, node, time, meter] - x_min)/d_range
                                
        else: V_rescaled = V_int
    
        joblib.dump(V_rescaled, V_rescaled_norm_path)
            
    else:
        V_rescaled = joblib.load(V_rescaled_norm_path)
    
    
    # For Current rescaling 
    if not os.path.exists(I_rescaled_norm_path):
        
        Iabc = joblib.load(Iabc_path)
            
        if rescale == True:
            x_max = Iabc.max()  
            x_min = Iabc.min()   
            d_range = x_max - x_min
        
            # transform data
            I_rescaled = np.empty(Iabc.shape, dtype=float)
            for scenario in range(len(Iabc)):
                for time in range(len(Iabc[0])):
                    for meter in range(len(Iabc[0,0])):
                            I_rescaled[scenario, time, meter] = (Iabc[scenario, time, meter] - x_min)/d_range
        
        else:
            I_rescaled = Iabc        
        
        joblib.dump(I_rescaled, I_rescaled_norm_path)
            
    else:
        I_rescaled = joblib.load(I_rescaled_norm_path)
    
    if choose_data == 'V': return V_rescaled
    elif choose_data == 'I': return I_rescaled