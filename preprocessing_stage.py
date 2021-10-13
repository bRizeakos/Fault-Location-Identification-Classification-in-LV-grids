import os
import pathlib
import joblib
import numpy as np
import math
import Dataset_Parser as parser
import create_topology as top

def interpolate(x, y):
    # New interpolated nodes
    x_new = np.linspace(0, 1, 5)
    length = len(x)
    
    samp_y = np.empty([3, length], dtype =float).tolist()
    int_y =  np.empty([3, length], dtype =float).tolist()
    y_new =  np.empty([250, 15], dtype =float)
    
    
    for time in range(250):
        for i in range(3*length): 
            samp_y[i%3][math.floor(i/3)] = y[time][i]
        for phase in range(3):
            int_y[phase] = np.interp(x_new, x, samp_y[phase])
        for i in range(15): 
            y_new[time][i] = int_y[i%3][math.floor(i/3)]   
            
    return y_new

def grid_int(paths, Voltages):
    meter_len = len(paths)
    V_int = np.empty([meter_len, 250, 15], dtype =float)
    
    for path in range(meter_len):
        x = np.array([i/paths[path][-1] for i in paths[path]])
        y = Voltages[path,:,:3*meter_len]
        y_new = interpolate(x, y)
        V_int[path] = y_new
    
    return V_int

def scenario_int(paths, dataset):
    scenario_num = len(dataset)
    meter_len = len(paths)
    Dataset_int = np.empty([scenario_num, meter_len, 250, 15], dtype =float)
    
    for idx in range(scenario_num):
        Voltages = dataset[idx]
        V_int = grid_int(paths, Voltages)
        Dataset_int[idx] = V_int
    
    return Dataset_int

def preprocess():
    # Access path where dataset is stored
    PATH = pathlib.Path(pathlib.Path(__file__).parent.resolve()).parent.resolve()
    directory = str(PATH) + r'\data\python_data'
    dataset_path = directory + r'\Dataset.joblib'
    voltage_int_path = directory + r'\V_int.joblib'
    Iabc_path = directory + r'\Iabc.joblib'
    class_path = directory + r'\Class.joblib'
    
    if not (os.path.exists(dataset_path) and os.path.exists(class_path)):
        parser.create_dataset()
    
    Dataset = joblib.load(dataset_path)   
    
    # Run two previous functions: Dataset_Parser, create_topology
    tree, nodes, grid_length = top.create_grid()
    leaf_nodes = top.give_leaves(nodes[0])
    
    # Find all leaf nodes
    grid_path = []
    branch_distance = []
    for leaf in leaf_nodes:
        arr, dist = top.give_path(nodes[0],leaf)
        grid_path.append(arr)
        branch_distance.append(dist)
        
    
    feeders_num = len(nodes[0].children) # Number of root's children: 3
    branches_num = len(grid_path)        # Number of grid's branches: 9
    metrics_num = len(nodes)             # Number of voltage meters: 33
    dataset_size = len(Dataset)          # Size of Dataset: 44580
    
    # Max number of nodes in a branch: 8
    max_branch_length = 1
    for i in range(branches_num):
        if len(grid_path[i]) > max_branch_length: max_branch_length = len(grid_path[i])
    
    # Create branch segments
    X_int = np.zeros([len(grid_length), 5])
    for i in range(len(grid_length)):
        for j in range(5):
            X_int[i][j] = (j * grid_length[i][1])/4
        
    # Divide Dataset in Current measurements and Voltage measurements ordered in branch they fall into
    # Initialize arrays needed
    
    if not os.path.exists(Iabc_path):
    
        Iabc = np.empty([dataset_size, 250, 3*feeders_num], dtype=float)
        idx_I_columns = range(3*metrics_num, 3*(metrics_num + feeders_num))
        
        for idx in range(len(Dataset)):
            Iabc[idx] = Dataset[idx][:,idx_I_columns]
            
        joblib.dump(Iabc, directory + r'\Iabc.joblib') 
            
        del(Iabc)
        
    if not os.path.exists(voltage_int_path):   
        
        V_branch = np.empty([int(dataset_size/4), branches_num, 250, 3*max_branch_length], dtype=float)
        idx_V_columns = range(3*metrics_num)
            
        for part in range(4):
            path = directory + r'\Dataset_int'+ str(part+1) + r'.joblib'
            if not os.path.exists(path):
                for idx in range(int(part*len(Dataset)/4), int((part+1)*len(Dataset)/4)):
                    val = Dataset[idx][:,idx_V_columns]
                    for branch in range(branches_num):
                        for node in range(len(grid_path[branch])):
                            i = int(grid_path[branch][node].data[1:]) - 1
                            lst = val[:,3*i:3*(i+1)]
                            idx_new = idx - int(part*len(Dataset)/4)
                            V_branch[idx_new,branch,:,3*node:3*(node+1)] = lst
                
                data_int = scenario_int(branch_distance, V_branch)
                
                joblib.dump(data_int, path)
                
                del(data_int)
        del(V_branch)
        del(Dataset)            
        
        
        for i in range(4):   
            path = directory + r'\Dataset_int'+ str(i+1) + r'.joblib'
            data = joblib.load(path)
            if i == 0:
                V_int = data
            else:   
                V_int = np.concatenate((V_int, data))  
            
        del(data)
        joblib.dump(V_int, voltage_int_path)
    
    if 'Dataset' in dir(): del(Dataset)         
    
    
    return V_int