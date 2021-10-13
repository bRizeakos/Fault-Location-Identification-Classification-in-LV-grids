import pathlib
import joblib
import create_topology as top
import numpy as np


PATH = pathlib.Path(pathlib.Path(__file__).parent.resolve()).parent.resolve()
directory = str(PATH) + r'\data\python_data'
Class_path = directory + r'\Class.joblib'   

# scenario_length = datapoints for each scenario
scenario_length = [3715, 9535, 9535, 9535, 1022, 1022, 1022, 2384, 2384, 2384, 817, 1225]
names = ['Healthy', 'A-G', 'B-G', 'C-G', 'A-B', 'B-C', 'A-C', 'A-B-G', 'B-C-G', 'A-C-G', 'A-B-C', 'A-B-C-G']
     
Class = joblib.load(Class_path)
Class_Fault = []
Class_Location = []
for out in Class:
    num = ''
    if out[-2].isdigit():
        num += out[-2]
    if out[-1].isdigit():
        num += out[-1]
    Class_Location.append(int(num))
    Class_Fault.append(out[:-(2+len(num))])

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
faults_num = len(names)              # Number of fault types: 12
dataset_size = len(Class)            # Size of Dataset: 44580
distance_sections = 5                # Number of sections a branch is divided: 5

# Number of branches for each feeder
branches_per_feeder = [0 for i in range(feeders_num)]
for idx in range(len(grid_length)):
    if grid_length[idx][0] == 1:
        branches_per_feeder[0] +=1
    elif grid_length[idx][0] == 2:
        branches_per_feeder[1] +=1
    else:
        branches_per_feeder[2] +=1
    
# Encode the output of Faulty Feeder RNN
Feeder_Output = np.zeros([dataset_size, feeders_num], dtype = int) 

for scenario in range(dataset_size):
    if Class_Fault[scenario] == names[0]:
        pass
    else:
        for feeder in range(feeders_num):
            if top.has_path(nodes[0].children[feeder], [], nodes[Class_Location[scenario]]):
                Feeder_Output[scenario,feeder] = 1

joblib.dump(Feeder_Output, directory + r'\Feeder_Output.joblib')
                
# Encode the output of Faulty Branch RNN
Branch_Output =[]
    
for scenario in range(scenario_length[0], dataset_size):
    for idx in range(feeders_num):
        if Feeder_Output[scenario, idx] == 1:
            feeder_id = idx
    for feeder in range(sum(branches_per_feeder[:feeder_id]), sum(branches_per_feeder[:feeder_id]) + branches_per_feeder[feeder_id]):
        if nodes[Class_Location[scenario]] in grid_path[feeder]:
            out = [0 for i in range(branches_per_feeder[feeder_id])]
            out[feeder - sum(branches_per_feeder[:feeder_id])] = 1
            Branch_Output.append(out)
            break

joblib.dump(Branch_Output, directory + r'\Branch_Output.joblib')
        
# Encode the output of Class Fault RNN
Fault_Output = np.zeros([dataset_size, faults_num-1], dtype = int)
for scenario in range(scenario_length[0], dataset_size):
    if Class_Fault[scenario] != names[0]:
        for idx in range(1, faults_num):
            if Class_Fault[scenario] == names[idx]:
                Fault_Output[scenario, idx-1] = 1
                break

joblib.dump(Fault_Output, directory + r'\Fault_Output.joblib')
            
# Encode the output of Fault Distance RNN
even_distance = np.linspace(0, 1, 5)

Distance_Output = np.zeros([dataset_size, distance_sections], dtype = int)
for scenario in range(scenario_length[0], dataset_size):
    node_id = Class_Location[scenario] 
    for branch in range(len(grid_path)):
        node = nodes[node_id]
        if node in grid_path[branch]:
            corr_branch = branch
            for idx in range(len(grid_path[branch])):
                if node == grid_path[branch][idx]:
                    corr_idx = idx
            break
    
    distance = branch_distance[corr_branch][corr_idx]
    relative_distance = distance / grid_length[corr_branch][1]   
    rounded_distance = min(even_distance, key=lambda x:abs(x-relative_distance))
    
    for idx in range(len(even_distance)):
        if even_distance[idx] == rounded_distance:
            Distance_Output[scenario, idx] = 1

joblib.dump(Distance_Output, directory + r'\Distance_Output.joblib')

#%%
from tensorflow.keras.utils import to_categorical
# Encode the to_categorical output of Faulty Feeder RNN
Feeder_Output = np.zeros(dataset_size, dtype = int) 

for scenario in range(dataset_size):
    if Class_Fault[scenario] == names[0]:
        Feeder_Output[scenario] = 0
    else:
        for feeder in range(feeders_num):
            if top.has_path(nodes[0].children[feeder], [], nodes[Class_Location[scenario]]):
                Feeder_Output[scenario] = feeder+1
Feeder_Output_to_categorical = to_categorical(Feeder_Output)
joblib.dump(Feeder_Output_to_categorical, directory + r'\Feeder_Output_to_categorical.joblib')


#%%
# Encode the output of Faulty Feeder RNN
Feeder_Output = np.zeros([dataset_size, feeders_num+1], dtype = int) 

for scenario in range(dataset_size):
    if Class_Fault[scenario] == names[0]:
        Feeder_Output[scenario,feeders_num] = 1
    else:
        for feeder in range(feeders_num):
            if top.has_path(nodes[0].children[feeder], [], nodes[Class_Location[scenario]]):
                Feeder_Output[scenario,feeder] = 1

joblib.dump(Feeder_Output, directory + r'\Feeder_Output_4_Outputs.joblib')