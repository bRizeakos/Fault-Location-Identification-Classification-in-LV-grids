import pathlib
import joblib
import tensorflow
import numpy as np
import random
import create_topology as top

random.seed()
rand_num = int(100 * random.random())
print("seed=", rand_num)
cce = tensorflow.keras.losses.CategoricalCrossentropy()

# # GPU usage
# tensorflow.get_logger().setLevel(logging.INFO)
# physical_devices = tensorflow.config.list_physical_devices("GPU")
# tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

# Load data
PATH = pathlib.Path(pathlib.Path(__file__).parent.resolve()).parent.resolve()
directory = str(PATH) + r'\data\python_data'
V_rescaled_noDC_path = directory + r'\V_rescaled_noDC.joblib'
Feeder_Class_path = directory + r'\Feeder_Output_4_Outputs.joblib'
Branch_Class_path = directory + r'\Branch_Output.joblib'
Fault_Class_path = directory + r'\Fault_Output.joblib'

V_rescaled_noDC = joblib.load(V_rescaled_noDC_path)
Feeder_Output = joblib.load(Feeder_Class_path)
Branch_Output = joblib.load(Branch_Class_path)
Fault_Class = joblib.load(Fault_Class_path)

tree, nodes, grid_length = top.create_grid()
leaf_nodes = top.give_leaves(nodes[0])

# Find all leaf nodes
grid_path = []
branch_distance = []
for leaf in leaf_nodes:
    arr, dist = top.give_path(nodes[0], leaf)
    grid_path.append(arr)
    branch_distance.append(dist)
feeders_num = len(nodes[0].children)  # Number of root's children: 3
branches_num = len(grid_path)         # Number of grid's branches: 9
metrics_num = len(nodes)              # Number of voltage meters: 33
distance_sections = 5                 # Number of sections a branch is divided: 5


def right_feeder(feeder_class):
    outp = []
    for scenario in range(len(feeder_class)):
        index = np.argmax(feeder_class[scenario])
        outp.append(index)
    return outp        


def right_branch(branch_class):
    outp = []
    for scenario in range(len(branch_class)):
        index = np.argmax(branch_class[scenario])
        outp.append(index)
    return outp  


def distributer(dataset, feeder_class, branch_class, fault_class, num_of_feeders, leafs):
    new_dataset = []
    new_class = []
   
    branch_per_feeder = []
    for idx in range(num_of_feeders):
        path = []
        for lf in leafs:
            ar, _ = top.give_path(nodes[1+idx], lf)
            if len(ar) != 0:
                path.append(ar)
        branch_per_feeder.append(len(path))
        
    offset = 0    
    for idx in range(len(feeder_class)):
        if feeder_class[idx] == num_of_feeders:
            offset += 1
        else:
            feeder = feeder_class[idx]
            branch = branch_class[idx - offset]
            
            new_dataset.append(dataset[idx, 
                                       sum(branch_per_feeder[:feeder])+branch, 
                                       :, :])
            new_class.append(fault_class[idx])
            
    return new_dataset, new_class


squeezed_feeder_class = right_feeder(Feeder_Output)
squeezed_branch_class = right_branch(Branch_Output)
V_branch_noDC, Fault_Class_sorted = distributer(V_rescaled_noDC, squeezed_feeder_class, squeezed_branch_class,
                                                 Fault_Class, feeders_num, leaf_nodes)


joblib.dump(V_branch_noDC, directory + r'\V_branch_noDC.joblib')
joblib.dump(Fault_Class_sorted, directory + r'\Fault_Class_sorted.joblib')
