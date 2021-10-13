%% Load Scenario structure
clear variables
clc
Scenario = load([pwd,'\Scenario.mat']);

%% Call Generate_Fault_Data function 
counter = 6400; % keeps track how many scenarios we have run 
for fault_type = 2 : 2 %length(Scenario.Name)
    fault_name = Scenario.Name{1,fault_type};
    for idx = 6401 : length(Scenario.Loads{1,fault_type}{1,1})
        counter = counter + 1;
        fault_loads = Scenario.Loads{1,fault_type}{1,1}(idx,:);
        fault_pv = Scenario.Loads{1,fault_type}{1,2}(idx,:);
        fault_resistance = Scenario.Rs{1,fault_type}(idx);
        fault_start = Scenario.Time{1,fault_type}{1,1}(idx);
        fault_duration = Scenario.Time{1,fault_type}{1,2}(idx);
        Data_out = Generate_Fault_Data(fault_name, fault_loads, fault_pv, fault_resistance, fault_start, fault_duration);
        % Indices for saving data (x from 1 to 15, y from 1 to 743 and z
        % from 1 to 4)
        x = ceil(counter./2972);
        y = ceil((counter-(x-1)*2972)./4);
        z = mod(counter-1,4)+1;
        Dataset_rest2.Output{z,y,x} = {Data_out{1,1} Data_out{1,2}};
        Dataset_rest2.Class{z,y,x}  = Data_out{1,3};
        save('Dataset_rest2.mat', '-struct', 'Dataset_rest2');
    end
end

%% Save Dataset

save('Dataset_rest2.mat', '-struct', 'Dataset_rest2');