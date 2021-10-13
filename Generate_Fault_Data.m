function out = Generate_Fault_Data(fault_name, fault_loads, fault_pv, fault_resistance, fault_start, fault_duration)
clc

%% Model to be tested

sys = 'LV_grid';
% Sampling time of node metrics
Tm = 20e-3;
% Stop time of simulation
Ts = 5.005;
%Power factor of loads
PF_load = 0.95;

Active_Loads = fault_loads;
Reactive_Loads = Active_Loads * tan(acos(PF_load));
PV_Loads = fault_pv;

%% Fault characteristics

FaultType     = fault_name;
Rground       = fault_resistance;
TonFault      = fault_start;
ToffFault     = fault_start + fault_duration;

%% Set parameters for Loads and Microgenerations

open_system(sys)

p1=Active_Loads(1);
P1 = num2str(p1);
set_param('LV_grid/Parallel RLC Load','ActivePower', P1);

q1=Active_Loads(1);
Q1 = num2str(q1);
set_param('LV_grid/Parallel RLC Load','InductivePower', Q1);

for i=1:47
    p=Active_Loads(i+1);
    P = num2str(p);
    path = strcat('LV_grid/Parallel RLC Load',int2str(i));
    set_param(path, 'ActivePower', P);
    
    q=Reactive_Loads(i+1);
    Q = num2str(q);
    set_param(path, 'InductivePower', Q);
end

for i=1:5
    p=PV_Loads(i);
    P_Ph = num2str(p);
    path = strcat('LV_grid/AC Voltage Source',int2str(i));
    set_param(path, 'Pref', P_Ph);
end

%% Characteristics of lines where faults can be applied (total of 38 fault locations)

%			   1	 2	   3	 4	   5	 6	   7	 8	   9
Lines.Name = {'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', ...
...            10	  11     12     13     14     15     16     17
              'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17', ...
...            18	  19     20     21     22     23     24     25                
              'L18', 'L19', 'L20', 'L21', 'L22', 'L23', 'L24', 'L25', ...
...            26	  27     28     29     30     31     32                  
              'L26', 'L27', 'L28', 'L29', 'L30', 'L31', 'L32'} ; % Line name
Lines.SName = Lines.Name;
Lines.Nphase = 3*ones(1,32); %number of phases
Lines.Nsection = [ 1 1 1 2 2 1 5 3 5 2 2 5 2 1 2 1 3 2 1 3 5 2 5 4 4 2 2 5 5 2 5 4] ; %number of sections


%% Selection of faulted lines

nLineSelect = 1:32; % run through all lines ex: [1,5]


%% Start

LineFaulted.Name     = {Lines.Name{nLineSelect}};
LineFaulted.SName    = {Lines.SName{nLineSelect}};
LineFaulted.Nphase   =  Lines.Nphase(nLineSelect);
LineFaulted.Nsection =  Lines.Nsection(nLineSelect);

iLineFaulted = randi(length(LineFaulted.Name));

%% Add Fault block in faulted line

LineName = [sys,'/',LineFaulted.Name{iLineFaulted}];
open_system(LineName) % open Line subsystem

Nphase		   = LineFaulted.Nphase(iLineFaulted);
Nsection	   = LineFaulted.Nsection(iLineFaulted);
FaultBlockName = [LineName, '/Fault'];

try
    hfb = add_block('powerlib/Elements/Three-Phase Fault',FaultBlockName);
catch ME
    % a Fault block is already present
    % delete it and replace with a new one at a different location (to avoid reconnection)
    warning('%s\n',ME.message)
    Position = get_param(FaultBlockName,'Position');
    LineHandles = get_param(FaultBlockName,'LineHandles');
    delete_line(LineHandles.LConn)
    delete_block(FaultBlockName);
    add_block('powerlib/Elements/Three-Phase Fault',FaultBlockName,'Position',Position+10);
end

%% Program fault resistance, type and timing
set_param(FaultBlockName,'GroundResistance',num2str(Rground));

if contains(FaultType,'A')
    set_param(FaultBlockName,'FaultA','on');
else
    set_param(FaultBlockName,'FaultA','off');
end

if contains(FaultType,'B')
    set_param(FaultBlockName,'FaultB','on');
else
    set_param(FaultBlockName,'FaultB','off');
end

if contains(FaultType,'C')
    set_param(FaultBlockName,'FaultC','on');
else
    set_param(FaultBlockName,'FaultC','off');
end

if contains(FaultType,'G')
    set_param(FaultBlockName,'GroundFault','on');
else
    set_param(FaultBlockName,'GroundFault','off');
end

set_param(FaultBlockName,'SwitchTimes',['[',num2str(TonFault),' ',num2str(ToffFault),']']);

%% Apply fault at line-section and terminal
iSection = randi(Nsection);
BlockName=[LineName,'/L',num2str(iSection)];
fprintf('Fault location: %s\n',BlockName);
hFault = get_param(FaultBlockName,'PortHandles');
hBlock = get_param(BlockName,'PortHandles');
hLine  = zeros(1,Nphase);
for iphase=1:Nphase
    hLine(iphase)=add_line(LineName,hFault.LConn(iphase),hBlock.RConn(iphase),'autorouting','smart');
end           


%% Simulate model 
simOut = sim('LV_grid');

simOut.Vabc  = {simOut.Vabc(2:end, :)};
simOut.Iabc  = {simOut.Iabc(2:end, :)};
simOut.Gen   = {simOut.Gen(2:end, :)};
simOut.Class = [FaultType,'-',LineFaulted.Name{iLineFaulted}];

out = {simOut.Vabc simOut.Iabc simOut.Class};

%% Delete Block after simulation

LineHandles = get_param(FaultBlockName,'LineHandles');
delete_line(LineHandles.LConn)
delete_block(FaultBlockName);