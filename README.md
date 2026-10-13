### Multi-Agent Deep Reinforcement Learning for Adaptive Traffic Signal Control
___________________________________________________________________
#### STATE SPACE: 
N_queue_length=0

S_queue_length=0

E_queue_length=0

W_queue_length=0

Current_Active_Phase=0  

0 - NSA - N,S left and straight

1 - NSLA - N,S right

2 - EWA - E,W left and straight

3 - EWLA - E,W right


####  Ambulance_Flag=0 [0,1]
#### Ambulance_Lane= -1 [0,1,2,3]
-------

### Time elapsed in current phase:

Minimum_Green_time=30 (seconds)

state_space = np.array([N_queue_length, S_queue_length, E_queue_length, W_queue_length,Current_Active_Phase, Ambulance_Lane,])

----

#### ACTION SPACE

action_space = np.array([0, 1, 2, 3]) # 0: Change to NSA, 1: Change to NSLA, 2: Change to EWA, 3: Change to EWLA

----

##### REWARD FUNCTION

#code in loop using sb3 and buffer for delta Q

r = - ( delta_Q + lambda_1*penalty_for_ambulance_noncompliance + lambda_2*penalty_for_starvation) 

Total_Q_Length = N_queue_length + S_queue_length + E_queue_length + W_queue_length

----
#### MODEL DEFINITION
HYPERPARAMETERS AS IN DOC
