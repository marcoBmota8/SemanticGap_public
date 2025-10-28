import numpy as np

# Ground truth DAG adjacency matrix
B_gt = np.array([
    [0,1,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,1,0],
])
DISEASE_IDX = 11 # Index of row and column for the disease node


f_names = [
    'Smoking',           # 1 
    'Yellow_Fingers',    # 2  
    'Anxiety',           # 3
    'Peer_Pressure',     # 4
    'Genetics',          # 5 
    'Attention_Disorder',# 6
    'Born_an_Even_Day',  # 7
    'Car_Accident',      # 8
    'Fatigue',           # 9
    'Allergy',           # 10
    'Coughing',           # 11
    'Lung_Cancer'
]

# Clinically available variables (used for Diagnosis not for prediction)
f_available = [
    # 'Smoking',           # 1 # not usually recorded
    'Yellow_Fingers',    # 2  
    'Anxiety',           # 3
    # 'Peer_Pressure',     # 4 # Not usually observed in EHR
    # 'Genetics',          # 5 # hard to have patients sequenced
    'Attention_Disorder',# 6
    'Born_an_Even_Day',  # 7
    'Car_Accident',      # 8
    'Fatigue',           # 9
    'Allergy',           # 10
    'Coughing'           # 11
]

# Plotting feature mapping
f_map = {
    'Smoking':'S',             # 1
    'Yellow_Fingers':'YF',     # 2 
    'Anxiety':'Anx',           # 3
    'Peer_Pressure':'PP',      # 4 
    'Genetics': 'G',           # 5 
    'Attention_Disorder':'AD', # 6
    'Born_an_Even_Day': 'BaED',# 7
    'Car_Accident':'CA',       # 8
    'Fatigue':'F',             # 9
    'Allergy':'All',           # 10
    'Coughing':'C'             # 11
}

# Targets for prediction 
DISEASE = ['Lung_Cancer']
DIAGNOSES = ['Diagnosis_clean', 'Diagnosis_noisy']
