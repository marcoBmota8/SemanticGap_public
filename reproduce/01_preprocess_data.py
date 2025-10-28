import pandas as pd

from scipy.io import loadmat

# Variables names 
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
    'Coughing'           # 11
]
target = 'Lung_Cancer'

# Transform to dataframe
train = pd.DataFrame(
    loadmat('./raw_data/lucas0_train.mat')['X'],
    columns = f_names
)

# Add the original dataset `target`, lung cancer disease, as another node of the Bayesian Causal Network samples
target_train = pd.read_csv('./raw_data/lucas0_train.targets', delimiter=',', header=None).replace({-1:0})
train[target] = target_train.astype(int)

# Save processed dataset to csv
train.to_csv('./data/data.csv')

# Save order of variables
with open("./data/variable_order.txt", "w") as f:
    for i,item in enumerate(train.columns.tolist()):
        f.write(f"{i}. {item}\n")


