import numpy as np

dataset = np.load("data_for_train/dataTrain.npz")
train_states = dataset['board_states']
train_policy_onehot = dataset['policy']
train_value_onehot = dataset['value']
train_value_onehot[train_value_onehot == np.max(train_value_onehot, axis=1, keepdims=True)] = 1
train_value_onehot[train_value_onehot != np.max(train_value_onehot, axis=1, keepdims=True)] = 0
print(f"check value prob: {np.sum(train_value_onehot, axis=0)}")
