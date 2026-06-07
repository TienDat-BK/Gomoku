import torch
from torch.utils.data import Dataset, DataLoader
from rapfi_core_tf import RapfiCustomModel, RapfiDataset
import numpy as np
import keras

model = RapfiCustomModel(
    in_channels=2,          # Channel: Ta, Địch
    mid_channels_for_directionBlock=32,
    out_channels_for_directionBlock=16,
    kernel_size=3
)

board_states = np.load("data_for_train/dataEval.npz")['board_states']
policy = np.load("data_for_train/dataEval.npz")['policy']
value = np.load("data_for_train/dataEval.npz")['value']
num_samples = board_states.shape[0]

dummy_input = keras.ops.zeros((1, 15, 15, 2))
_ = model(dummy_input, training=False)
model.load_weights("model_saved/rapfi_model_policy_only_rotate_8.weights.h5")

model.compile(
    optimizer="adam",
    loss={
        "value": keras.losses.CategoricalCrossentropy(),    # Cho Value (3,)
        "policy": keras.losses.CategoricalCrossentropy(), # Cho Policy (225,)
    },
    metrics={
        "value": [keras.metrics.CategoricalAccuracy()], # Đo xem đoán thắng/thua đúng bao nhiêu %
        "policy": [ keras.metrics.CategoricalAccuracy()] # Đo xem đoán nước đi đúng bao nhiêu %
    },
)

results = model.evaluate(
    x=board_states, 
    y={"value": value, "policy": policy},
    batch_size=256,
    
)

print(results)

