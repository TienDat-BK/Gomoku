import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from rapfi_core_tf import RapfiCustomModel, RapfiDataset
import numpy as np
import keras

model = RapfiCustomModel(
    in_channels=2,          # Channel: Ta, Địch
    mid_channels_for_directionBlock=32,
    out_channels_for_directionBlock=16,
    kernel_size=3
)


opt = keras.optimizers.Adam(learning_rate=0.001, weight_decay=5e-4)
model.compile(
    optimizer=opt,
    loss={
        "value": keras.losses.CategoricalCrossentropy(from_logits= True, label_smoothing=0.12),    # Cho Value (3,)
        "policy": keras.losses.CategoricalCrossentropy(from_logits= True, label_smoothing=0.12), # Cho Policy (225,)
    },
    
    loss_weights={
        "value": 1.0,    # đang tắt loss value để test policy
        "policy": 1.0, 
    },
    metrics={
         # Đo xem đoán thắng/thua đúng bao nhiêu %
            "policy": ["accuracy", ], # Đo xem đoán nước đi đúng bao nhiêu %
            "value": [keras.metrics.CategoricalAccuracy(),
                      keras.metrics.Precision(thresholds=0.33),
                      keras.metrics.Recall(thresholds=0.33),
                      keras.metrics.F1Score()]
    }
)

# load data
data = np.load("data_for_train/dataTrain.npz")
train_states = data['board_states']
train_policy_onehot = data['policy']
train_value_onehot = data['value']
num_samples = train_states.shape[0]

# load eval data
board_states_eval = np.load("data_for_train/dataEval.npz")['board_states']
policy_eval = np.load("data_for_train/dataEval.npz")['policy']
value_eval = np.load("data_for_train/dataEval.npz")['value']

# 1. Lấy nhãn từ dữ liệu gốc để tính trọng số cho toàn bộ tập dữ liệu
# Giả sử value_labels của bạn có shape (N, 3)
labels = np.argmax(train_value_onehot, axis=1)

# 2. Tính trọng số cho từng CLASS
# Tỷ lệ: Hòa (117k), Thắng (32k), Thua (18k)
class_weights = [6, 0, 0.2] # [Win_weight, Draw_weight, Lose_weight ]

# 3. Tạo một danh sách trọng số cho TỪNG MẪU (Sample)
sample_weights = [class_weights[l] for l in labels]
# sample_weights = torch.DoubleTensor(sample_weights)

# 4. Tạo Sampler
sampler = WeightedRandomSampler(
    weights=sample_weights, 
    num_samples=len(sample_weights), 
    replacement=True # Cho phép lấy lặp lại mẫu Thắng/Thua để cân bằng batch
)

# load dataset
dataset = RapfiDataset(train_states, train_value_onehot, train_policy_onehot)
dataLoader = DataLoader(dataset, batch_size= 128*2*2, num_workers=0, shuffle=True)

# load dataset_eval
dataset_eval = RapfiDataset(board_states_eval, value_eval, policy_eval)
dataLoader_eval = DataLoader(dataset_eval, batch_size=128*2*2, shuffle=True)

# # check phan phoi trong 1 batch
# for x, y in dataLoader:
#     val_labels = y['value'].numpy()
#     counts = np.sum(val_labels, axis=0)
#     print(f"Thực tế trong 1 Batch - Thua: {counts[0]}, Hòa: {counts[1]}, Thắng: {counts[2]}")
#     break # Chỉ xem 1 batch rồi dừng
# input()

model.fit(
    dataLoader,
    epochs=12,
    validation_data=dataLoader_eval,
    verbose=2
)

model.save_weights("model_saved/rapfi_model_policy_only_rotate_8.weights.h5")