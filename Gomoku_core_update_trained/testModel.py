import os
from rapfi_core_tf import RapfiCustomModel, RapfiDataset
import keras
import torch
from torch.utils.data import Dataset, DataLoader

def test_rapfi_model():
    # 1. Khởi tạo Model
    model = RapfiCustomModel(
        in_channels=2,          # Channel: Ta, Địch
        mid_channels_for_directionBlock=32,
        out_channels_for_directionBlock=16,
        kernel_size=3
    )

    # Để model dựng hình (build) và in ra summary, ta cần gọi nó 1 lần với data giả
    # Input shape: (Batch=1, H=15, W=15, C=3)
    dummy_input = keras.ops.zeros((1, 15, 15, 2))
    value, policy = model(dummy_input).values()

    print(f"Value shape: {value.shape}")   # Mong đợi: (1, 3)
    print(f"Policy shape: {policy.shape}") # Mong đợi: (1, 225)

    model.summary()

    first_weight = model.weights[0]
    # Truy cập vào tensor thực tế bên dưới
    print(f"📍 Trọng số đang nằm trên: {first_weight.value.device}")

    # 2. Compile Model (Quan trọng)
    model.compile(
        optimizer="adam",
        
        # Map hàm Loss theo tên layer output (hoặc theo thứ tự trả về)
        loss={
            "value": keras.losses.CategoricalCrossentropy(from_logits=True),    # Cho Value (3,)
            "policy": keras.losses.CategoricalCrossentropy(from_logits=True), # Cho Policy (225,)
        },
        
        # Trọng số loss (AlphaZero thường để 1:1 hoặc chỉnh nhẹ)
        loss_weights={
            "value": 1.0,    
            "policy": 1.0, 
        },
        
        # Metrics để theo dõi độ chính xác
        metrics={
            "value": ["accuracy"], # Đo xem đoán thắng/thua đúng bao nhiêu %
            "policy": ["accuracy"] # Đo xem đoán nước đi đúng bao nhiêu %
        }
    )

    print("Model compiled successfully!")

    # 3. Huấn luyện Model (Ví dụ giả lập)
    import numpy as np

    # Giả lập data huấn luyện
    num_samples = 100000
    train_states = np.random.rand(num_samples, 15, 15, 2).astype(np.float32) # Input: (15,15,3)
    train_wdl = np.random.randint(0, 3, size=(num_samples,))                  # Output WDL: 0,1,2
    train_policy = np.random.randint(0, 225, size=(num_samples,))             # Output Policy: 0-224
    train_wdl_onehot = np.eye(3)[train_wdl]  # Chuyển sang one-hot encoding
    train_policy_onehot = np.eye(225)[train_policy]  # Chuyển sang one-hot encoding


    # load dataset
    dataset = RapfiDataset(train_states, train_wdl_onehot, train_policy_onehot)
    dataloader = DataLoader(dataset, batch_size=64*2*2, shuffle=True, num_workers=0)

    model.fit(
        dataloader,
        epochs=5
    )

if __name__ == "__main__":
    test_rapfi_model()