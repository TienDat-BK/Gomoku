import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Chọn 1 trong 3: "tensorflow", "torch", hoặc "jax"
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import ops
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class DirConv2d(keras.Layer):
    def __init__(self, in_channels, out_channels, direction, kernel_size=3, **kwargs):
        super(DirConv2d, self).__init__(**kwargs)
        self.direction = direction
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.init_layer_weights()

    def init_layer_weights(self):
        # 1. Tạo kernel
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size, self.kernel_size, self.in_channels, self.out_channels),
            initializer='he_uniform',
            trainable=True,
        )
    
        # 2. Tạo bias
        self.bias = self.add_weight(
            name='bias',
            shape=(self.out_channels,),
            initializer='zeros',
            trainable=True,
        )
        
        # 3. Tạo mask
        self.mask = self.create_direction_mask()
        self.built = True

    def create_direction_mask(self):
        center = self.kernel_size // 2
        mask = np.zeros((self.kernel_size, self.kernel_size, self.in_channels, self.out_channels), dtype=np.float32)

        if self.direction == "horizontal":
            mask[center, :, :, :] = 1.0
        elif self.direction == "vertical":
            mask[:, center, :, :] = 1.0
        elif self.direction == "diagonal":
            for i in range(self.kernel_size):
                mask[i, i, :, :] = 1.0
        elif self.direction == "anti-diagonal":
            for i in range(self.kernel_size):
                mask[i, self.kernel_size - 1 - i, :, :] = 1.0
        else: 
            raise ValueError("Invalid direction.")
            
        return ops.convert_to_tensor(mask, dtype="float32")
    

    def call(self, x):
        # Phép nhân mask ở đây để đảm bảo autodiff theo dõi được phép toán
        masked_kernel = ops.multiply(self.kernel, self.mask)
        
        x = ops.conv(
            x,
            kernel=masked_kernel,
            strides=1,
            padding='same',
            data_format="channels_last", # NHWC
        )
        
        return ops.add(x, self.bias)
    
class ResBlock(keras.Layer):
    def __init__(self, in_channels, direction, kernel_size=3):
        # (x -> DirConv2d -> Silu -> 1x1 Conv)  +  x -> output
        super(ResBlock, self).__init__()
        self.out_channels = in_channels

        self.activation = keras.layers.Activation('swish')  # SiLU activation
        # khoi thu 1
        self.bn1 = keras.layers.BatchNormalization()
        self.dir_conv = DirConv2d(in_channels, in_channels, direction, kernel_size=kernel_size)

        # khoi thu 2
        self.conv1d = keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=1,
            padding='same',
            data_format="channels_last", # NHWC
        )
        self.bn2 = keras.layers.BatchNormalization()
    
    def call(self, x, training=False):
        residual = x
        # khoi thu 1
        x = self.dir_conv(x)
        x = self.bn1(x, training=training)
        x = self.activation(x)

        # khoi thu 2
        x = self.conv1d(x)
        x = self.bn2(x, training=training)
        x = ops.add(x, residual)
        x = self.activation(x)
        return x

class OutBlock(keras.Layer):
    def __init__(self, in_channels):
        super(OutBlock, self).__init__()
        # ((x -> conv1x1 -> conv1x1 ->) + x ) +  conv1x1 -> output
        self.out_channels = in_channels
        self.in_channels = in_channels
        self.activation = keras.layers.Activation('swish')  # SiLU activation

        self.conv1 = keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=1,
            padding='same',
            data_format="channels_last", # NHWC
        )
        self.bn1 = keras.layers.BatchNormalization()

        self.conv2 = keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=1,
            padding='same',
            data_format="channels_last", # NHWC
        )
        self.bn2 = keras.layers.BatchNormalization()
    
        

    def call(self, x, training=False):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        x = ops.add(x, residual)

        return x

class ExtractDirectionalFeatures(keras.Layer):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3):
        super(ExtractDirectionalFeatures, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        
        directions = ["horizontal", "vertical", "diagonal", "anti-diagonal"]

        self.dir_convs = []

        for direction in directions:
            dir_conv = keras.Sequential([
                DirConv2d(in_channels, mid_channels, direction, kernel_size=kernel_size),
                keras.layers.BatchNormalization(),
                
                ResBlock(mid_channels, direction, kernel_size=kernel_size),
                ResBlock(mid_channels, direction, kernel_size=kernel_size),
                ResBlock(mid_channels, direction, kernel_size=kernel_size),
                ResBlock(mid_channels, direction, kernel_size=kernel_size),
                OutBlock(mid_channels),
                keras.layers.Conv2D(
                    filters=out_channels,
                    kernel_size=1,
                    padding='same',
                    data_format="channels_last", # NHWC
                ),
            ], name=f"dir_conv_{direction}")
            self.dir_convs.append(dir_conv)
        
        # Để Keras track được weights trong list, ta có thể đặt tên cho từng branch
        for i, model in enumerate(self.dir_convs):
            setattr(self, f"dir_conv_{directions[i]}", model)
    
    def call(self, x, training=False):
        y = []
        for dir_conv in self.dir_convs:
            x_dir = dir_conv(x, training=training)
            y.append(x_dir)
        
        return ops.concatenate(y, axis=-1)  # Nối theo kênh cuối

class CategoricalValueHead(keras.Layer):
    def __init__(self, hidden_dim=256, **kwargs):
        super().__init__(**kwargs)
        
        # 1. Bottleneck Conv: Giữ 4 kênh đặc trưng không gian
        # Input: (B, 15, 15, 64) -> Output: (B, 15, 15, 4)
        self.conv_reduce = keras.Sequential([
            keras.layers.Conv2D(filters=4, kernel_size=1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('swish')
        ])
        
        # 2. Fully Connected Layers
        self.flatten = keras.layers.Flatten()
        
        # Dense 1: Tổng hợp
        self.dense1 = keras.layers.Dense(hidden_dim, activation='swish')
        self.dropout = keras.layers.Dropout(0.3)
        
        # Dense 2: Logic sâu hơn
        self.dense2 = keras.layers.Dense(hidden_dim // 2, activation='swish')
        
        # 3. Output Layer: 3 Units (W, D, L)
        # Để activation=None để trả về Raw Logits (tốt hơn cho tính Loss)
        # Hoặc activation='softmax' nếu bạn muốn ra % ngay lập tức
        self.final_dense = keras.layers.Dense(3, activation=None, name="wdl_output") 

    def call(self, x, training=False):
        x = self.conv_reduce(x, training=training)
        x = self.flatten(x)
        
        x = self.dense1(x)
        if training:
            x = self.dropout(x, training=training)
            
        x = self.dense2(x)
        
        return self.final_dense(x) # Shape: (Batch, 3)

class PolicyHead(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.activation = keras.layers.Activation('relu')

        # Layer 1: Cửa sổ lớn để nhìn bao quát
        self.conv5x5 = keras.layers.Conv2D(
            filters=16, 
            kernel_size=5, 
            padding='same', 
            data_format="channels_last"
        )
        self.bn5x5 = keras.layers.BatchNormalization()

        # Layer 2: Tinh chỉnh
        self.conv3x3 = keras.layers.Conv2D(
            filters=8, 
            kernel_size=3, 
            padding='same', 
            data_format="channels_last"
        )
        self.bn3x3 = keras.layers.BatchNormalization()

        # Layer 3: Bottleneck trước khi Flatten
        self.conv1x1 = keras.layers.Conv2D(
            filters=4, 
            kernel_size=1, 
            padding='same', 
            data_format="channels_last"
        )
        self.bn1x1 = keras.layers.BatchNormalization()
        
        self.flatten = keras.layers.Flatten()
        
        # Output: (Batch, 225)
        self.final_dense = keras.layers.Dense(
            225, 
            name="policy_output"
        )
        

    def call(self, x, training=False):
        x = self.conv5x5(x)
        x = self.bn5x5(x, training=training)
        x = self.activation(x)

        x = self.conv3x3(x)
        x = self.bn3x3(x, training=training)
        x = self.activation(x)

        x = self.conv1x1(x)
        x = self.bn1x1(x, training=training)
        x = self.activation(x)

        x = self.flatten(x)
        x = self.final_dense(x)  # Shape: (Batch, 225)
        
        # Trả về vector phẳng là tốt nhất cho hàm Loss
        return x

class RapfiCustomModel(keras.Model):
    def __init__(self, in_channels=2, mid_channels_for_directionBlock=32, out_channels_for_directionBlock=16, kernel_size=3, **kwargs):
        super(RapfiCustomModel, self).__init__(**kwargs)
        
        self.feature_extractor = ExtractDirectionalFeatures(
            in_channels=in_channels,
            mid_channels=mid_channels_for_directionBlock,
            out_channels=out_channels_for_directionBlock,
            kernel_size=kernel_size
        )
        
        self.categorical_value_head = CategoricalValueHead(hidden_dim=256)
        self.policy_head = PolicyHead()
    
    def call(self, x, training=False):
        features = self.feature_extractor(x, training=training)
        
        value_output = self.categorical_value_head(features, training=training)
        policy_output = self.policy_head(features, training=training)
        
        return {"value": value_output, "policy": policy_output}

class RapfiDataset(Dataset):
    def __init__(self, input_states, value_labels, policy_labels):
        self.input_states = torch.from_numpy(input_states).float()
        self.value_labels = torch.from_numpy(value_labels).float()
        self.policy_labels = torch.from_numpy(policy_labels).float()
    
    def __len__(self):
        return len(self.input_states)
    def __getitem__(self, idx):
        x = self.input_states[idx]
        y = {
            "value": self.value_labels[idx],
            "policy": self.policy_labels[idx]}
        return x, y
