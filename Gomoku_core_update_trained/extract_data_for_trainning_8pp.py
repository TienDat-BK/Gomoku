
import pandas as pd
import numpy as np

import numpy as np
import re
from FastHeuristic7 import *
# --- SỬ DỤNG ---

heuristic = FastHeuristic7()


board_states_for_train = np.load("Gomoku/gomoku_dataset_split/train/full_board/board_states.npy")
board_states_for_eval = np.load("Gomoku/gomoku_dataset_split/test/full_board/board_states.npy")

next_move_coords_for_train = np.load("Gomoku/gomoku_dataset_split/train/full_board/next_moves_coords.npy")
next_move_players_for_train = np.load("Gomoku/gomoku_dataset_split/train/full_board/next_moves_players.npy")

next_move_coords_for_eval = np.load("Gomoku/gomoku_dataset_split/test/full_board/next_moves_coords.npy")
next_move_players_for_eval = np.load("Gomoku/gomoku_dataset_split/test/full_board/next_moves_players.npy")


num_states = board_states_for_train.shape[0]
num_states_eval = board_states_for_eval.shape[0]

value_for_train = np.zeros((num_states, 3), dtype=np.float32)
value_for_train[:, 1] = 1.0  # Giả sử tất cả đều là hòa
value_for_eval = np.zeros((num_states_eval, 3), dtype=np.float32)
value_for_eval[:, 1] = 1.0  # Giả sử tất cả

# 1. Khởi tạo mảng (15, 15, 2)
board_states_formatted = np.zeros((num_states, 15, 15, 2), dtype=np.float32)
board_states_for_eval_formatted = np.zeros((num_states_eval, 15, 15, 2), dtype=np.float32)

# 2. Xử lý Channel bằng Vectorization (Không dùng vòng lặp)
# Lấy mask của người chơi 1 và người chơi -1
mask_p1 = (board_states_for_train == 1.0)
mask_p2 = (board_states_for_train == -1.0)

mask_p1_eval = (board_states_for_eval == 1.0)
mask_p2_eval = (board_states_for_eval == -1.0)

# Lấy mask của người đến lượt đi (next_move_players)
is_p1_turn = (next_move_players_for_train == 1).reshape(-1, 1, 1)
is_p1_turn_eval = (next_move_players_for_eval == 1).reshape(-1, 1, 1)

# Nếu là lượt P1: Channel 0 = P1, Channel 1 = P2
# Nếu là lượt P2: Channel 0 = P2, Channel 1 = P1
board_states_formatted[..., 0] = np.where(is_p1_turn, mask_p1, mask_p2)
board_states_formatted[..., 1] = np.where(is_p1_turn, mask_p2, mask_p1)

board_states_for_eval_formatted[..., 0] = np.where(is_p1_turn_eval, mask_p1_eval, mask_p2_eval)
board_states_for_eval_formatted[..., 1] = np.where(is_p1_turn_eval, mask_p2_eval, mask_p1_eval)
# ... (Phần load data và xử lý channel giữ nguyên như code của bạn) ...

# 3. Xử lý Policy (Giữ nguyên logic của bạn: x + y*15)
# Lưu ý: Hãy chắc chắn x là Cột, y là Dòng để khớp với Row-Major
x_coords = next_move_coords_for_train[:, 0]
y_coords = next_move_coords_for_train[:, 1]
flat_indices = x_coords + y_coords * 15

x_coords_eval = next_move_coords_for_eval[:, 0]
y_coords_eval = next_move_coords_for_eval[:, 1]
flat_indices_eval = x_coords_eval + y_coords_eval * 15

policy_for_train = np.zeros((num_states, 225), dtype=np.float32)
policy_for_train[np.arange(num_states), flat_indices.astype(int)] = 1.0

policy_for_eval = np.zeros((num_states_eval, 225), dtype=np.float32)
policy_for_eval[np.arange(num_states_eval), flat_indices_eval.astype(int)] = 1.0

# --- CẢI TIẾN HÀM AUGMENT DATA ---
def augment_data(board_states, policy_flat):
    # board_states shape: (N, 15, 15, 2)
    # policy_flat shape: (N, 225)
    
    # Reshape policy về dạng ảnh 2D để có thể xoay/lật đồng bộ với bàn cờ
    # policy shape: (N, 15, 15)
    policy_img = policy_flat.reshape(-1, 15, 15)

    augmented_boards = []
    augmented_policies = []

    # Hàm rotate của numpy: np.rot90(m, k=1, axes=(0, 1))
    # Board có shape (N, H, W, C) -> Xoay trục H(1) và W(2)
    # Policy có shape (N, H, W)    -> Xoay trục H(1) và W(2)

    # Lặp 4 góc xoay: 0 độ (k=0), 90 độ (k=1), 180 độ (k=2), 270 độ (k=3)
    for k in range(4):
        # 1. Tạo bản xoay k lần
        b_rot = np.rot90(board_states, k=k, axes=(1, 2))
        p_rot = np.rot90(policy_img, k=k, axes=(1, 2))

        # --> Lưu bản xoay
        augmented_boards.append(b_rot)
        augmented_policies.append(p_rot.reshape(-1, 225)) # Flatten lại ngay

        # 2. Tạo bản LẬT NGANG (Horizontal Flip) từ bản đã xoay
        # Lật trục Width (Trục 2 của Board và Policy)
        # Lưu ý: np.flip(axis=2) với board (N,15,15,2) là lật chiều ngang (cột 0 <-> cột 14)
        b_flip = np.flip(b_rot, axis=2)
        p_flip = np.flip(p_rot, axis=2)

        # --> Lưu bản lật
        augmented_boards.append(b_flip)
        augmented_policies.append(p_flip.reshape(-1, 225))

    return np.concatenate(augmented_boards), np.concatenate(augmented_policies)

print("Dang thuc hien Data Augmentation (x8 lan du lieu)...")
board_states_formatted, policy_for_train = augment_data(board_states_formatted, policy_for_train)

print("heuristic cho value_train")
# heuristic cho value_train
num_states = board_states_formatted.shape[0]
value_for_train = np.zeros((num_states, 3), dtype=np.float32)
for i in range(num_states):
    value_for_train[i] = heuristic.evaluate(board_states_formatted[i])

print("heuristic cho value_eval")
# heuristic cho value_eval
num_states_eval = board_states_for_eval_formatted.shape[0]
value_for_eval = np.zeros((num_states_eval, 3))
for i in range(num_states_eval):
    value_for_eval[i] = heuristic.evaluate(board_states_for_eval_formatted[i])

# check phân phối value
print(f"check: {value_for_eval[10]}")

# # filter data
# ratio_draw_keep = 0.2
# ratio_loss_keep = 0.34

# labels = np.argmax(value_for_train, axis= 1)
# win_index = np.where(labels == 0)[0]
# draw_index = np.where(labels == 1)[0]
# loss_index = np.where(labels == 2)[0]
# print(f"Chi tiết: Win={len(win_index)}, Draw={len(draw_index)}, Loss={len(loss_index)}")

# num_draw_keep = int(len(draw_index) * ratio_draw_keep)
# num_loss_keep = int(len(loss_index) * ratio_loss_keep)

# idx_draw_choice = np.random.choice(draw_index, num_draw_keep, replace=False)
# idx_loss_choice = np.random.choice(loss_index, num_loss_keep, replace=False)

# final_choice = np.concatenate([win_index, idx_draw_choice, idx_loss_choice])
# np.random.shuffle(final_choice)

# # ap dung len data goc
# board_states_formatted = board_states_formatted[final_choice]
# policy_for_train = policy_for_train[final_choice]
# value_for_train = value_for_train[final_choice]



# Save data
print("Saving data...")
np.savez("data_for_train/dataTrain.npz", 
    board_states = board_states_formatted,
    policy = policy_for_train,
    value = value_for_train
)
np.savez("data_for_train/dataEval.npz", 
    board_states = board_states_for_eval_formatted, 
    policy = policy_for_eval,
    value = value_for_eval
)

print(f"Final Train Shape: {board_states_formatted.shape}")
print(f"Final Policy Shape: {policy_for_train.shape}")
print(f"final value train Shape: {value_for_train.shape}")
print("Data for training saved successfully.")

print(f"Final eval Shape: {board_states_for_eval_formatted.shape}")
print(f"Final Policy  eval Shape: {policy_for_eval.shape}")
print(f"final value eval Shape: {value_for_eval.shape}")
print("Data for training saved successfully.")