
import pandas as pd
import numpy as np


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

# 3. Xử lý Policy bằng Vectorization
x_coords = next_move_coords_for_train[:, 0]
y_coords = next_move_coords_for_train[:, 1]
flat_indices = x_coords  + y_coords * 15

x_coords_eval = next_move_coords_for_eval[:, 0]
y_coords_eval = next_move_coords_for_eval[:, 1]
flat_indices_eval = x_coords_eval  + y_coords_eval * 15

policy_for_train = np.zeros((num_states, 225), dtype=np.float32)
policy_for_train[np.arange(num_states), flat_indices.astype(int)] = 1.0

policy_for_eval = np.zeros((num_states_eval, 225), dtype=np.float32)
policy_for_eval[np.arange(num_states_eval), flat_indices_eval.astype(int)] = 1.0

# tăng thêm data bằng cách xoay 90 độ và xoay 180 độ, 270 độ
def rotate_board(board):
    return np.rot90(board, k=1, axes=(1, 2))

def augment_data(board_states, policy):
    augmented_boards = [board_states]
    augmented_policies = [policy]

    for _ in range(3):  # Rotate 90, 180, and 270 degrees
        board_states = rotate_board(board_states)
        policy = rotate_board(policy.reshape(-1, 15, 15, 1)).reshape(-1, 225)
        augmented_boards.append(board_states)
        augmented_policies.append(policy)

    return np.concatenate(augmented_boards), np.concatenate(augmented_policies)

board_states_formatted, policy_for_train = augment_data(board_states_formatted, policy_for_train)
num_states = board_states_formatted.shape[0]
value_for_train = np.zeros( (num_states, 3), dtype=np.float32)
value_for_train[:, 1] = 1

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
print(board_states_formatted.shape)
print(policy_for_train.shape)
print(value_for_train.shape)

print(board_states_for_eval_formatted.shape)
print(policy_for_eval.shape)
print(value_for_eval.shape)
print("Data for training saved successfully.")