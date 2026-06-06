
from core.Board import *
from core.rapfi_core_tf import RapfiCustomModel
import numpy as np
import torch

class AI_agent:
    def __init__(self, board : Board, color : str):
        self.board = board
        self.pieceColor = color
        self.rapfi_core = RapfiCustomModel()
        
        self.rapfi_core(np.zeros((1,15,15,2), dtype=np.float32))  # build model
        self.rapfi_core.load_weights('rapfi_model_policy_only_rotate_8.weights.h5')  # load weights

    def int_to_board(self,state1: int, state2: int):
        board = np.zeros((15, 15, 2), dtype=np.float32)
        for i in range(15 * 15):
            y, x = divmod(i, 15)
            if (state1 >> i) & 1:
                board[y, x, 0] = 1  # player 1
            if (state2 >> i) & 1:
                board[y, x, 1] = 1  # player 2
        return board  # shape (15, 15, 2)


    def action(self, state1: int, state2: int):
        board_state = self.int_to_board(state1, state2)  # (15,15,2)
        board_tensor = torch.tensor(board_state).float().unsqueeze(0)  # (1,  15, 15, 2) chuẩn cho keras 

        
        output_tensor = self.rapfi_core(board_tensor)  # (1,255)
        output_tensor = output_tensor['policy']
        # loại bỏ chiều cuối cùng
        output_tensor = output_tensor.squeeze(0)  # (225)
        # chuyển đổi về dạng (15, 15)
        output_tensor = output_tensor.view(15, 15)  # (15,
        
        # Mask hợp lệ
        occupied = board_state[..., 0] + board_state[..., 1]  # (15,15)
        masked_output = output_tensor.clone().squeeze()  # (1, 15, 15)
        # Đặt giá trị của ô đã đánh là -inf để không chọn lại
        masked_output[occupied == 1] = float('-inf') 

        flat_probs = masked_output.view(-1)
        final_policy = flat_probs.view(15, 15)

        # lấy theo lớn nhất
        move_index = torch.argmax(final_policy).item()

        row, col = divmod(move_index, 15)

        # thực hiện đánh
        self.board.set_cell(col, row,1, -1)  # AI đánh là -1

        return ( col, row)

    
        




