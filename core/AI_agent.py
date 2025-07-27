from core.Rapfi_core import *
from core.Board import *


class AI_agent:
    def __init__(self, board  : Board, color : str):
        self.board = board
        self.pieceColor = color
        self.rapfi_core = MyModel()
        if type(self.rapfi_core) is MyModel:
            self.rapfi_core.load_state_dict(torch.load('model_3.pth', map_location=torch.device('cpu')))
            self.rapfi_core.eval()  # Set model to evaluation mode
            

    def int_to_board(self,state1: int, state2: int):
        board = np.zeros((15, 15, 2), dtype=np.float32)
        for i in range(15 * 15):
            y, x = divmod(i, 15)
            if (state1 >> i) & 1:
                board[y, x, 0] = 1  # player 1
            if (state2 >> i) & 1:
                board[y, x, 1] = 1  # player 2
        return board  # shape (15, 15, 2)

    def extract_patterns(self,board):  # board: (15, 15, 2)
        H, W, C = board.shape
        D = 4  # 4 hướng: ngang, dọc, chéo /, chéo \
        P = 11
        result = np.zeros((1, H, W, C, D, P), dtype=np.float32)
        
        directions = [  # dy, dx
            (0, 1),   # ngang →
            (1, 0),   # dọc ↓
            (1, 1),   # chéo ↘
            (-1, 1)   # chéo ↗
        ]

        for y in range(H):
            for x in range(W):
                for d, (dy, dx) in enumerate(directions):
                    for p in range(P):
                        ny = y + (p - P // 2) * dy
                        nx = x + (p - P // 2) * dx
                        if 0 <= ny < H and 0 <= nx < W:
                            result[0, y, x, :, d, p] = board[ny, nx]
        return result  # shape: (1, 15, 15, 2, 4, 11)

    def action(self, state1: int, state2: int):
        board_state = self.int_to_board(state1, state2)  # (15,15,2)
        # board_state = np.pad(board_state, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)  # Đảm bảo kích thước là (15, 15, 3)
        # board_state[...,2] = board_state[..., 0] + board_state[..., 1]  # Thêm chiều thứ 3 là tổng của hai người chơi
        board_tensor = torch.tensor(board_state).float().unsqueeze(0)  # (1, 15, 15, 3)

        with torch.no_grad():
            output_tensor = self.rapfi_core.forward(board_tensor)  # (1,15,15,1)
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

        # lấy theo xác suất có trọng số
        # move_index = torch.multinomial(flat_probs, 1).item()

        # lấy theo lớn nhất
        move_index = torch.argmax(final_policy).item()

        row, col = divmod(move_index, 15)

        # thực hiện đánh
        self.board.set_cell(col, row,1, -1)  # AI đánh là -1

        return ( col, row)

    
        




