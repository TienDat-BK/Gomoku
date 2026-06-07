import numpy as np
from scipy.signal import convolve2d
from itertools import product
import re

class FastHeuristic7:
    def __init__(self):
        self.SCORES = {}
        
        # Kernel lũy thừa cơ số 3 cho 7 ô: [1, 3, 9, 27, 81, 243, 729]
        self.kernel_val = np.power(3, np.arange(7))
        self.kernel_h = self.kernel_val.reshape(1, 7)       # Ngang
        self.kernel_v = self.kernel_h.T                     # Dọc
        self.kernel_d1 = np.diag(self.kernel_val)           # Chéo chính
        self.kernel_d2 = np.fliplr(self.kernel_d1)          # Chéo phụ

        # --- ĐĂNG KÝ CÁC MẪU (Pattern) ---
        # 1. WIN 5
        self.register_pattern("11111", 100000) 

        # 2. LIVE 4 (4 Sống - Bắt buộc 2 đầu là 0)
        self.register_pattern("011110", 10000)
        
        # 3. LIVE 4 Lủng (Broken Live 4)
        self.register_pattern("0111010", 9000)
        self.register_pattern("0101110", 9000)
        self.register_pattern("0110110", 9000)

        # 4. DEAD 4 (4 Chết - Bị chặn bởi 2)
        self.register_pattern("211110", 2500)
        self.register_pattern("011112", 2500)
        
        # Dead 4 Lủng
        self.register_pattern("211101", 2000)
        self.register_pattern("101112", 2000)
        self.register_pattern("210111", 2000)
        
        # 5. LIVE 3 (3 Sống)
        self.register_pattern("001110", 3000)
        self.register_pattern("011100", 3000)
        self.register_pattern("010110", 2800)
        self.register_pattern("011010", 2800)

    def register_pattern(self, core_pat, score):
        """Tự động sinh các biến thể lấp đầy cửa sổ 7 ô"""
        len_p = len(core_pat)
        needed = 7 - len_p
        if needed < 0: return

        # Duyệt qua vị trí bắt đầu của core_pat trong cửa sổ 7 ô
        for start_idx in range(needed + 1):
            prefix_len = start_idx
            suffix_len = 7 - len_p - start_idx
            
            # 0: Trống, 1: Ta, 2: Địch/Tường
            prefixes = list(product('012', repeat=prefix_len))
            suffixes = list(product('012', repeat=suffix_len))
            
            for p in prefixes:
                for s in suffixes:
                    full_str = "".join(p) + core_pat + "".join(s)
                    self.add_id(full_str, score)

    def add_id(self, s, score):
        pid = 0
        for i, c in enumerate(s):
            pid += int(c) * (3**i)
        # Lưu điểm cao nhất cho ID đó
        if pid not in self.SCORES:
            self.SCORES[pid] = score
        else:
            self.SCORES[pid] = max(self.SCORES[pid], score)

    def calculate_raw_score(self, board_padded):
        """Hàm tính điểm nội bộ"""
        # Convolution
        conv_h = convolve2d(board_padded, self.kernel_h, mode='valid')
        conv_v = convolve2d(board_padded, self.kernel_v, mode='valid')
        conv_d1 = convolve2d(board_padded, self.kernel_d1, mode='valid')
        conv_d2 = convolve2d(board_padded, self.kernel_d2, mode='valid')
        
        # Lookup điểm từ dictionary
        total = 0
        # Flatten ra để list comprehension nhanh hơn
        # Mẹo: Dùng .get(x, 0)
        total += sum([self.SCORES.get(x, 0) for x in conv_h.flatten()])
        total += sum([self.SCORES.get(x, 0) for x in conv_v.flatten()])
        total += sum([self.SCORES.get(x, 0) for x in conv_d1.flatten()])
        total += sum([self.SCORES.get(x, 0) for x in conv_d2.flatten()])
        
        return total

    def evaluate(self, state):
        """
        Input: state (15, 15, 2) - Channel 0: Ta, Channel 1: Địch
        Output: [P_Win, P_Draw, P_Loss]
        """
        pad_width = 3 # (7-1)/2
        
        # --- 1. TÍNH ĐIỂM CHO TA (Player 1) ---
        # Ta = 1, Địch = 2, Trống = 0
        board_my = state[:, :, 0] * 1 + state[:, :, 1] * 2
        # Padding viền bằng 2 (Tường/Địch)
        board_my_padded = np.pad(board_my, pad_width, mode='constant', constant_values=2)
        
        my_score = self.calculate_raw_score(board_my_padded)
        
        # --- 2. TÍNH ĐIỂM CHO ĐỊCH (Player 2) ---
        # Đảo vai trò: Địch (channel 1) thành Ta (1), Ta (channel 0) thành Địch (2)
        board_op = state[:, :, 1] * 1 + state[:, :, 0] * 2
        # Padding vẫn bằng 2 (Tường luôn là vật cản)
        board_op_padded = np.pad(board_op, pad_width, mode='constant', constant_values=2)
        
        op_score = self.calculate_raw_score(board_op_padded)

        # --- 3. QUY ĐỔI RA XÁC SUẤT ---
        # Logic cứng: Thắng ngay lập tức
        if my_score >= 100000: return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if op_score >= 100000: return np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Logic mềm (Softmax)
        # Giảm draw_bias xuống 1.0 để AI bớt đoán hòa
        draw_bias = 1.0 
        scale = 0.0005 

        logit_win = my_score * scale
        logit_loss = op_score * scale
        logit_draw = draw_bias
        
        logits = np.array([logit_win, logit_draw, logit_loss])
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum()
    
    class PatternEvaluator:
        def __init__(self):
            # ĐỊNH NGHĨA HỆ SỐ ĐIỂM (Càng nguy hiểm điểm càng cao)
            self.SCORES = {
                'WIN_5': 100000,   # 5 con liên tiếp (Thắng ngay)
                'LIVE_4': 10000,   # 4 con hở 2 đầu (Chắc chắn thắng lượt sau)
                'DEAD_4': 1000,    # 4 con bị chặn 1 đầu (Buộc đối thủ phải chặn)
                'LIVE_3': 1000,    # 3 con hở 2 đầu (Tạo ra Live 4)
                'DEAD_3': 100,     # 3 con bị chặn 1 đầu
                'LIVE_2': 10,      # 2 con hở 2 đầu
            }

        def get_lines(self, board_2d):
            """
            Chuyển bàn cờ thành các chuỗi (String) theo 4 hướng:
            Ngang, Dọc, Chéo Chính, Chéo Phụ.
            Quy ước: 
            - Quân mình (P): 'X'
            - Quân địch (O): 'O'
            - Ô trống   (.): '.'
            """
            lines = []
            rows, cols = board_2d.shape
            
            # 1. Ngang (Rows)
            for r in range(rows):
                lines.append("".join(board_2d[r, :]))

            # 2. Dọc (Cols)
            for c in range(cols):
                lines.append("".join(board_2d[:, c]))

            # 3. Chéo chính (Diagonals)
            # Đường chéo chính offset từ -rows+1 đến cols-1
            for k in range(-rows + 1, cols):
                diag = np.diagonal(board_2d, offset=k)
                if len(diag) >= 5: # Chỉ lấy đường chéo dài >= 5
                    lines.append("".join(diag))

            # 4. Chéo phụ (Anti-diagonals)
            # Lật ngược bàn cờ rồi lấy chéo chính
            flipped_board = np.fliplr(board_2d)
            for k in range(-rows + 1, cols):
                diag = np.diagonal(flipped_board, offset=k)
                if len(diag) >= 5:
                    lines.append("".join(diag))
                    
            return lines

        def count_patterns(self, lines, player_char='X'):
            """
            Đếm số lượng các mẫu cờ của người chơi (player_char)
            """
            score = 0
            
            # Ký tự đối thủ
            opp_char = 'O' if player_char == 'X' else 'X'
            
            # Regex Patterns
            # (?=...) là lookahead để bắt các pattern chồng lên nhau (vd: XXXXX bắt được 5)
            
            # 1. WIN 5 (XXXXX)
            p_win5 = f"{player_char}{{5}}" 
            
            # 2. LIVE 4 (.XXXX.) -> Cực mạnh
            p_live4 = rf"\.{player_char}{{4}}\."
            
            # 3. DEAD 4 (OXXXX. hoặc .XXXXO hoặc X.XXX hoặc XX.XX)
            # Bị chặn 1 đầu hoặc bị lủng ở giữa
            p_dead4_a = rf"{opp_char}{player_char}{{4}}\." # OXXXX.
            p_dead4_b = rf"\.{player_char}{{4}}{opp_char}" # .XXXXO
            p_dead4_c = rf"{player_char}{{1}}\.{player_char}{{3}}" # X.XXX (Lủng)
            p_dead4_d = rf"{player_char}{{3}}\.{player_char}{{1}}" # XXX.X (Lủng)
            p_dead4_e = rf"{player_char}{{2}}\.{player_char}{{2}}" # XX.XX (Lủng)

            # 4. LIVE 3 (.XXX. hoặc .X.XX. hoặc .XX.X.)
            p_live3_a = rf"\.{player_char}{{3}}\." # .XXX.
            p_live3_b = rf"\.{player_char}{{1}}\.{player_char}{{2}}\." # .X.XX.
            p_live3_c = rf"\.{player_char}{{2}}\.{player_char}{{1}}\." # .XX.X.

            # Quét qua tất cả các dòng
            full_text = "||".join(lines) # Nối lại thành 1 string lớn để regex 1 lần (nhanh hơn)

            # --- TÍNH ĐIỂM ---
            
            # Win 5
            if len(re.findall(p_win5, full_text)) > 0: 
                return self.SCORES['WIN_5']
                
            # Live 4
            count = len(re.findall(p_live4, full_text))
            score += count * self.SCORES['LIVE_4']
            
            # Dead 4 (Tổng hợp các loại 4 bị chặn/lủng)
            count = len(re.findall(p_dead4_a, full_text)) + \
                    len(re.findall(p_dead4_b, full_text)) + \
                    len(re.findall(p_dead4_c, full_text)) + \
                    len(re.findall(p_dead4_d, full_text)) + \
                    len(re.findall(p_dead4_e, full_text))
            score += count * self.SCORES['DEAD_4']
            
            # Live 3
            count = len(re.findall(p_live3_a, full_text)) + \
                    len(re.findall(p_live3_b, full_text)) + \
                    len(re.findall(p_live3_c, full_text))
            score += count * self.SCORES['LIVE_3']

            # Dead 3 (Đơn giản hóa: Bị chặn 1 đầu)
            # Logic: Tìm pattern XXX nhưng không phải Live 3 -> Khả năng cao là Dead 3
            # Phần này làm kỹ sẽ rất dài, tạm thời bỏ qua để tối ưu tốc độ, 
            # vì Live 3 và Dead 4 mới là quan trọng nhất.
            
            return score

        def evaluate(self, state_3d):
            """
            Input: State (15, 15, 2)
            Output: Softmax Probabilities [Win, Draw, Loss]
            """
            # 1. Chuyển State thành mảng ký tự
            # Channel 0: Ta (1), Channel 1: Địch (1)
            # -> Ta map thành: Ta='X', Địch='O', Trống='.'
            
            board_char = np.full((15, 15), '.', dtype='<U1')
            
            # Lấy mask
            p1_mask = state_3d[:, :, 0] == 1
            p2_mask = state_3d[:, :, 1] == 1
            
            board_char[p1_mask] = 'X'
            board_char[p2_mask] = 'O'
            
            # 2. Lấy các đường (lines)
            lines = self.get_lines(board_char)
            
            # 3. Tính điểm cho TA và ĐỊCH
            my_score = self.count_patterns(lines, 'X')
            op_score = self.count_patterns(lines, 'O')
            
            # 4. Logic phân định Thắng/Thua
            # Nếu Ta có 5 -> Thắng tuyệt đối
            if my_score >= self.SCORES['WIN_5']:
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)
            
            # Nếu Địch có 5 -> Thua tuyệt đối
            if op_score >= self.SCORES['WIN_5']:
                return np.array([0.0, 0.0, 1.0], dtype=np.float32)
                
            # 5. Chuyển điểm thành Logits
            # Scale điểm xuống để softmax mượt hơn
            # Mẹo: Điểm Live4 là 10000 -> Logit tầm 10.0 là đẹp
            scale = 0.001 
            
            draw_bias = 2.0 # Bias cho cửa hòa
            
            logit_win = my_score * scale
            logit_loss = op_score * scale
            logit_draw = draw_bias
            
            # Softmax
            logits = np.array([logit_win, logit_draw, logit_loss])
            e_x = np.exp(logits - np.max(logits))
            return e_x / e_x.sum()