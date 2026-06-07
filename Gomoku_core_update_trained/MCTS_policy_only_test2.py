# Hoặc nếu dùng Torch Backend
import torch
if torch.cuda.is_available():
    torch.cuda.is_available = lambda : False
import numpy as np
from rapfi_core_tf import RapfiCustomModel
from scipy.signal import convolve2d
import numpy as np
import numpy as np
from scipy.signal import convolve2d

# --- HELPER FUNCTIONS ---
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class MCTS_Policy_only():
    def __init__(self,):
        self.model = RapfiCustomModel()
        # Dummy run để init model
        dummy_input = np.zeros((1, 15, 15, 2), dtype=np.float32)
        _ = self.model(dummy_input, training=False)
        self.model.load_weights("model_saved/rapfi_model_policy_only_rotate_8.weights.h5")
        
        self.kernels = [
            np.ones((1, 5)), np.ones((5, 1)), np.eye(5), np.fliplr(np.eye(5))
        ]

    def return_terminal(self, board):
        # board: (15, 15, 2)
        for k in self.kernels:
            if (convolve2d(board[:, :, 0], k, mode='valid') == 5).any():
                return 1
            if (convolve2d(board[:, :, 1], k, mode='valid') == 5).any():
                return -1
        return 0

    # --- SMART ROLLOUT: Check thắng ngay lập tức ---
    def check_immediate_win(self, state):
        # Kiểm tra xem có nước đi nào giúp state hiện tại (người chơi channel 0) thắng ngay không
        # Đây là heuristic giúp rollout khôn hơn nhiều
        # Quét đơn giản: Nếu có 4 con thẳng hàng -> Đánh vào ô còn lại -> Thắng
        # (Ở đây viết demo đơn giản, thực tế nên dùng conv2d để tìm pattern 4)
        return None 

    def rollout(self, node, depth_limit=14):
        cur_state = node.state.copy()
        current_rollout_player = 1 # 1: Người chơi tại node lá, -1: Đối thủ
        
        # Check ngay tại lá
        val = self.return_terminal(cur_state)
        if val != 0: return val

        for depth in range(depth_limit):
            # 1. Prediction & Masking
            policy_dict = self.model(np.expand_dims(cur_state, axis=0), training=False)
            policy_logits = policy_dict['policy'].detach().cpu().numpy().flatten()
            
            mask = cur_state[:,:,0].flatten() + cur_state[:,:,1].flatten()
            policy_logits[mask == 1] = -1e9 # Masking
            
            # 2. Check xem có hết nước đi không
            if np.all(mask == 1): return 0 

            # 3. Action Selection (Softmax Sampling)
            probs = softmax(policy_logits)
            action = np.random.choice(225, p=probs)
            
            # 4. Update State
            x, y = divmod(action, 15)
            cur_state[x, y, 0] = 1 
            
            # 5. Check Terminal
            term = self.return_terminal(cur_state)
            if term != 0:
                # term trả về 1 (người vừa đánh thắng) hoặc -1 (người kia thắng - vô lý vì vừa đánh xong)
                # Ta cần trả về: 1 nếu người bắt đầu rollout thắng, -1 nếu thua
                return 1 * current_rollout_player

            # 6. Swap
            cur_state = np.flip(cur_state, axis=2)
            current_rollout_player *= -1
            
        return 0

    def slection(self, node):
        current_node = node
        while not current_node.is_leaf:
            if self.return_terminal(current_node.state) != 0:
                return current_node
            current_node = current_node.go_down()
        
        # Chỉ expand nếu chưa kết thúc
        if self.return_terminal(current_node.state) == 0:
            current_node.expand()
            current_node.is_leaf = False
        return current_node

    def backpropagate(self, node, value):
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            current_node.value_sum += value # Dùng biến value_sum cho rõ nghĩa
            value *= -1 
            current_node = current_node.parent

    def simulate(self, root, simulations_number=800):
        for i in range(simulations_number):
            # print(f"Simulation {i+1}")
            leaf_node = self.slection(root)
            value = self.rollout(leaf_node)
            self.backpropagate(leaf_node, value)
    
    def update_root(self, root, policy):
        # Chọn nước đi có visits cao nhất (Robust nhất)
        action = int(np.argmax(policy)) 
        print(f"AI Move: {divmod(action, 15)} | Visits Dist: {policy[action]:.4f}")
        
        x, y = divmod(action, 15)
        new_state = root.state.copy()
        new_state[x, y, 0] = 1
        new_state = np.flip(new_state, axis=2)
        return Node(new_state, MCTS=self) # Cắt cây cũ, tạo gốc mới

    def test(self):
        # Case: Mình sắp thắng (3 con liên tiếp, channel 0)
        # Bàn cờ đang là lượt của MÌNH (channel 0)
        root_state = np.zeros((15, 15, 2), dtype=np.float32)
        # cho TH thế thủ đôi chéo
        root_state[8, 8, 0] = 1
        root_state[10, 7, 0] = 1
        root_state[7, 5, 0] = 1
        root_state[8, 6, 1] = 1
        root_state[8 , 7, 1] = 1
        root_state[9 , 7, 1] = 1
        root_state[9 ,8, 1] = 1
        root_state[10 ,8, 1] = 1
        
        root = Node(root_state, MCTS=self)
        
        # Tăng số simulation lên! 20 là quá ít để tìm đường thắng.
        self.simulate(root, simulations_number=300) 
        
        policy = root.get_policy() # Policy này là normalized visits
        
        print("--- Kết quả ---")
        best_action = np.argmax(policy)
        print(f"Best Action (by visits): {divmod(int(best_action), 15)}")
        print(f"Visit Ratio: {policy[best_action]:.4f}")
        
        # In Q-value của nước đi tốt nhất
        best_child = root.children.get(best_action)
        if best_child:
            print(f"Q-value (Win Rate): {-best_child.q_value:.4f}") # Dấu âm vì view từ cha

    def run(self):
        root_state = np.zeros((15, 15, 2), dtype=np.float32)
        states_buffer = []
        coords_buffer = []

        # mô phỏng 5 game
        for _ in range(1):
            root = Node(root_state, MCTS=self)

            while self.return_terminal(root.state) == 0:
                self.simulate(root, simulations_number=100) 

                policy = root.get_policy() # Policy này là normalized visits
                print("--- Kết quả ---")
                best_action = np.argmax(policy)
                print(f"Best Action (by visits): {divmod(int(best_action), 15)}")
                print(f"Visit Ratio: {policy[best_action]:.4f}")
                # In Q-value của nước đi tốt nhất
                best_child = root.children.get(best_action)
                if best_child:
                    print(f"Q-value (Win Rate): {-best_child.q_value:.4f}") # Dấu âm vì view từ cha
                
                # lưu vào buffer
                states_buffer.append(root.state)
                coords_buffer.append(policy)

                root = self.update_root(root, policy)
                self.print_state(root.state)
        
        # save buffer
        states_buffer = np.stack(states_buffer, axis=0)
        coords_buffer = np.stack(coords_buffer, axis=0)

        np.savez("data_for_train/data_buffer_MCTS.npz",
                 states_buffer = states_buffer,
                 coords_buffer = coords_buffer)
            
    def print_state(self, state):
        state_ = state[:, :, 0] - state[:,:,1]
        print(state_)


class Node():
    def __init__(self, state, parent=None, MCTS=None):
        self.MCTS = MCTS
        self.state = state.copy()
        self.parent = parent
        self.children = dict()
        
        self.visits = 0 # ### FIX: Phải là 0
        self.value_sum = 0.0 # ### FIX: Đổi tên để tránh nhầm lẫn
        self.is_leaf = True
        
        self.policy = []
        self.candidate_actions = []

    @property
    def q_value(self):
        if self.visits == 0: return 0
        return self.value_sum / self.visits

    def expand(self):
        if self.MCTS is None: raise ValueError("No MCTS")
        
        res = self.MCTS.model(np.expand_dims(self.state, axis=0), training=False)
        logits = res['policy'].detach().cpu().numpy().flatten()
        
        # Masking
        mask = self.state[:,:,0].flatten() + self.state[:,:,1].flatten()
        logits[mask == 1] = -1e9
        # có thêm temperature-------------------------------------
        T = 10
        logits = logits / T
        # Softmax 1 lần duy nhất tại đây
        self.policy = softmax(logits)
        
        # Lấy Top 20 actions có xác suất cao nhất
        self.candidate_actions = np.argpartition(self.policy, -20)[-20:]

    def expand_child_node(self, action):
        x, y = divmod(action, 15)
        new_state = self.state.copy()
        new_state[x, y, 0] = 1 
        new_state = np.flip(new_state, axis=2)
        self.children[action] = Node(new_state, parent=self, MCTS=self.MCTS)

    def calculate_uct_at_child_node(self, action, c=1.4):
        child = self.children.get(action)
        
        n_child = child.visits if child else 0
        # ### FIX: Dấu trừ cho Zero-sum. 
        # Lấy value_sum/visits chính là Q.
        q_child = -child.q_value if child else 0 
        
        if self.policy is None:
            return 0
        p_action = self.policy[action]
        
        # ### FIX: Công thức PUCT chuẩn (visits = N_parent)
        u_score = c * p_action * (np.sqrt(self.visits) / (1 + n_child))
        
        return q_child + u_score
    
    def go_down(self):
        best_score = -float('inf')
        best_action = None
        
        for action in self.candidate_actions:
            score = self.calculate_uct_at_child_node(action)
            pos = divmod(action, 15)
            # print(f"pos: {pos} - policy: {self.policy[action]} - score: {score}")
            if score > best_score:
                best_score = score
                best_action = action

        if best_action is not None:
            if best_action not in self.children:
                self.expand_child_node(best_action)
            return self.children[best_action]
        
        return self # Fallback nếu không có action nào

    def get_policy(self):
        # Trả về phân phối xác suất dựa trên số lần thăm (Visits)
        policy = np.zeros(225)
        for action, child in self.children.items():
            policy[action] = child.visits
        
        # s = np.sum(policy)
        # if s > 0: policy /= s
        policy = softmax(policy)

        return policy
    

MCTS = MCTS_Policy_only()
MCTS.run()