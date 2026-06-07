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
import time
def plot_board_console(state):
    # Gộp 2 channel thành 1 board duy nhất: Ta (1), Địch (-1), Trống (0)
    board = state[:, :, 0] - state[:, :, 1]
    
    print("   " + " ".join([f"{i:2}" for i in range(15)])) # In số cột
    print("   " + "-" * 44)
    
    for r in range(15):
        row_str = f"{r:2} |"
        for c in range(15):
            if board[r, c] == 1:
                row_str += " X "  # Quân Ta
            elif board[r, c] == -1:
                row_str += " O "  # Quân Địch
            else:
                row_str += " . "  # Ô trống
        print(row_str + "|")
    print("   " + "-" * 44)

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
        self.maxDepth = 0

    def return_terminal(self, state, pos):
        """
        Logic: Kiểm tra xem người vừa đánh (hiện đang ở channel 1) có thắng không.
        state: (15, 15, 2) sau khi đã flip.
        pos: (x, y) tọa độ nước vừa đánh.
        """
        x, y = pos
        
        # Nước đi vừa rồi nằm ở channel 1 (đối thủ của người hiện tại)
        player = 1 
        
        # 4 hướng: Ngang, Dọc, Chéo chính, Chéo phụ
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1  # Chính là quân vừa đánh
            
            # Kiểm tra hướng tiến
            for i in range(1, 5):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < 15 and 0 <= ny < 15 and state[nx, ny, player] == 1:
                    count += 1
                else:
                    break
                    
            # Kiểm tra hướng lùi
            for i in range(1, 5):
                nx, ny = x - i * dx, y - i * dy
                if 0 <= nx < 15 and 0 <= ny < 15 and state[nx, ny, player] == 1:
                    count += 1
                else:
                    break
            
            # Nếu đủ 5 quân
            if count >= 5:
                # Vì đây là channel 1 (người vừa thực hiện nước đi), 
                # nên giá trị trả về đối với Node hiện tại là -1 (mình bị thua)
                # Hoặc trả về 1 tùy vào cách bạn thiết lập Backpropagation.
                return 1 
            
        return 0

    def slection(self, node):
        current_node = node
        while not current_node.is_leaf:
            # if self.return_terminal(current_node.state, current_node.pos) != 0:
            #     return current_node
            current_node = current_node.go_down()
        
        # Chỉ expand nếu chưa kết thúc
        if self.return_terminal(current_node.state, current_node.pos) == 0:
            current_node.expand()
            current_node.is_leaf = False
        else:
            current_node.value_predict = self.return_terminal(current_node.state, current_node.pos)
        return current_node

    def backpropagate(self, node, value):
        current_node = node
        cnt = 0
        while current_node is not None:
            cnt+=1
            current_node.visits += 1
            
            current_node.value_sum += value # Dùng biến value_sum cho rõ nghĩa
            value *= -1 
            current_node = current_node.parent
        
        self.maxDepth = max(self.maxDepth, cnt)

    def simulate(self, root, simulations_number=800):
        a = time.time()
        for i in range(simulations_number):
            # print(f"Simulation {i+1}")
            leaf_node = self.slection(root)
            value = leaf_node.value_predict
            self.backpropagate(leaf_node, value)
            # root.printDebug()
            # input()
        print(f"time simulation per slection: {(time.time() - a) / simulations_number}")
        
    
    def update_root(self, root, policy):
        # Chọn nước đi có visits cao nhất (Robust nhất)
        action = int(np.argmax(policy)) 
        print(f"AI Move: {divmod(action, 15)} | Visits Dist: {policy[action]:.4f}")
        
        x, y = divmod(action, 15)
        new_state = root.state.copy()
        new_state[x, y, 0] = 1
        new_state = np.flip(new_state, axis=2)
        return Node(new_state,(x, y), MCTS=self) # Cắt cây cũ, tạo gốc mới

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
        
        root = Node(root_state,(8, 8), MCTS=self)
        
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
        root_state[7,7,1] = 1
        pos = (7,7)
        states_buffer = []
        coords_buffer = []

        # mô phỏng 5 game
        for _ in range(1):
            root = Node(root_state,pos, MCTS=self)
            cnt = 1
            while self.return_terminal(root.state, root.pos) == 0:
                self.simulate(root, simulations_number=800) 

                policy = root.get_policy() # Policy này là normalized visits
                print("--- Kết quả ---")
                best_action = np.argmax(policy)
                print(f"Best Action (by visits): {divmod(int(best_action), 15)}")
                print(f"Visit Ratio: {policy[best_action]:.4f}")
                # In Q-value của nước đi tốt nhất
                best_child = root.children.get(best_action)
                if best_child:
                    print(f"Q-value (Win Rate): {-best_child.q_value:.4f}") # Dấu âm vì view từ cha
                print(f"maxDepth: {self.maxDepth}")
                self.maxDepth = 0
                
                # lưu vào buffer
                states_buffer.append(root.state)
                coords_buffer.append(policy)

                root = self.update_root(root, policy)
                if cnt > 0:
                    plot_board_console(np.flip(root.state, axis=2))
                else:
                    plot_board_console(root.state)
                cnt*=-1

        
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
    def __init__(self, state, pos, parent=None, MCTS=None):
        self.MCTS = MCTS
        self.state = state.copy()
        self.parent = parent
        self.children = dict()
        self.pos = pos
        
        self.visits = 0 # ### FIX: Phải là 0
        self.value_sum = 0.0 # ### FIX: Đổi tên để tránh nhầm lẫn
        self.is_leaf = True
        self.value_predict = None
        self.policy = []
        self.candidate_actions = []

        # self.expand()

    @property
    def q_value(self):
        if self.visits == 0: return 0
        return self.value_sum / self.visits

    def expand(self):
        if self.MCTS is None: raise ValueError("No MCTS")
        
        input = torch.from_numpy(
            np.expand_dims(self.state, axis=0).astype(np.float32)
        )

        with torch.no_grad():
            res = self.MCTS.model(input, training=False)

        logits = res['policy'].numpy().flatten()
        value  = res['value'].numpy().flatten()
        self.value_predict = float(value[0] - value[2])
        
        # Masking
        mask = self.state[:,:,0].flatten() + self.state[:,:,1].flatten()
        logits[mask == 1] = -1e9
        # có thêm temperature-------------------------------------
        T = 7
        logits = logits / T
        # Softmax 1 lần duy nhất tại đây
        self.policy = softmax(logits)

        # Lấy Top 30 actions có xác suất cao nhất
        numCandidate = 50
        self.candidate_actions = np.argpartition(self.policy, -numCandidate)[-numCandidate:]

        # nếu là root, thêm nhiễu
        if self.parent is None:
            eps = 0.25
            dir = np.random.dirichlet(0.3 * np.ones(numCandidate))
            for i, idx in enumerate(self.candidate_actions):
                self.policy[idx] = (1 - eps) * self.policy[idx] + eps * dir[i]

        # debug cho root
        if self.parent is None:
            print(f"distribute {numCandidate} Candidate: {self.policy[self.candidate_actions]}")

    def expand_child_node(self, action):
        x, y = divmod(action, 15)
        new_state = self.state.copy()
        new_state[x, y, 0] = 1 
        new_state = np.flip(new_state, axis=2)
        self.children[action] = Node(new_state,(x,y), parent=self, MCTS=self.MCTS)

    def calculate_uct_at_child_node(self, action, c=5):
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
        
        s = np.sum(policy)
        if s > 0: policy /= s
        

        return policy
    
    def printDebug(self):
        """
        In ra danh sách visits, UCT score, Q-value của các node con
        self là Node object
        """
        print("\n" + "="*70)
        print(f"DEBUG NODE: pos={self.pos} | visits={self.visits} | q_value={self.q_value:.4f}")
        print("="*70)
        print(f"{'Action':<10} {'Pos':<12} {'Visits':<10} {'Q-value':<12} {'UCT Score':<12}")
        print("-"*70)
        
        if not self.children:
            print("No children nodes")
        else:
            for action in sorted(self.children.keys()):
                child = self.children[action]
                pos = divmod(action, 15)
                pos = (int(pos[0]), int(pos[1]))
                uct = self.calculate_uct_at_child_node(action)
                
                print(f"{action:<10} {str(pos):<12} {child.visits:<10} {-child.q_value:<12.4f} {uct:<12.4f}")
        
        print("="*70 + "\n")
MCTS = MCTS_Policy_only()
MCTS.run()