import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Ẩn GPU, ép Keras dùng CPU
import numpy as np
from rapfi_core_tf import RapfiCustomModel
from scipy.signal import convolve2d
import numpy as np

class MCTS_Policy_only():
    def __init__(self,):
        self.model = RapfiCustomModel()
        dummy_input = np.zeros((1, 15, 15, 2), dtype=np.float32)
        _ = self.model(dummy_input, training=False)
        self.model.load_weights("model_saved/rapfi_model_policy_only_rotate_8.weights.h5")
        self.kernels = [
            np.ones((1, 5)), # Ngang
            np.ones((5, 1)), # Dọc
            np.eye(5),       # Chéo chính
            np.fliplr(np.eye(5)) # Chéo phụ
        ]


    def return_terminal(self, board):
        # board shape (15, 15, 2)
        # Định nghĩa 4 bộ lọc tương ứng với 4 hướng thắng (ngang, dọc, 2 chéo)
        # Mỗi bộ lọc là một mảng 5x5 có 5 số 1 nằm trên hướng đó
        # return 1 -> quan mình thắng [0] 
        # return -1 -> quan địch thắng [1] 
        # return 0 -> hòa hoặc chưa kết thúc
        
        
        for k in self.kernels:
            # convolve2d sẽ quét bộ lọc qua bàn cờ. 
            # Nếu ở đâu có 5 quân 1 liên tiếp, giá trị tại đó sẽ là 5.
            if (convolve2d(board[:, :, 0], k, mode='valid') == 5).any():
                return 1
            if (convolve2d(board[:, :, 1], k, mode='valid') == 5).any():
                return -1
        return 0

    def rollout(self, node, depth_limit = 20):
        cur_state = node.state.copy()
        turn = 1

        value = self.return_terminal(cur_state)
        if value != 0 :
            return value

        for depth in range(depth_limit):
            policy = self.model(np.expand_dims(cur_state, axis=0), training=False)
            policy = policy['policy'].detach().cpu().numpy().flatten()

            # masking cac nuoc di khong hop le
            mask = cur_state[:,:,0].flatten() + cur_state[:,:,1].flatten()
            masked_policy = policy + (mask * -1e9)  # loai bo cac nuoc di khong hop le

            if np.sum(masked_policy) == 0:
                return 0 # draw
            
            # lấy theo sác xuât suất policy, policy là mảng 225 rồi
            # softmax
            exp_policy = np.exp(masked_policy - np.max(masked_policy))
            prob_policy = exp_policy / np.sum(exp_policy)
            # action = np.random.choice(225, p=prob_policy)
            action = int(np.argmax(prob_policy))
            x, y = divmod(action, 15)
            
            # update state
            cur_state[x, y, 0] = 1 

            # check terminal
            value = self.return_terminal(cur_state)
            if value != 0:
                # print(f" rollout terminal {cur_state[:, :, 0] - cur_state[:, :, 1]}")
                # print(f"value rollout: {value*turn}")
                return value * turn
            elif depth == depth_limit - 1:
                return 0 # draw

            # swap player
            cur_state = np.flip(cur_state, axis=2)

            turn*= -1
        
    def slection(self, node):
        current_node = node
        while not current_node.is_leaf:
            # nếu nó là terminal thì return luôn
            if self.return_terminal(current_node.state.copy()) != 0:
                return current_node
            current_node = current_node.go_down()
        
        current_node.expand()
        current_node.is_leaf = False
        return current_node

    def backpropagate(self, node, value):
        current_node = node
        current_node.value += value
        value *= -1  # Đổi giá trị cho đối thủ
        current_node = current_node.parent

        while current_node is not None:
            current_node.visits += 1
            current_node.value += value
            value *= -1  # Đổi giá trị cho đối thủ
            current_node = current_node.parent

    def simulate(self, root, simulations_number = 800):
        for _ in range(simulations_number):
            print(f"simulation {_+1}")
            leaf_node = self.slection(root)
            # print(f"Node slected: {leaf_node.state[:,:,0] + -1*leaf_node.state[:,:,1]}")
            value = self.rollout(leaf_node)
            print(f"value of leaf Node: {value}")
            self.backpropagate(leaf_node, value)
            # input()
    
    def update_root(self, root, policy):
        action = int(np.argmax(policy))
        x, y = divmod(action, 15)
        new_state = root.state.copy()
        new_state[x, y, 0] = 1  # Đặt quân của người chơi hiện tại
        new_state = np.flip(new_state, axis=2)  # Đổi lượt cho đối thủ
        new_root = Node(new_state, MCTS=self)
        return new_root

    def test(self):
        root_state = np.zeros((15, 15, 2), dtype=np.float32)

        # cho TH thế thủ đôi chéo
        root_state[8, 8, 1] = 1
        root_state[9, 9, 1] = 1
        root_state[9, 9, 1] = 1
        root_state[8, 12, 1] = 1
        root_state[10 , 8, 0] = 1
        root_state[8 , 9, 0] = 1
        root_state[8 , 10, 0] = 1


        root = Node(root_state, MCTS=self)
        self.simulate(root, simulations_number=20)
        print(f"root.value = {root.value}")
        final_node = root.get_min_Q_child()
        root = self.update_root(root, root.get_policy())

        print("New root state after one move:")
        print(final_node.state[:,:,0] + -1*final_node.state[:,:,1])  # In ra trạng thái mới của bàn cờ
        

class Node():
    def __init__(self, state, parent=None, MCTS=None):
        self.MCTS = MCTS
        
        self.state = state.copy()
        self.parent = parent
        self.children = dict()
        self.visits = 1
        self.value = 0.0
        self.is_leaf = True

    def expand(self):
        if self.MCTS is None:
            raise ValueError("MCTS instance must be provided")
        self.policy = self.MCTS.model(np.expand_dims(self.state, axis=0), training=False)['policy'].detach().cpu().numpy().flatten()
        self.candidate_actions = self.get_candidate_actions(10) # Lấy 20 nước đi tiềm năng nhất, nó là index trong 225

    def expand_child_node(self, action):
        x, y = divmod(action, 15)
        if self.state[x, y, 0] == 0 and self.state[x, y, 1] == 0:
            new_state = self.state.copy()
            new_state[x, y, 0] = 1  # Đặt quân của người chơi hiện tại
            new_state = np.flip(new_state, axis=2)  # Đổi lượt cho đối thủ
            self.children[action] = Node(new_state, parent=self, MCTS=self.MCTS)

    def sofmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    # softmax cho policy luôn
    def get_candidate_actions(self, topK = 10):
        mask = self.state[:,:,0].flatten() + self.state[:,:,1].flatten()
        policy = self.policy + (mask * -1e5)  # loai bo cac nuoc di khong hop le
        T = 10
        policy = policy / T
        policy = self.sofmax(policy)

        candidate_actions = np.argpartition(policy, -topK)[-topK:]
        return candidate_actions

    def calculate_uct_at_child_node(self, action, c=4):
        # đảm bao các child node chưa được mở rộng sẽ luôn được chọn
        if action not in self.children.keys():
            return ( c * self.policy[action] * (np.sqrt(self.visits) ) )
        exploitation = -self.children[action].value / self.children[action].visits
        exploration = c * self.policy[action] * (np.sqrt(self.visits) /(1+ self.children[action].visits))
        return exploitation + exploration
    
    def go_down(self):
        
        
        
        best_score = -float('inf')
        best_action = None
        for action in self.candidate_actions:
            uct_score = self.calculate_uct_at_child_node(action)
            if uct_score > best_score:
                best_score = uct_score
                best_action = action

        if(best_action not in self.children.keys()):
            self.expand_child_node(best_action)
        
        if best_action is not None:
            best_action = int(best_action)
        return self.children[best_action]

    def get_policy(self):
        policy = np.zeros(225)
        for action in self.children.keys():
            policy[action] = self.children[action].visits
        
        policy = policy / sum(policy)
        return policy

    def get_min_Q_child(self):
        minQ = 1e9
        best_action = 0
        for action in self.candidate_actions:
            if(action not in self.children.keys()):
                continue
            if self.children[action].value / self.children[action].visits < minQ:
                best_action = action
                minQ = self.children[action].value / self.children[action].visits

        return self.children[action]


def check_policy(MCTS, state):
    policy = MCTS.model(state)['policy'].detach().cpu().numpy().flatten()
    mask = state.flatten()[:225] + state.flatten()[225:450]
    policy = policy + (mask * -1e9)
    T = 10
    policy = policy / T
    exp_policy = np.exp(policy - np.max(policy))
    policy = exp_policy / np.sum(exp_policy)
    decenst = np.argsort(policy)[::-1]
    for i in range(10):
        pos = divmod(decenst[i], 15)
        print(f"pos: {pos} -> {policy[decenst[i]]}")


MCTS = MCTS_Policy_only()
# MCTS.test()


state = np.zeros((1,15,15,2), dtype=np.float32)
# cho TH thế thủ đôi chéo
state[0 ,8, 8, 1] = 1
state[0 ,9, 9, 1] = 1
state[0 ,9, 9, 1] = 1
state[0 ,8, 12, 1] = 1
state[0 ,10 , 8, 0] = 1
state[0 ,8 , 9, 0] = 1
state[0 ,8 , 10, 0] = 1

check_policy(MCTS, state)

# policy = MCTS.model(state, training = False)['policy']
# policy = policy.detach().cpu().numpy().flatten()
# mask = state[:,:,:,0].flatten() +state[:,:,:,1].flatten()
# policy = policy + (mask * -1e5)
# print( divmod(int(np.argmax(policy)), 15)  )

