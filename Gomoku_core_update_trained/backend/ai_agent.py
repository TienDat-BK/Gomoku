
import sys
import os
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Chọn 1 trong 3: "tensorflow", "torch", hoặc "jax"
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import keras
from keras import ops
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rapfi_core_tf import RapfiCustomModel

class GomokuAI:
    def __init__(self, weights_path):
        self.model = RapfiCustomModel(
            in_channels=2,
            mid_channels_for_directionBlock=32,
            out_channels_for_directionBlock=16,
            kernel_size=3
        )
        # Dummy build
        dummy_input = np.zeros((1, 15, 15, 2), dtype=np.float32)
        _ = self.model(dummy_input, training=False)
        
        print(f"Loading weights from {weights_path}")
        self.model.load_weights(weights_path)
        print("Model loaded successfully")

    def preprocess_state(self, board, ai_role):
        """
        board: 15x15 list or numpy array. 0=Empty, 1=Black, -1=White (or 2=White)
        ai_role: The value representing AI's stones on the board.
        
        Returns: (1, 15, 15, 2)
        Channel 0: AI's stones (1 if AI has stone there, else 0)
        Channel 1: Opponent's stones (1 if Opponent has stone there, else 0)
        """
        board = np.array(board)
        state = np.zeros((15, 15, 2), dtype=np.float32)
        
        # Channel 0: Own stones
        state[:, :, 0] = (board == ai_role).astype(np.float32)
        
        # Channel 1: Opponent stones
        # Assuming opponent is whatever is not 0 and not ai_role
        # Common convention: Black=1, White=-1 or 2.
        # Let's assume input board uses: 1 for P1 (Black), -1 for P2 (White) as per MCTS code in rapfi_core_tf.py context (implied)
        # Re-reading MCTS code: "board = state[:, :, 0] - state[:, :, 1]" -> 0 is Ta, 1 is Dich.
        # So we just map logical board to this.
        
        opponent_mask = (board != 0) & (board != ai_role)
        state[:, :, 1] = opponent_mask.astype(np.float32)
        
        return np.expand_dims(state, axis=0)

    def get_move(self, board, ai_role=-1):
        """
        AI plays 'ai_role' (default -1 for White/Player2).
        Returns: (x, y) coordinates of the best move.
        """
        # 1. Preprocess
        input_state = self.preprocess_state(board, ai_role)
        
        # 2. Predict
        output = self.model.predict(input_state, verbose=0)
        policy = output['policy'][0] # Shape (225,)
        
        # 3. Mask invalid moves
        board_flat = np.array(board).flatten()
        policy[board_flat != 0] = -1e9
        
        # 4. Argmax
        best_action = np.argmax(policy)
        x, y = divmod(best_action, 15)
        
        return int(x), int(y)

if __name__ == "__main__":
    # Test
    agent = GomokuAI("../model_saved/rapfi_model_policy_only_rotate_8.weights.h5")
    empty_board = np.zeros((15, 15))
    # Let Black (1) play center
    empty_board[7, 7] = 1
    
    # AI (White/-1) moves
    move = agent.get_move(empty_board, ai_role=-1)
    print(f"AI suggests: {move}")
