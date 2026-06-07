
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import sys

# Add parent directory to path to find ai_agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai_agent import GomokuAI

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Initialize App
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('frontend/index.html')

# Global AI Agent
agent = None

@app.on_event("startup")
async def startup_event():
    global agent
    # Path to weights - Adjust relative path as needed
    # We are in e:\Python\Gomoku_core_update\backend
    # Weights are in e:\Python\Gomoku_core_update\model_saved\...
    weights_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model_saved", 
        "rapfi_model_policy_only_rotate_8.weights.h5"
    )
    if not os.path.exists(weights_path):
        print(f"WARNING: Weights not found at {weights_path}")
    
    try:
        agent = GomokuAI(weights_path)
        print("AI Agent initialized.")
    except Exception as e:
        print(f"Error initializing AI Agent: {e}")
        # Dont crash, maybe user can fix path
        pass

class BoardState(BaseModel):
    board: List[List[int]] # 15x15 2D array. 0=Empty, 1=User(Black), -1=AI(White) (or 2)
    # Actually, let's agree on protocol.
    # Frontend usually uses 1 for Player 1, 2 for Player 2.
    # We will accept that and map it.

@app.post("/predict")
async def get_move(data: BoardState):
    global agent
    if agent is None:
        raise HTTPException(status_code=500, detail="AI Agent not initialized")
    
    try:
        # Frontend sends: 1=Black (User), 2=White (AI) usually.
        # Our AI expects to play as "White" usually if User moves first.
        # But `ai_agent.py` preprocess expects: board has `ai_role` value for AI's stones.
        # If Frontend sends 1 for User, -1 for AI.
        # Let's standardize: Frontend sends 1 for User, 2 for AI.
        # We need to map 2 -> -1 if our model uses -1, OR just tell `ai_agent` that `ai_role=2`.
        # Taking a look at `MCTS_policy_only.py`, it used logic `state[:, :, 0] - state[:, :, 1]`.
        # `plot_board_console` logic: `board[r, c] == 1` is X, `-1` is O.
        # So internally it seems safe to just pass the raw numbers if we tell `ai_agent` who is who.
        
        # Let's assume Frontend sends: 0=Empty, 1=User, 2=AI.
        move = agent.get_move(data.board, ai_role=2)
        return {"x": move[0], "y": move[1]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Gomoku AI Backend is running"}
