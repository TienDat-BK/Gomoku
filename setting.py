import os
from enum import Enum, auto

SCREEN_WIDTH = 700
SCREEN_HEIGHT = 700



BOARD_SIZE = min(SCREEN_HEIGHT,SCREEN_WIDTH)

VALID_RADIUS = 460/14/3 * (BOARD_SIZE / 500)
PIECE_SIZE = 460/14 / 1.5 * (BOARD_SIZE / 500)

# thiết lập thôn tin về button có dạng
# (  )

button_menu = {'image' : os.path.join(os.getcwd(),'assets/Blue-button.png'),
               'width' : 960 / 4.25,
               'height' : 242 / 4.25,
               'font' : None,
               'text_size' : 36,
               'text_color' : (51, 51, 51)}

board_image_path = os.path.join(os.getcwd(),'assets/board.jpg')

class GameStates(Enum):
    MenuState = auto()
    Human_VS_Human = auto()
    VsAIState = auto()
    GameOver = auto()

_state_factory = {}

def declare_newState(newState):
    def decorator(State):
        _state_factory[newState] = State
        return State
    return decorator

def lay_vitri_quanco(pos):
    # pos là tuple (x, y)
    x, y = pos
    xx = BOARD_SIZE * (23/350 * x + 1/25) 
    yy = BOARD_SIZE * (23/350 * y + 1/25)
    return (xx, yy)

def Lay_o_quanco(pos):
    # pos là tuple (x, y)
    x, y = pos
    xx = (x / BOARD_SIZE) * (350/23) + 9/23 - 1
    yy = (y / BOARD_SIZE) * (350/23) + 9/23 - 1
    return (xx, yy)

