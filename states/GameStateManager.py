from states.Menu import *
from states.HumanHumanState import *
from states.VsAIState import *
from setting import *
from setting import _state_factory  # import trực tiếp để dùng biến bắt đầu bằng _
import globals


class GameStateManager:
    def __init__(self):
        self.curState = None
        self.events = None

        # DANH SÁCH CÁC STATES
        self.states = {}

    def handle_events(self,events):
        #Maganer sẽ giữ event
        self.events = events

    def change_state(self, nextState : GameStates):
        if nextState not in self.states:
            # chuyển state thì cho events về trống
            self.events = []
            self.states[nextState] = _state_factory[nextState]()

        self.curState = self.states[nextState]
        self.curState.manager = self
    
    def update(self):
        self.curState.update()
    
    def render(self):
        self.curState.render()
