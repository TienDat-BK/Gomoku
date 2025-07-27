import pygame
import os
import sys
from states.BaseState import *
from Gui.piece import *
from Gui.button import *
from setting import *
from display import *
from core.Board import *
from core.Human import *
from core.AI_agent import AI_agent
import globals
import time

# phải có declare_newState
@ declare_newState(GameStates.VsAIState)
class VsAIState(BaseState):
    def __init__(self):
        super().__init__()

        self.board_image = pygame.image.load(board_image_path).convert_alpha()
        self.board_image = pygame.transform.smoothscale(self.board_image, (min(SCREEN_HEIGHT,SCREEN_WIDTH), min(SCREEN_HEIGHT,SCREEN_WIDTH)))

        self.piece_UI = Piece_UI('white')

        self.board = Board()

        self.Player1 = Human('white',self.board)
        self.Player2 = AI_agent(self.board, 'black')

        self.turn = 1

        

    def Player_action(self):
        # Player1 là người chơi, Player2 là AI
        # Người chơi sẽ đánh trước
        # state1 là state của người chơi, state2 là state của AI
        # sẽ train theo góc nhin của người chơi
        # nếu người chơi đánh thành công thì sẽ train AI
        if self.turn == 1:
            if self.Player1.action(self.mouse_pos_o,self.manager.events) == True:
                self.turn = 2
                # self.Player2.train(self.board.state1, self.board.state2, self.mouse_pos_o)
                # Turn swap
                self.piece_UI.set_color(self.Player2.pieceColor)
        elif self.Player2.action(self.board.state2, self.board.state1):
            # Turn swap
            self.turn = 1
            
            self.piece_UI.set_color(self.Player1.pieceColor)
        

    def check_win(self):
        self.render()
        pygame.display.flip()
        if self.board.check_win(1) or self.board.check_win(-1):
            if self.board.check_win(1):
                time.sleep(1)
                globals.WhoWin = 'white'
                self.manager.change_state(GameStates.GameOver)
            else:
                time.sleep(1)
                globals.WhoWin = 'black'
                self.manager.change_state(GameStates.GameOver)
            self.reset()
            

    def update(self):
        globals.LastState = GameStates.VsAIState

        self.piece_UI.handle_events(self.manager.events)
        self.piece_UI.update()
        self.mouse_pos_o = self.piece_UI.get_nearest_pos_o()

        self.Player_action()

        self.check_win()
        
    def render(self):
        screen.blit(self.board_image, (0,0))
        self.piece_UI.render()
        self.board.render()
    
    def reset(self):
        self.board.reset()
        self.turn = 1
        self.piece_UI.set_color('white')
        self.Player1 = Human('white',self.board)
        self.Player2 = AI_agent(self.board, 'black')