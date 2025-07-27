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
from states.GameOverState import *
import time
import globals

# phải có declare_newState
@ declare_newState(GameStates.Human_VS_Human)
class HumanHumanState(BaseState):
    def __init__(self):
        super().__init__()

        self.board_image = pygame.image.load(board_image_path).convert_alpha()
        self.board_image = pygame.transform.smoothscale(self.board_image, (min(SCREEN_HEIGHT,SCREEN_WIDTH), min(SCREEN_HEIGHT,SCREEN_WIDTH)))

        self.piece_UI = Piece_UI('white')

        self.board = Board()

        self.Player1 = Human('white',self.board)
        self.Player2 = Human('black',self.board)

        self.turn = 1
     

    def Player_action(self):
        if self.Player1.action(self.mouse_pos_o,self.manager.events) == True:
            # Turn swap
            self.Player1 , self.Player2 = self.Player2 , self.Player1
            self.piece_UI.set_color(self.Player1.pieceColor)

    def check_win(self):
        self.render()
        pygame.display.flip()
        if self.board.check_win(1):
            self.reset()
            time.sleep(1)
            globals.WhoWin = 'white'
            self.manager.change_state(GameStates.GameOver)
        if self.board.check_win(2):
            self.reset()
            time.sleep(1)
            globals.WhoWin = 'black'
            self.manager.change_state(GameStates.GameOver)

    def update(self):
        globals.LastState = GameStates.Human_VS_Human

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
        self.Player2 = Human('black',self.board)