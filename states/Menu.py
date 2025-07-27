import pygame
import os
import sys
from states.BaseState import *
from Gui.button import *
from setting import *
from display import *


# phải có declare_newState
@ declare_newState(GameStates.MenuState)
class MenuState(BaseState):
    def __init__(self):
        super().__init__()
        self.background = None

        self.Load_image()
        self.declare_obj()

    def Load_image(self):
        self.mainPath = os.path.dirname(os.path.dirname(__file__))
        self.background = pygame.image.load(os.path.join(self.mainPath,'assets/background.png')).convert_alpha()
        self.background = pygame.transform.smoothscale(self.background, (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.tile_game = pygame.image.load(os.path.join(self.mainPath, 'assets/gomoku-Photoroom.png')).convert_alpha()
        # giảm kích thước hình ảnh tile_game
        self.tile_game = pygame.transform.smoothscale(
            self.tile_game, 
            (self.tile_game.get_width() // 1.75, self.tile_game.get_height() // 1.75)
        )
        # đặt vị trí của tile_game
        self.tile_game_rect = self.tile_game.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))

    def declare_obj(self):
        # khởi tạo các OBJ có trong MENU
        self.obj = {'exit' : Button(button_menu, 'Exit', callback=self.exit_button_process),
                    'Human_VS_Human' : Button(button_menu, '2 Player', callback=self.human_human_process),
                    '1_Player_Vs_AI' : Button(button_menu, '1 Player Vs AI', callback=self.training_process)
                    }
        # khởi tạo vii trí các button
        self.obj['1_Player_Vs_AI'].change_pos(SCREEN_WIDTH / 2, SCREEN_HEIGHT * 0.45)
        self.obj['Human_VS_Human'].change_pos(SCREEN_WIDTH / 2, SCREEN_HEIGHT * (0.45 + 0.55 / 5))
        self.obj['exit'].change_pos(SCREEN_WIDTH / 2, SCREEN_HEIGHT  * (0.45 + 0.55 / 5 * 2) )

    def update(self):
        for obj in self.obj.values():
            obj.handle_events(self.manager.events)
            if obj.isClick(self.manager.events):
                obj.callback()

    def render(self):
        screen.blit(self.background, (0, 0))
        screen.blit(self.tile_game, self.tile_game_rect)
        # render các obj có trong menu
        for obj in self.obj.values():
            obj.render()
            
    def exit_button_process(self):
        pygame.quit()
        sys.exit()
    
    def human_human_process(self):
        self.manager.change_state(GameStates.Human_VS_Human)
    
    def training_process(self):
        self.manager.change_state(GameStates.VsAIState)

    
