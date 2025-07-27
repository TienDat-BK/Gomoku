
from setting import *
import pygame
from states.BaseState import *
from states.GameStateManager import *
from Gui.button import *
from display import *
import globals

@declare_newState(GameStates.GameOver)
class GameOverState(BaseState):
    def __init__(self):
        super().__init__()

        self.font = pygame.font.Font(None, 64)
        self.result_text = f"{globals.WhoWin} Wins!"

        self.obj = {
            'menu': Button(button_menu, 'Menu', callback=self.menu_process),
            'retry': Button(button_menu, 'Retry', callback=self.retry_process)
        }
        # Đặt vị trí nút
        self.obj['menu'].change_pos(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60)
        self.obj['retry'].change_pos(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 140)

    def menu_process(self):
        self.manager.change_state(GameStates.MenuState)

    def retry_process(self):
        self.manager.change_state(globals.LastState)
        
    def update(self):
        for obj in self.obj.values():
            obj.handle_events(self.manager.events)
            if obj.isClick(self.manager.events):
                obj.callback()
                return

    def render(self):
        # screen.blit(self.background, (0, 0))
        # Vẽ kết quả
        text_surf = self.font.render(self.result_text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 60))
        screen.blit(text_surf, text_rect)
        # Vẽ nút
        for obj in self.obj.values():
            obj.render()