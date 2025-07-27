import pygame
from setting import *
from states import  GameStateManager, Menu 

pygame.init()


Game = GameStateManager.GameStateManager()
# State bắt đầu sẽ là Menu State
Game.change_state(GameStates.MenuState)



running = True

while running:
    events = pygame.event.get()
    for e in events:
        if e.type == pygame.QUIT:
            running = False
    
    Game.handle_events(events)
    Game.update()
    Game.render()

    # test vị trí

    pygame.display.update()


