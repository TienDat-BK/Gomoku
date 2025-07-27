import pygame
import os
from setting import *
from display import *
import math

class Piece_UI:
    # pieceColor = 'black' or ...
    def __init__(self, pieceColor):
        if pieceColor == 'black':
            self.image = pygame.image.load(os.path.join(os.getcwd(), 'assets/black_piece.jpg')).convert_alpha()
        else:
            self.image = pygame.image.load(os.path.join(os.getcwd(), 'assets/white_piece.jpg')).convert_alpha()
        
        self.image = pygame.transform.smoothscale(self.image, (PIECE_SIZE,PIECE_SIZE) )
        self.image.set_colorkey((255, 255, 255))

        self.rect = None
        # self.nearest_pos là quân cờ trên bàn cơ pixel
        self.nearest_pos = None
        # self.nearest_pos là vị trí quân cờ trên bàn cờ theo Ô
        self.nearest_pos_o = None

        self.mousePos = None
        self.events = None
    
    def handle_events(self,events):
        self.mousePos = pygame.mouse.get_pos()
        self.events = events

    def update(self):
        # trong 4 ô xuing quanh tìm ô gần nhất
        pos_o = Lay_o_quanco(self.mousePos)
        x,y = pos_o

        pos1 = (math.ceil(x), math.ceil(y))
        pos2 = (math.ceil(x), math.floor(y))
        pos3 = (math.floor(x), math.floor(y))
        pos4 = (math.floor(x), math.ceil(y))

        positions = [pos1, pos2, pos3, pos4]
        distances = [math.hypot(x - px, y - py) for px, py in positions]
        self.nearest_pos_o = positions[distances.index(min(distances))]
        self.nearest_pos = lay_vitri_quanco(self.nearest_pos_o)

    def get_nearest_pos_o(self):
        self.update()
        return self.nearest_pos_o 

    def checkValid(self):
        # Kiểm tra self.nearest_pos_o đều lớn hơn 1 và bé hơn 15
        if self.nearest_pos_o is None:
            return False
        x, y = self.nearest_pos_o
        if not (0<= x and x < 15 and 0 <= y and y < 15):
            return False
        
        # trả về true nếu con chuột có trong bán kính hợp lệ ở một ô cờ gần nhất
        if not (math.dist(self.nearest_pos,self.mousePos) < VALID_RADIUS) :
            return False
        
        return True

    def set_color(self,pieceColor):
        if pieceColor == 'black':
            self.image = pygame.image.load(os.path.join(os.getcwd(), 'assets/black_piece.jpg')).convert_alpha()
        else:
            self.image = pygame.image.load(os.path.join(os.getcwd(), 'assets/white_piece.jpg')).convert_alpha()

    def render(self):
        if self.checkValid() and self.nearest_pos is not None:
            self.rect = self.image.get_rect(center=self.nearest_pos)
            # thiết lập cho piece mờ mờ đi
            self.image.set_alpha(128)
            screen.blit(self.image, self.rect)




