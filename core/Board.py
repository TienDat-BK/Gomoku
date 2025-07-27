import pygame
from display import *
from setting import *

class Board:
    def __init__(self):
        # Sử dụng int để lưu trạng thái bàn cờ 15x15 (225 bit)
        # Mặc định state1 là white_piece còn state2 là black_piece

        self.white_piece = pygame.image.load(os.path.join(os.getcwd(), 'assets/white_piece.jpg')).convert_alpha()
        self.black_piece = pygame.image.load(os.path.join(os.getcwd(), 'assets/black_piece.jpg')).convert_alpha()

        self.state1 = 0
        self.state2 = 0
        self.o_vua_danh = None
        self.is_start = True

    def set_cell(self, x, y, value, state):
        # value: 0 hoặc 1
        # state 1 hoặc -1
        # state 1 là white_piece, state -1 là black_piece

        if not self.isCellValid((x,y)):
            return False
        # đã kiểm tra vị trí đánh hợp lệ
        self.is_start = False
        self.o_vua_danh = (x,y)

        idx = y * 15 + x
        if state == 1:
            if value:
                self.state1 |= (1 << idx)
            else:
                self.state1 &= ~(1 << idx)
        else:
            if value:
                self.state2 |= (1 << idx)
            else:
                self.state2 &= ~(1 << idx)
        
        return True

    def get_cell(self, x, y,state):
        idx = y * 15 + x
        if state == 1:
            return (self.state1 >> idx) & 1
        else:
            return (self.state2 >> idx) & 1

    def isCellValid(self, pos: tuple):
        x, y = pos
        if not ((0 <= x < 15) and (0 <= y < 15)):
            return False
        return self.get_cell(x, y, 1) == 0 and self.get_cell(x, y, -1) == 0


    def check_win(self  , player_state: int , pos = None ) -> bool:
        if self.is_start:
            return False
        
        if pos == None:
            pos = self.o_vua_danh

        if pos == None:
            # print('ô vừa đánh ko hợp lệ')
            return False

        x, y = pos
        board = self.state1 if player_state == 1 else self.state2

        def count_in_direction(dx, dy):
            count = 0
            if self.get_cell(x,y,player_state) == 1:
                count = 1  # ô vừa đánh
            else:
                count = 0 

            for dir in [1, -1]:  # hai phía
                nx, ny = x + dir * dx, y + dir * dy
                while 0 <= nx < 15 and 0 <= ny < 15:
                    idx = ny * 15 + nx
                    if (board >> idx) & 1:
                        count += 1
                        nx += dir * dx
                        ny += dir * dy
                    else:
                        break
            return count

        # 4 hướng
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            if count_in_direction(dx, dy) >= 5:
                return True
        return False


    def render(self):
        for i in range(15):
            for j in range(15):
                if self.get_cell(i,j,1) != 0:
                    rect = self.white_piece.get_rect(center = lay_vitri_quanco((i,j)))
                    screen.blit(self.white_piece,rect)
                if self.get_cell(i,j,-1) != 0:
                    rect = self.black_piece.get_rect(center = lay_vitri_quanco((i,j)))
                    screen.blit(self.black_piece,rect)

    def reset(self):
        self.state1 = 0
        self.state2 = 0
        self.o_vua_danh = None
                

