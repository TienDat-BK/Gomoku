from core.Player import *
from core.Board import *

class Human(Player):
    def __init__(self,color,board : Board):
        super().__init__()
        self.pieceColor = color

        if color == "white":
            self.color = 1
        else:
            self.color = -1
        self.board = board
    
    def action(self, pos: tuple, events: list):
        # đánh luôn, để logic kiểm tra vị trí đánh hợp lệ ở Board
        # nếu action thành công sẽ trả theo set_cell
        super().action(pos)
        x, y = pos
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Your logic for mouse click
                if self.board.set_cell(x,y,1,state= self.color):
                    return True
                else:
                    return False


