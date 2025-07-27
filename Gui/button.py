import pygame
from display import *

class Button:
    def __init__(self, info, content, callback):
        # constructor ( info, content  )
        self.info = info
        self.callback = callback
        self.content  = content

        self.image = pygame.image.load( info['image']).convert_alpha()
        self.width = info['width']
        self.height = info['height']

        self.image = pygame.transform.smoothscale(self.image, (self.width,self.height) )

        # Constructor chưa khởi tạo x,y => chua thể tạo rect => phải dùng change_pos
        self.x = None
        self.y = None
        self.rect = None

        # self.state thế hiện trạng thái hiện tại của nút bấm
        self.state = 'normal'


    def change_size(self,width, height):
        self.height = height
        self.width = width
        self.image = pygame.transform.smoothscale(self.image, (self.width,self.height) )


    def change_pos(self,x,y):
        # thay đổi vị trí image, đang lấy theo topmid
        self.x = x
        self.y = y
        self.rect = self.image.get_rect(midtop=(self.x, self.y))

    def handle_events(self,events):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect and self.rect.collidepoint(mouse_pos):
            self.state = 'hovered'
        else:
            self.state = 'normal'

    def isClick(self,events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.rect and self.rect.collidepoint(event.pos):
                    return True
        return False

    def change_content(self, content):
        # chỉnh sửa chữ hiện trên button
        self.content = content

    def render(self):
        screen.blit(self.image, self.rect)

        # hiệu ứng hovered: làm mờ button khi hovered
        if self.state == 'hovered' and self.rect:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 100))  # alpha 100 làm mờ nhẹ
            screen.blit(overlay, self.rect.topleft)
        
        # render Content lên button
        if self.content and self.rect:
            font = pygame.font.Font(self.info['font'],self.info['text_size'])  # chọn font và cỡ chữ
            text_surf = font.render(self.content, True, self.info['text_color'])  # màu chữ đen
            text_rect = text_surf.get_rect(center=self.rect.center) # ở giữa buton
            screen.blit(text_surf, text_rect)