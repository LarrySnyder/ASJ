import pygame
from utils import scale_image

class Button:
    """Create a button, then blit the surface in the while loop"""
 
    def __init__(self, pos, image, reduction=1):
        self.x, self.y = pos
        self.image = scale_image(pygame.image.load(image), reduction)
 
 
    def show(self, win):
        win.blit(self.image, (self.x, self.y))

 
    def clicked(self, event, otherevent=True):
        x, y = pygame.mouse.get_pos()
        width,height = self.image.get_width(), self.image.get_height()
        if event.type == pygame.MOUSEBUTTONDOWN and otherevent:
            # if pygame.mouse.get_pressed()[0]:
            if self.x <= x <= self.x+width and self.y <= y <= self.y+height:
                return True