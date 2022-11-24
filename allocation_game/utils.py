import pygame
import math
import time

def scale_image(img, factor):

    size = round(img.get_width() * factor), round(img.get_height() *factor)
    return pygame.transform.scale(img, size)    


def blit_rotate_center(win, image, top_left, angle):

    rotated_image = pygame.transform.rotate(image, math.degrees(angle))
    new_rect = rotated_image.get_rect(center=image.get_rect(x = top_left[0], y=top_left[1]).center)
    win.blit(rotated_image, new_rect.topleft)


