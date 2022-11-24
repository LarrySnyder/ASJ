import pygame
import math
from utils import blit_rotate_center

TRUCK =  pygame.image.load("images/truck.png")
MAP   = pygame.image.load("images/map.png")

LOCATIONS = [(62, 164), (116, 412), (194, 640), (380, 306), (542, 525), (711, 690), (720, 471), (837, 222)]




PATH = [(62, 164), (100.5, 333), (73.5, 366), 
        (116, 412), (157.5, 484.5), (153, 526.5), (194, 640),
        (228, 600), (261, 597), (273, 529.5), (321, 478), (380, 306), (528, 501), (542, 525), (541, 609),
        (658, 612), (744, 615), (756, 676), (711, 690), (756, 676), (760, 555), (697.5, 510), (720, 471), (837, 222)]



bg_str = pygame.image.tostring(MAP, "RGB")
bg_size = MAP.get_size()


class Truck:

    def __init__(self, positions):
        # angle = math.pi
        self.positions = positions
        self.point_in_path = 0
        self.location = 0
        init_position = positions[0]
        self.x = init_position[0]
        self.y = init_position[1]
        self.angle = math.pi
        self.img = TRUCK
        self.path = PATH
        self.rotation_vel = 1
        self.velocity = 3
        self.stop = True
        self.arrived = False


    def draw_truck(self, win):
        width, height = self.img.get_width(), self.img.get_height()
        blit_rotate_center(win, self.img, (self.x- width/2, self.y-height/2), self.angle)

    
    def move_truck(self):
        self.stop = False

    def move(self):

        if (self.point_in_path+1 >= len(self.path)):
            self.arrived = True
            return

        if (self.stop):
            return
        
        self.calculate_angle_update_path_point()
        vertical = math.cos(self.angle)*self.velocity
        horizontal = math.sin(self.angle)*self.velocity

        self.y -= vertical
        self.x -= horizontal

        target_x, target_y = self.path[self.point_in_path+1]
        if abs(self.x - target_x) < 2*horizontal or abs(self.y - target_y) < 2*vertical:
            self.point_in_path += 1
            if self.is_next_location(self.point_in_path):
                self.stop = True
                self.location += 1

            


    def calculate_angle_update_path_point(self):
        target_x, target_y = self.path[self.point_in_path + 1]
        x_diff = target_x - self.x
        y_diff = target_y - self.y


        if y_diff == 0:
            desired_angle = math.pi/2
        else: 
            desired_angle = math.atan(x_diff/y_diff)

        if target_y > self.y:
            desired_angle += math.pi

        difference_angle = self.angle - desired_angle

        if difference_angle >= math.pi:
            difference_angle -= 2*math.pi
        
        if difference_angle >0:
            self.angle -= min(self.rotation_vel, abs(difference_angle))
        else:
            self.angle += min(self.rotation_vel, abs(difference_angle))



    def is_next_location(self, point):
        index_in_path = PATH.index(LOCATIONS[self.location+1])
        return (point == index_in_path)

        
    def get_location(self):
        return self.location


    def reset(self):
        self.point_in_path = 0
        self.location = 0
        init_position = self.positions[0]
        self.x = init_position[0]
        self.y = init_position[1]
        self.angle = math.pi
        self.stop = True
        self.arrived = False

    