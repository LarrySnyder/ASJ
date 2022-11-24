from turtle import color
import pygame
import numpy as np



class Operator:

    def __init__(self, budget, sizes, truck):

        self.budget    = budget.copy()
        self.available = budget
        self.numtypes  = len(budget)
        self.sizes     = sizes
        self.types     = ['omnivore', 'vegetarian', 'meal-prep only']
        self.showed_up = [0, 0, 0]
        self.truck     = truck


    def draw_num_ind(self, win):

        font =  pygame.font.Font("arcade_n.ttf", 12)

        num  = self.sizes[self.truck.location]

        for i in range(self.numtypes):
            text_type = font.render(str(int(num[i]))+' '+self.types[i], True, (0,0,0))

            win.blit(text_type, (925, 310+i*30))

        


    def give_resource(self, t):

        for i in range(self.numtypes):
            self.available[i] = max(self.available[i]- t[i], 0)
    
    def draw_resources(self, win):

        num_resources = len(self.budget)
        percentage = np.zeros((num_resources,))

        pos = [920, 1000, 1080]

        colors = [(0, 100, 255), (255, 0, 0), (100, 0, 255)]

        for i in range(num_resources):
            percentage[i] = self.available[i]/self.budget[i]

            pygame.draw.rect(win, colors[i], (pos[i], 160 - 100*percentage[i], 40, 100*percentage[i]),border_radius=2)  # width = 3

            pygame.draw.rect(win,(255, 255, 255),[pos[i]+5, 100,30,15])
            font =  pygame.font.Font("arcade_n.ttf", 12)
            text = font.render(str(int(self.available[i])), True, (0, 0, 0))
            if self.available[i] <100:
                win.blit(text , (pos[i]+10, 102))
            else:
                win.blit(text , (pos[i]+4, 102))




    def compute_efficiency_score(self):
        arr = [(self.budget[i] - self.available[i]) /self.budget[i] for i in range(self.numtypes)]

        return (sum(arr)*100)/3 


    def reset(self):
        self.available = self.budget.copy()
        self.showed_up = [0, 0, 0]


    def get_size_at_location(self):
        return self.sizes[self.truck.location]






