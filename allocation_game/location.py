import pygame

from utils import scale_image


TEXT = ['O', 'V', 'M']


def round_closest(num):
    if ( num > 80):
        return 100
    elif num > 60:
        return 80
    elif num > 40:
        return 60
    elif num > 20:
        return 40
    else:
        return 20

def help_draw(i):
    if i == 1:
        return -20
    else:
        return 0


class location:

    def __init__(self, pos, amount_given=[0, 0, 0]):
        self.x, self.y = pos
        self.amount = amount_given
        self.satisfied = [100, 100, 100]
        self.envy = [0, 0, 0]
        self.set  = False 


    def set_amount(self, amount):
        self.set = True
        self.amount = amount

    def get_amount(self):
        return self.amount

    def get_satisfied(self, type):
        return self.satisfied[type]

    def update_envy(self, new_ass):
        for i in range(len(self.satisfied)):
            if self.amount[i]< new_ass[i]:
                new_envy = new_ass[i] - self.amount[i]
                if (new_envy > self.envy[i]):
                    self.envy[i] = new_envy


    def update_satisfaction(self, max_envy):
        for i in range(len(self.satisfied)):
            if (max_envy[i] > 0):
                self.satisfied[i] = (1- self.envy[i]/max_envy[i]) *100 # Avoid dividing by 0
            else:
                self.satisfied[i] = 100




    def draw(self, win):
        FONTARC = pygame.font.Font("arcade_n.ttf", 15)

        if self.set:
            for i in range(len(self.amount)):
                text = FONTARC.render(TEXT[i] , True , (0,0,0))
                im = scale_image(pygame.image.load("images/" +str(round_closest(self.satisfied[i])) +".png"), 0.7)
                win.blit(im, (self.x -50 + 40*i, self.y - 50 + help_draw(i)))
                win.blit(text, (self.x -50 + 40*i+5, self.y - 50 + help_draw(i) - 15 ))


    def reset(self):
        self.amount = [0, 0, 0]
        self.satisfied = [100, 100, 100]
        self.envy = [0, 0, 0]
        self.set  = False 
