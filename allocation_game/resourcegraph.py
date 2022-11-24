import pygame
import button
import operator_fp



class Resource:

    def __init__(self, pos, oper):

        self.buttonplus = [button.Button(p, "images/plus.png", 0.7) for p in pos]

        self.buttonminus = [button.Button((p[0] + 50, p[1]-1) , "images/minus.png", 0.7) for p in pos]

        self.number = [0 for _ in pos]

        self.nr = len(pos)

        self.pos = pos
        self.font =  pygame.font.Font("arcade_n.ttf", 12)

        self.oper = oper

    def show_to_add(self, win):

        for i in range(self.nr):
            self.buttonplus[i].show(win)
            pygame.draw.rect(win,(255, 255, 255),[self.pos[i][0]+28, self.pos[i][1]+4,25,20])
            text = self.font.render(str(self.number[i]), True, (0, 0, 0))
            win.blit(text , (self.pos[i][0]+31, self.pos[i][1]+7))
            self.buttonminus[i].show(win)


    def update(self, event):

        for i in range(self.nr):
            
            if self.buttonplus[i].clicked(event):
                self.number[i] = int(min(self.oper.available[i], self.number[i]+1))

            if self.buttonminus[i].clicked(event):
                self.number[i] = max(0, self.number[i]-1)

    def reset(self):
        self.number = [0 for _ in range(self.nr)]

    def get_number(self):
        return self.number