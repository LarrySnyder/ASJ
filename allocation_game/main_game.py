import pygame
from utils import scale_image
import truck
import numpy as np
import operator_fp
import button, location
import resourcegraph
# from sklearn.feature_extraction import image
# from PIL import Image



pygame.init()

MAP   = pygame.image.load("images/map.png")
MAP_BLURRED = pygame.image.load("images/map_blurred.png")
PIN   = scale_image(pygame.image.load("images/pin.png"), 0.02)


WIDTH, HEIGHT = MAP.get_width(), MAP.get_height()
ROUTE = pygame.image.load("images/route_all.png")
MAPBL = pygame.image.load("images/map_part.png")

PASTA = scale_image(pygame.image.load("images/pasta.png"), 0.08)
MEAT = scale_image(pygame.image.load("images/meat.png"), 0.08)
MEAL = scale_image(pygame.image.load("images/meal.png"), 0.08)


WIN = pygame.display.set_mode((WIDTH, HEIGHT))

#Frames per second
FPS = 60

LOCATIONS = [(62, 164), (116, 412), (194, 640), (380, 306), (542, 525), (711, 690), (720, 471), (837, 222)]



#Some functions

def generate_data():
    num_types = 3
    n = 8
    weights = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sum_of_rows = weights.sum(axis=1)
    weights = weights / sum_of_rows[:, np.newaxis]

    act_mean = np.asarray([3.5, 4.5, 5.5])
    mean_size = act_mean

    size = 5+ np.random.poisson(size = (n, num_types), lam = mean_size)
            # print('Size: ' + str(size))

    budget = np.asarray([np.dot(act_mean, weights[:,0])*n, np.dot(act_mean, weights[:,1])*n, np.dot(act_mean, weights[:,2])*n])

    return budget, size




def draw(win, images):

    for im, pos in images:
        win.blit(im, pos)



def get_fairness_score(locations):

    lst = [sum(locations[i].satisfied)/3 for i in range(len(locations))]
    return sum(lst) / len(lst)




def max_envy_so_far(locations): 
    max_envys = [0, 0, 0]
    for loc in locations:
        if loc.set:
            for i in range(3):
                if max_envys[i] < loc.envy[i]:
                    max_envys[i] = loc.envy[i]
    return max_envys


truck = truck.Truck(LOCATIONS)


budget, sizes = generate_data()

oper = operator_fp.Operator(budget, sizes, truck)
res = resourcegraph.Resource([(900, 500), (980, 500), (1060, 500)], oper)

color = (255,255,255)


# light shade of the button
color_light = (170,170,170)
  
# dark shade of the button
color_dark = (100,100,100)

smallfont = pygame.font.SysFont('Corbel',35)
arcadefont = pygame.font.Font("arcade_n.ttf", 22)
font_large = pygame.font.SysFont("Roboto", 25)
arcadesmall = pygame.font.Font("arcade_n.ttf", 12)
arcadetiny = pygame.font.Font("arcade_n.ttf", 15)
arcadehuge = pygame.font.Font("arcade_n.ttf", 40)
arcademedium = pygame.font.Font("arcade_n.ttf", 30)
# arcadehumong = pygame.font.Font("arcade_n.ttf", 40)



  
# Text Definition
text_1 = arcadetiny.render('Next' , True , color)
text_2 = arcadetiny.render('Finish' , True , color)
text_r = arcadefont.render('Resources' , True , (0,0,0))
text_after = arcadetiny.render('Came Today', True , (0,0,0))
text_g = arcadesmall.render('I will give them:', True, (0, 0, 0))
text_opening = arcadefont.render('Welcome to the Mobile Food Bank Game' , True , (0,0,0))
text_score = arcadehuge.render('Your Score is' , True, (0,0,0))


wbl, hbl = MAPBL.get_width(), MAPBL.get_height()
images = [(MAP, (0, 0)), (ROUTE, (-1, 0)), (MAPBL, (WIDTH- wbl-20, 20)), (MEAT, (920, 170)), (PASTA, (1000, 170)), (MEAL, (1080, 170))]
width_pin, height_pin = PIN.get_width(), PIN.get_height()
pins = [(PIN, (l[0]- width_pin/2, l[1]-height_pin)) for l in LOCATIONS]

pygame.display.set_caption("Allocation Game!")

# Buttons
start_button = button.Button(((450, 490)), "images/Button_pixelart.png")
text_sb = arcadehuge.render('START', True, (0, 0, 0))

rd1_button = button.Button(((200, 420)), "images/Button_pixelart.png")
text_rd1 = arcademedium.render('ROUND 1', True, (0, 0, 0))

rd2_button = button.Button(((700, 420)), "images/Button_pixelart.png")
text_rd2 = arcademedium.render('ROUND 2', True, (0, 0, 0))


res_button = button.Button((450, 490), "images/Button_pixelart.png")
text_res = arcademedium.render('Restart', True, (0, 0, 0))




run = True
not_finished = True
Started = False
clock = pygame.time.Clock()
clicked = False
isround2 = False
Rounds = False
show_score = False

location_fairness = [location.location(i) for i in LOCATIONS]

while run:

    if truck.arrived and location_fairness[truck.location].set:
        not_finished = False
        

    clock.tick(FPS)

    if (Started):

        draw(WIN, images)
        draw(WIN, pins)
        pygame.draw.rect(WIN, (0, 0, 0), (WIDTH- wbl-20, 20, wbl, hbl), 1,3)  # width = 3

        truck.draw_truck(WIN)

        #Draw the  inventory

        pygame.draw.rect(WIN, (0, 100, 255), (920, 60, 40, 100), 2,2)  # width = 3
        pygame.draw.rect(WIN, (255, 0, 0), (1000, 60,40, 100), 2,2)  # width = 3
        pygame.draw.rect(WIN, (100, 0, 255), (1080, 60, 40, 100), 2,2)  # width = 3

        oper.draw_resources(WIN)
        oper.draw_num_ind(WIN)
        truck.move()

        if (isround2):
            for i in range(len(LOCATIONS)):
                location_fairness[i].draw(WIN)

        if truck.stop and clicked:
            clicked =  False



        WIN.blit(text_r , (920,25))
        text_i = arcadetiny.render('At location '+str(truck.get_location()+1), True, (0, 0, 0))
        


        WIN.blit(text_i, (915, 260))
        WIN.blit(text_after, (915, 420))
        WIN.blit(text_g , (915,480))

        res.show_to_add(WIN)
        # stores the (x,y) coordinates into
        # the variable as a tuple
        mouse = pygame.mouse.get_pos()
        
        # if mouse is hovered on a button it
        # changes to lighter shade 
        if truck.stop or truck.arrived:
            if 950 <= mouse[0] <= 1090 and 600 <= mouse[1] <= 640:
                pygame.draw.rect(WIN,color_light,[950, 600,140,40], border_radius=3)
                
            else:
                pygame.draw.rect(WIN,color_dark,[950, 600,140,40], border_radius=3)
            
            # superimposing the text onto our button
            if (not_finished):
                WIN.blit(text_1 , (990,610))
            else:
                WIN.blit(text_2 , (970,610))

    else:

        draw(WIN, [(MAP_BLURRED, (0,0))])


        if (Rounds and not_finished):
            WIN.blit(text_opening, (180, 150))
            rd1_button.show(WIN)
            rd2_button.show(WIN)
            WIN.blit(text_rd1, (240, 455))
            WIN.blit(text_rd2, (740, 455))
        elif ((not Rounds) and not_finished):
            # draw(WIN, [(MAP_BLURRED, (0,0))])
            WIN.blit(text_opening, (180, 150))
            start_button.show(WIN)
            WIN.blit(text_sb, (500, 525))

        if (show_score):
            WIN.blit(text_score, (360, 250))
            score_eff = arcadefont.render('Efficiency: '+str(int(oper.compute_efficiency_score())), True, (0,0,0))
            WIN.blit(score_eff, (450, 380))
            if isround2:
                score_fair = arcadefont.render('Fairness: '+str(int(get_fairness_score(location_fairness))), True, (0,0,0))
                WIN.blit(score_fair, (450, 420))
            res_button.show(WIN)
            WIN.blit(text_res, (500, 525))



    for event in pygame.event.get():
        # button_try.click(event)
        res.update(event)
        if event.type == pygame.QUIT:
            run = False
            break
        if event.type == pygame.MOUSEBUTTONDOWN:
            
            #if the mouse is clicked on the
            # button the game is terminated
            if (Started):
                if 950 <= mouse[0] <= 1090 and 600 <= mouse[1] <= 640 and (truck.stop or truck.arrived):
                    clicked = True
                    truck.move_truck()

                    ressources_chosen = res.get_number()
                    oper.give_resource(ressources_chosen)
                    l = truck.get_location()
                    size = oper.get_size_at_location()
                    per_person = [ressources_chosen[i]/size[i] for i in range(len(ressources_chosen))]
                    location_fairness[l].set_amount(per_person)

                    if (isround2):

                        for loc in range(l):
                            prev = location_fairness[loc]
                            location_fairness[l].update_envy(prev.get_amount())
                            prev.update_envy(per_person)

                        max_envy = max_envy_so_far(locations=location_fairness)
                          

                        for loc in range(l+1):
                            location_fairness[loc].update_satisfaction(max_envy)
                    if not not_finished:

                        Started = False
                        show_score = True

                    res.reset()
            else:
                if (start_button.clicked(event)):
                    Rounds = True


                if Rounds:
                    if rd1_button.clicked(event):
                        Started = True
                        Rounds = False
                    elif rd2_button.clicked(event):
                        Started = True
                        Rounds  = False
                        isround2 = True

                if res_button.clicked(event):
                    Rounds = True
                    not_finished = True
                    show_score = False
                    isround2 = False
                    # Reset everything
                    truck.reset()
                    oper.reset()
                    for loc in location_fairness:
                        loc.reset()

                    

    
    pygame.display.update()

    

pygame.quit()