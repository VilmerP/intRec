import pygame
from pygame import gfxdraw
import numpy as np
from nn.network import Network
import math

network = Network()

window_width, window_height = 1000, 500
canvas_scale = 30

pygame.init()
pygame.display.set_caption('intRec')

#Create display
surface = pygame.display.set_mode((1000, 500))
window = pygame.display.set_mode((window_width, window_height))

#values for canvas position
canvas_pos = (window_width/16, window_height/16)
canvas_size = (3*window_width/8, 7*window_height/8)

#Create different surfaces
canvas = pygame.Surface((28, 28))
clear_but = pygame.Surface((100, 20))
check_but = pygame.Surface((100, 20))
int_but = pygame.Surface((300, 300))
output = pygame.Surface((100, 100))
cover = pygame.Surface((300, 300))

# create rectangles
canvas_rect = pygame.Rect(canvas_pos, canvas_size)
clear_but_rect = pygame.Rect((10*window_width/16, 12*window_height/16), (2*window_width/8, 0.8*window_height/8))
check_but_rect = pygame.Rect((10*window_width/16, 10*window_height/16), (2*window_width/8, 0.8*window_height/8))
int_but_rect = pygame.Rect((10*window_width/16, 2*window_height/16), (2*window_width/8, 0.8*window_height/8))

#Bools/values for checking mouse position across surfaces and drawing.
last_pos = None
drawing = False
running = True
mouse_on_can = False
mouse_on_cle = False
mouse_on_che = False

#Colors, filling surfaces with said colors.
blue = (0, 0, 255)
red = (205, 0, 0)
green = (0, 225, 0)
white = (255, 255, 255)
black = (0, 0, 0)
canvas.fill((0, 0, 0))
output.fill((155, 155, 155))
check_but.fill(green)
clear_but.fill(red)
int_but.fill(black)
cover.fill(black)
surface.fill((100, 100, 100))

#Create font for the answer guessed by the NN.
font_path = pygame.font.match_font("couriernew")
font = pygame.font.Font(font_path, 20)
int_font = pygame.font.Font(font_path, 300)
check_font = pygame.font.Font.render(font, "Check", True, white)
clear_font = pygame.font.Font.render(font, "Clear", True, white)

brush_size = 2 #radius of brush
BLUE = (0, 0, 255)

#Creates a surface that can be drawn (blitted) onto the canvas later.
def create_brush(radius: int, color: tuple) -> pygame.Surface:
    brush = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
    for i in range(radius*2):
        for j in range(radius*2):
            dist = math.sqrt((i - radius)**2 + (j - radius)**2)
            if dist < radius:
                alpha = int(255 * (1 - dist / radius))
                brush.set_at((i, j), (*color, alpha))
    return brush

#Create surface.
brush = create_brush(brush_size, white)

#Input a string and get out it's representation as a pygame.font.Font, which can be shown (blitted) on a surface
def integer_font(ans_string: str) -> pygame.font.Font:
    ret_font = pygame.font.Font.render(int_font, ans_string, True, white)
    return ret_font

#Calculate the position of the mouse on the canvas, returns tuple
def convert_can_pix(position: tuple) -> tuple:
    arr = list(position)
    arr[0] = round((-window_width/16 + arr[0])/13.7)
    arr[1] = round((-window_height/16 + arr[1])/15.6)
    ret_tup = tuple(arr)
    return ret_tup

#Converts the canvas array 28x28 2d-array into a 1d-array in the same format as the MNIST uses. 
def arrayfxr(array: list) -> np.ndarray:
    new_array = np.array(array)
    new_array = np.transpose(new_array)
    return new_array.flatten()


while running:
    for event in pygame.event.get():
        if (event.type == pygame.QUIT):
            running = False
        elif (event.type == pygame.MOUSEBUTTONUP):  
            drawing = False
        elif (mouse_on_cle and event.type == pygame.MOUSEBUTTONDOWN):
            canvas.fill((0, 0, 0))
            int_but.blit(cover, (0, 0))
        elif (mouse_on_che and event.type == pygame.MOUSEBUTTONDOWN):
            arr = pygame.surfarray.array_blue(canvas)
            check = arrayfxr(arr)
            result = network.evaluate(check)    #Run drawn digit through the NN
            int_but.blit(integer_font(str(result)), (50, 0))
        
        #During mouse motion, check on which surface the mouse is on.
        elif (event.type == pygame.MOUSEMOTION):
            mouse_pos = pygame.mouse.get_pos()
            if(canvas_rect.collidepoint(mouse_pos)):
                mouse_on_can = True 
                if(drawing and mouse_on_can):       #Draw on canvas.
                    mouse_position = pygame.mouse.get_pos()
                    if last_pos is not None:
                        canvas.blit(brush, (convert_can_pix(mouse_position)[0] - brush_size, convert_can_pix(mouse_position)[1] - brush_size))
                    last_pos = mouse_position
            elif(check_but_rect.collidepoint(mouse_pos)):
                mouse_on_che = True
            elif(clear_but_rect.collidepoint(mouse_pos)):
                mouse_on_cle = True
            else:
                mouse_on_cle = False
                mouse_on_can = False
                mouse_on_che = False
        elif (mouse_on_can and event.type == pygame.MOUSEBUTTONDOWN):
            drawing = True   


    #Blit (draw) all the surfaces
    window.blit(pygame.transform.scale(canvas, (3*window_width/8, 7*window_height/8)), (window_width/16, window_height/16)) 
    window.blit(pygame.transform.scale(output, (3*window_width/8, 7*window_height/8)), (9*window_width/16, window_height/16))
    window.blit(pygame.transform.scale(clear_but, (2*window_width/8, 0.8*window_height/8)), (10*window_width/16, 12*window_height/16))
    window.blit(pygame.transform.scale(check_but, (2*window_width/8, 0.8*window_height/8)), (10*window_width/16, 10*window_height/16))
    window.blit(pygame.transform.scale(int_but, (2*window_width/8, 3.75*window_height/8)), (10*window_width/16, 2*window_height/16))
    check_but.blit(check_font, (18, 0))
    clear_but.blit(clear_font, (18, 0))
    
    pygame.display.flip()