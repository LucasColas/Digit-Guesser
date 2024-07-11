import pygame
from gui.gui import GUI

# Initialize pygame 
pygame.init()

def main():

    screen = pygame.display.set_mode((280, 280))
    pygame.display.set_caption('Digit Recognizer')
    gui = GUI('DeepLearning/netV14.pth')
    gui.run()

if __name__ == '__main__':
    main()