import pygame
from gui.gui import GUI
# Initialize the game
pygame.init()

def main():
    # Initialize the GUI
    screen = pygame.display.set_mode((280, 280))
    pygame.display.set_caption('MNIST Digit Recognizer')
    gui = GUI('DeepLearning/mnist_netV5.pth')
    gui.run()

if __name__ == '__main__':
    main()