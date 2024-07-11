import os
from tkinter import Tk, Label, Button

import pygame
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

from DeepLearning.nn import Net, transform


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)   

class GUI:
    def __init__(self, model_path, image_size=(96, 96)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Net(10).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.image_size = image_size
        self.screen = pygame.display.get_surface()

        self.drawing = False
        self.last_pos = None
        

    def save_label(self, label, path='DeepLearning/dataset/'):
        name = f'{label}_' + str(len(os.listdir(f'{path}{label}'))+1)
        print(f"Saving {name}.png")
        image = pygame.surfarray.array3d(self.screen)
        image = np.flipud(image)  # Invert along Y axis
        image = np.rot90(image, k=-1).copy()
        
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
        # Save the image but don't apply self.transform
        save_image(image_tensor, f'{path}{label}/{name}.png')

    def display_prediction(self, prediction):
        popup = Tk()
        popup.wm_title("Prediction")

        label = Label(popup, text=f"Predicted Digit: {prediction}", font=("Helvetica", 16))
        label.pack(side="top", fill="x", pady=20, padx=20)

        button = Button(popup, text="OK", command=popup.destroy)
        button.pack(side="bottom", pady=10)

        popup.mainloop()


    def run(self):
        
        self.screen.fill(BLACK)
        while True:
           
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.drawing = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.drawing = False
                    self.last_pos = None
                elif event.type == pygame.MOUSEMOTION:
                    
                    if self.drawing:
                        pos = pygame.mouse.get_pos()
                        if self.last_pos:
                            
                            pygame.draw.line(self.screen, WHITE, self.last_pos, pos, 10)
                        self.last_pos = pos

                
                elif event.type == pygame.K_SPACE:
                    self.screen.fill(BLACK)

                elif event.type == pygame.KEYDOWN:
                    
                    # press enter
                    if event.key == pygame.K_RETURN:

                        # get label
                        #self.save_label(2)
                        
                        
                        
                        
                        image = pygame.surfarray.array3d(self.screen)
                        image = np.flipud(image)  # Invert along Y axis
                        image = np.rot90(image, k=-1).copy()
                        
                        image = self.transform(image).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            output = self.model(image)
                            # apply softmax
                            output = torch.softmax(output, dim=1)
                            print(output)
                            _, prediction = output.max(1)
                            
                            print(f"Predicted Digit: {prediction.item()}")
                            self.display_prediction(prediction.item())

                    
                        # Clear the screen
                        self.screen.fill(BLACK)
            pygame.display.update()
            