import os

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
        self.model = Net(4).to(self.device)
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
        

    def save_label(self, label):
        name = f'{label}_' + str(len(os.listdir(f'DeepLearning/dataset/{label}'))+1)
        print(f"Saving {name}.png")
        image = pygame.surfarray.array3d(self.screen)
        image = np.flipud(image)  # Invert along Y axis
        image = np.rot90(image, k=-1).copy()
                       
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
        # Save the image but don't apply self.transform
        save_image(image_tensor, f'DeepLearning/dataset/{label}/{name}.png')


    def run(self):
        # Run the game
        self.screen.fill(BLACK)
        while True:
            # Check for events
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
                        

                        """
                        # Get the image
                        image = pygame.surfarray.array3d(self.screen)
                        image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
                        
                        # Resize to 28x28 (assuming MNIST-like input size)
                        image_resized = pygame.surfarray.make_surface(image_gray).convert()
                        image_resized = pygame.transform.scale(image_resized, self.image_size)
                        
                        # Convert to tensor and normalize
                        image_array = pygame.surfarray.pixels2d(image_resized)
                        image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0).float()
                        image_tensor /= 255.0  # Scale to [0, 1] range

                        # Get the prediction
                        with torch.no_grad():
                            output = self.model(image_tensor)
                            _, prediction = output.max(1)
                            print(f"Predicted Digit: {prediction.item()}")
                        """
                        # Clear the screen
                        self.screen.fill(BLACK)
            pygame.display.update()
            