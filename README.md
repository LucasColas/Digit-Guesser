# Digit-Guesser

GUI that uses AI to predict a drawn digit. The AI is a simple convolutionnal neural network (trained from scratch). The dataset was made by myself. I also used MNIST but with my dataset I got a better accuracy.

## How to use
git clone :
```bash
git clone git@github.com:LucasColas/Digit-Guesser.git
```
Go to the directory :
```bash
cd Digit-Guesser
```
Install the requirements (with pip for exemple) :
```bash
pip install -r requirements.txt
```

and then run the code : 
```bash
python main.py
```

## Draw and predict a digit
Draw a digit by pressing the left button of your mouse and going over the gui. Unpress the left button when you don't want to draw anything.
<img src="https://github.com/LucasColas/Digit-Guesser/blob/main/img/gui.png" width=25% height=25%>

press enter. And it will predict the digit : 

<img src="https://github.com/LucasColas/Digit-Guesser/blob/main/img/gui_pred.png" width=45% height=35%>


Other keyboard inputs. Press the delete key to erase the current draw on the gui.

## Train the neural network
Feel free to add more images. You can add more images to the dataset with images made thanks to the gui. Uncomment `self.save_label(2)` in `gui/gui.py` (line 93). The number passed in parameter in save_label is the label of the image. Then to train the neural network, run the cells of `DeepLearning/deep_learning.ipynb`. 

