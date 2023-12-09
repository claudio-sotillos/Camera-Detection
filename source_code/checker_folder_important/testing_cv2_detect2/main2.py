from Detector import *

# this code will not run properly on Google Colab, since the CV2 imshow method does not work well,
# this code ran on linux 20.04
# date: 25 october 2023

detector = Detector()
detector.onImage("input.jpg")