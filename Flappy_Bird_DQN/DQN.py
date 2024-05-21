import os
import pygame
DISPLAY = True
if not DISPLAY:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import os, sys
sys.path.append('game/')
import flappy_wrapped as game
import cv2
import numpy as np
import time
import collections
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

#############
ACTIONS = [0,1]
EXPERIENCE_BUFFER_SIZE = 2000
STATE_DIM = 4
GAMMA = 0.99
EPSILON_START = 1
EPSILON_FINAL = 0.001
EPSILON_DECAY_FRAMES = (10**4)/3
MEAN_GOAL_REWARD = 10
BATCH_SIZE = 32
MIN_EXP_BUFFER_SIZE = 500
SYNC_TARGET_FRAMES = 30
LEARNING_RATE = 1e-4
SKIP_FRAME = 2
INITIAL_SKIP = [0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1]

EPOCHS_MAX = 20000

KERNEL = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
IMAGE_SIZE = 84
def processFrame(image):
    # env.SCREENWIDTH = 288
    # env.BASEY = 404
    frame = image[55:288, :404] # crop image ->
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # convert image to black and white
    frame = cv2.resize(frame,(IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_AREA)
    _ , frame = cv2.threshold(frame,50,255,cv2.THRESH_BINARY)
    frame = cv2.filter2D(frame,-1,KERNEL)
    frame = frame.astype(np.float64)/255.0
    return frame

