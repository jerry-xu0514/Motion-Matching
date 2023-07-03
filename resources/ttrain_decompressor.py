
# Imports
import sys
import struct
import contextlib  
import os

import cupy as cp
import tquat
import txform
import bvh

import time
import datetime

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from train_common import load_database, load_features, save_network

from tqdm import tqdm

# check if GPU is available, and if not, revert back and use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# allow prints to work with tqdm
class DummyFile(object):
  file = None
  def __init__(self, file):
    self.file = file

  def write(self, x):
    # Avoid print() second call (useless \n)
    if len(x.rstrip()) > 0:
        tqdm.write(x, file=self.file)

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout

# define the network

class Compressor(nn.Module):
   
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Compressor, self).__init__()

        self.linear0 = nn.Linear(input_size, hidden_size).to(device)
        self.linear1 = nn.Linear(hidden_size, hidden_size).to(device)
        self.linear2 = nn.Linear(input_size, hidden_size).to(device)
        self.linear3 = nn.Linear(hidden_size, output_size).to(device)
    
    def forward(self, x):
        nbatch, nwindow = x.shape[:2]

        x = x.reshape(nbatch*nwindow, -1)
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x.reshape(nbatch, nwindow, -1)
    
class Decompressor(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Decompressor, self).__init__()

        self.linear0 = nn.Linear(input_size, hidden_size).to(device)
        self.linear1 = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, x):
        nbatch, nwindow = x.shape[:2]
        x = x.reshape([nbatch * nwindow, -1])
        x = F.relu(self.linear0(x))
        x = self.linear1(x)
        return x.reshape(nbatch, nwindow, -1)
    


# Training Procedure

if __name__ == '__main__':

    outputs = open("outputs.txt", 'a')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.backends.cudnn.benchmark = True

    (name, batchsize_group, test_run) = sys.argv
    run_config = []
    if(batchsize_group == "-high"):
        x = [(1, 128), (2, 128), (4, 128), (8, 128), (16, 128)]
    elif(batchsize_group == "-mid"):
        x = [(1, 64), (2, 64), (4, 64), (8, 64), (16, 64)]
    elif(batchsize_group == "-low"):
        x = [(1, 32), (2, 32), (4, 32), (8, 32), (16, 32)]

    for threadcount, curr_batchsize in tqdm(x):
        
        # Check run time for current configuration
        tic = time.time()

        # Load data
        database = load_database('./database.bin')

        parents = database['bone_parents']
        contracts = database['contact_states']
        range_starts = database['range_starts']
        range_stops = database['range_stops']

        X = load_features('./features.bin')
