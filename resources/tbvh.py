import re, os, ntpath
import torch

channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}

def load(filename, order=None):

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = torch.tensor([], dtype=torch.float64).reshape((0,4))
    offsets = torch.tensor([], dtype=torch.float64).reshape((0,3))
    parents = torch.tensor([], dtype=torch.int32)

    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = torch.cat((offsets, torch.zeros((1,3),dtype=torch.float64)),dim=0)
            orients = torch.cat((orients, torch.zeros((1,3),dtype=torch.float64)),dim=0)
            parents = torch.cat(parents, active)
            active = len(parents) -1
            continue
        
    