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
            orients = torch.cat((orients, torch.tensor([[1,0,0,0]])),dim=0)
            parents = torch.cat(parents, active)
            active = len(parents) -1
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = torch.tensor([list(map(float, offmatch.groups()))])  # please double check this i have no clue what this is trying to do
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue
        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(rmatch.group(1))
            offsets = torch.cat((offsets, torch.zeros((1,3),dtype=torch.float64)),dim=0)
            orients = torch.cat((orients, torch.tensor([[1,0,0,0]])),dim=0)
            parents = torch.cat(parents, active)
            active = len(parents) -1
            continue
        
        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            fnum = int(fmatch.group(1))
            positions = offsets.unsqueeze(0).repeat(fnum,1)
            rotations = torch.zeros((fnum, len(orients), 3), dtype=torch.float64)
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            ftime = float(fmatch.group(1))
            continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = torch.tensor((list(map(float, dmatch))), dtype=torch.float64)
            N = len(parents)
            fi = i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1
    f.close()

    return {
        'rotations': rotations,
        'positions': positions,
        'offsets': offsets,
        'parents': parents,
        'names': names,
        'order': order,
    }

def save_joint(f, data, t, i, save_order, order='zyx', save_positions=False):

    save_order.append(i)
    
    f.write("%sJOINT %s\n" % (t, data['names'][i]))
    f.write("%s{\n" % t)
    t += '\t'
  
    f.write("%sOFFSET %f %f %f\n" % (t, data['offsets'][i,0], data['offsets'][i,1], data['offsets'][i,2]))
    
    if save_positions:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    else:
        f.write("%sCHANNELS 3 %s %s %s\n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    
    end_site = True
    
    for j in range(len(data['parents'])):
        if data['parents'][j] == i:
            t = save_joint(f, data, t, j, save_order, order=order, save_positions=save_positions)
            end_site = False
    
    if end_site:
        f.write("%sEnd Site\n" % t)
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
        t = t[:-1]
        f.write("%s}\n" % t)
  
    t = t[:-1]
    f.write("%s}\n" % t)
    
    return t
    
def save(filename, data, frametime=1.0/60.0, save_positions=False):
    
    order = data['order']
    
    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, data['names'][0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % (t, data['offsets'][0,0], data['offsets'][0,1], data['offsets'][0,2]) )
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % 
            (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))

        save_order = [0]
            
        for i in range(len(data['parents'])):
            if data['parents'][i] == 0:
                t = save_joint(f, data, t, i, save_order, order=order, save_positions=save_positions)
      
        t = t[:-1]
        f.write("%s}\n" % t)

        rots, poss = data['rotations'], data['positions']

        f.write("MOTION\n")
        f.write("Frames: %i\n" % len(rots));
        f.write("Frame Time: %f\n" % frametime);
        
        for i in range(rots.shape[0]):
            for j in save_order:
                
                if save_positions or j == 0:
                
                    f.write("%f %f %f %f %f %f " % (
                        poss[i,j,0],                  poss[i,j,1],                  poss[i,j,2], 
                        rots[i,j,ordermap[order[0]]], rots[i,j,ordermap[order[1]]], rots[i,j,ordermap[order[2]]]))
                
                else:
                    
                    f.write("%f %f %f " % (
                        rots[i,j,ordermap[order[0]]], rots[i,j,ordermap[order[1]]], rots[i,j,ordermap[order[2]]]))

            f.write("\n")
    