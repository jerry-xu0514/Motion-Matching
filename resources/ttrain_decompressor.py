
# Imports
import sys
import struct
import contextlib  
import os

import numpy as np
import tquat as quat
import txform
import tbvh as bvh

import time
import datetime

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ttrain_common import load_database, load_features, save_network

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
        database = load_database('./database.bin').to(device)

        parents = database['bone_parents']
        contacts = database['contact_states']
        range_starts = database['range_starts']
        range_stops = database['range_stops']

        X = load_features('./features.bin')['features'].type(torch.float32).to(device)
        Ypos = database['bone_positions'].type(torch.float32)
        Yrot = database['bone_rotations'].type(torch.float32)
        Yvel = database['bone_velocities'].type(torch.float32)
        Yang = database['bone_angular_velocities'].type(torch.float32)

        nframes = Ypos.shape[0]
        nbones = Ypos.shape[1]
        nextra = contacts.shape[1]
        nfeatures = X.shape[1]
        nlatent = 32

        # Parameteres

        seed = 1202
        batchsize = curr_batchsize
        lr = 0.001
        niter = 500000
        window = 2
        dt = 1.0 / 60.0

        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(threadcount)

        # Compute world space

        Grot, Gpos, Gvel, Gang = quat.fk_vel(Yrot, Ypos, Yvel, Yang, parents)

        # Compute character Space

        Qrot = quat.inv_mul(Grot[:,0:1], Grot)
        Qpos = quat.inv_mul_vec(Grot[:,0:1], Gpos - Gpos[:,0:1])
        Qvel = quat.inv_mul_vec(Grot[:,0:1], Gvel)
        Qang = quat.inv_mul_vec(Grot[:,0:1], Gang)

        # Compute transformation matrix

        Yxfm = quat.toxform(Yrot)
        Qxfm = quat.toxform(Qrot)

        # Compute two-column transformation matrix

        Ytxy = quat.to_xform_xy(Yrot).type(torch.float32)
        Qtxy = quat.to_xform_xy(Qrot).type(torch.float32)

        # Compute local root velocity

        Yrvel = quat.inv_mul_vec(Yrot[:,0:1], Yvel[:,0:1])
        Yrang = quat.inv_mul_vec(Yrot[:,0:1], Yang[:,0:1])

        # Compute extra outputs (contacts)

        Yextra = contacts.type(torch.float32)

        # Compute means/stds

        Ypos_scale = Ypos[:,1:].std()
        Ytxy_scale = Ytxy[:,1:].std()
        Yvel_scale = Yvel[:,1:].std()
        Yang_scale = Yang[:,1:].std()

        Qpos_scale = Qpos[:,1:].std()
        Qtxy_scale = Qtxy[:,1:].std()
        Qvel_scale = Qvel[:,1:].std()
        Qang_scale = Qang[:,1:].std()
        
        Yrvel_scale = Yrvel.std()
        Yrang_scale = Yrang.std()
        Yextra_scale = Yextra.std()

        decompressor_mean_out = torch.as_tensor(torch.hstack([
            Ypos[:,1:].mean(axis=0).ravel(),
            Ytxy[:,1:].mean(axis=0).ravel(),
            Yvel[:,1:].mean(axis=0).ravel(),
            Yang[:,1:].mean(axis=0).ravel(),
            Yrvel.mean(axis=0).ravel(),
            Yrang.mean(axis=0).ravel(),
            Yextra.mean(axis=0).ravel(),
        ]).type(torch.float32)).to(device)

        decompressor_std_out = torch.as_tensor(torch.hstack([
            Ypos[:,1:].std(axis=0).ravel(),
            Ytxy[:,1:].std(axis=0).ravel(),
            Yvel[:,1:].std(axis=0).ravel(),
            Yang[:,1:].std(axis=0).ravel(),
            Yrvel.std(axis=0).ravel(),
            Yrang.std(axis=0).ravel(),
            Yextra.std(axis=0).ravel(),
        ]).type(torch.float32)).to(device)

        decompressor_mean_in = torch.zeros([nfeatures + nlatent], dtype=torch.float32)
        decompressor_std_in = torch.ones([nfeatures + nlatent], dtype=torch.float32)

        compressor_mean_in = torch.as_tensor(torch.hstack([
            Ypos[:,1:].mean(axis=0).ravel(),
            Ytxy[:,1:].mean(axis=0).ravel(),
            Yvel[:,1:].mean(axis=0).ravel(),
            Yang[:,1:].mean(axis=0).ravel(),
            Qpos[:,1:].mean(axis=0).ravel(),
            Qtxy[:,1:].mean(axis=0).ravel(),
            Qvel[:,1:].mean(axis=0).ravel(),
            Qang[:,1:].mean(axis=0).ravel(),
            Yrvel.mean(axis=0).ravel(),
            Yrang.mean(axis=0).ravel(),
            Yextra.mean(axis=0).ravel(),
        ]).type(torch.float32))

        compressor_std_in = torch.as_tensor(torch.hstack([
            Ypos_scale.repeat((nbones-1)*3),
            Ytxy_scale.repeat((nbones-1)*6),
            Yvel_scale.repeat((nbones-1)*3),
            Yang_scale.repeat((nbones-1)*3),
            Qpos_scale.repeat((nbones-1)*3),
            Qtxy_scale.repeat((nbones-1)*6),
            Qvel_scale.repeat((nbones-1)*3),
            Qang_scale.repeat((nbones-1)*3),
            Yrvel_scale.repeat(3),
            Yrang_scale.repeat(3),
            Yextra_scale.repeat(nextra),
        ]).type(torch.float32))

        # Make networks

        network_compressor = Compressor(len(compressor_mean_in), nlatent)
        network_decompressor = Decompressor(nfeatures + nlatent, len(decompressor_mean_out))

        def save_compressed_database():

            with torch.no_grad():

                # Pass database through compressor

                Z = network_compressor((torch.cat([
                    Ypos[:,1:].reshape([1, nframes, -1]),
                    Ytxy[:,1:].reshape([1, nframes, -1]),
                    Yvel[:,1:].reshape([1, nframes, -1]),
                    Yang[:,1:].reshape([1, nframes, -1]),
                    Qpos[:,1:].reshape([1, nframes, -1]),
                    Qtxy[:,1:].reshape([1, nframes, -1]),
                    Qvel[:,1:].reshape([1, nframes, -1]),
                    Qang[:,1:].reshape([1, nframes, -1]),
                    Yrvel.reshape([1, nframes, -1]),
                    Yrang.reshape([1, nframes, -1]),
                    Yextra.reshape([1, nframes, -1]),
                ], dim=-1) - compressor_mean_in) / compressor_std_in)[0]

                # Write latent variables

                with open('latent.bin', 'wb') as f:
                    f.write(struct.pack('II', nframes, nlatent) + Z.detach().cpu().numpy().astype(np.float32).ravel().tobytes())

                # Function to generate test animation for comparison

        def generate_animation():

            with torch.no_grad():

                # get slice of database for first clip

                start = range_starts[2]
                stop = min(start + 1000, range_stops[2])
                start = range_starts[2]
                stop = min(start + 1000, range_stops[2])
                
                Ygnd_pos = Ypos[start:stop][None]
                Ygnd_rot = Yrot[start:stop][None]
                Ygnd_txy = Ytxy[start:stop][None]
                Ygnd_vel = Yvel[start:stop][None]
                Ygnd_ang = Yang[start:stop][None]
                
                Qgnd_pos = Qpos[start:stop][None]
                Qgnd_txy = Qtxy[start:stop][None]
                Qgnd_vel = Qvel[start:stop][None]
                Qgnd_ang = Qang[start:stop][None]
                
                Ygnd_rvel = Yrvel[start:stop][None]
                Ygnd_rang = Yrang[start:stop][None]
                Ygnd_extra = Yextra[start:stop][None]
                
                Xgnd = X[start:stop][None]

                    # pass through compressor

                Zgnd = network_compressor((torch.cat([
                    Ygnd_pos[:,:,1:].reshape([1, stop-start, -1]),
                    Ygnd_txy[:,:,1:].reshape([1, stop-start, -1]),
                    Ygnd_vel[:,:,1:].reshape([1, stop-start, -1]),
                    Ygnd_ang[:,:,1:].reshape([1, stop-start, -1]),
                    Qgnd_pos[:,:,1:].reshape([1, stop-start, -1]),
                    Qgnd_txy[:,:,1:].reshape([1, stop-start, -1]),
                    Qgnd_vel[:,:,1:].reshape([1, stop-start, -1]),
                    Qgnd_ang[:,:,1:].reshape([1, stop-start, -1]),
                    Ygnd_rvel.reshape([1, stop-start, -1]),
                    Ygnd_rang.reshape([1, stop-start, -1]),
                    Ygnd_extra.reshape([1, stop-start, -1]),
                ], dim=-1) - compressor_mean_in) / compressor_std_in)

                # Pass through decompressor

                Ytil = (network_decompressor(torch.cat([Xgnd, Zgnd], dim=-1)) *
                        decompressor_std_out + decompressor_mean_out)
                
                # Extract required components

                Ytil_pos = Ytil[:,:, 0*(nbones-1): 3*(nbones-1)].reshape([1, stop-start, nbones-1, 3])
                Ytil_txy = Ytil[:,:, 3*(nbones-1): 9*(nbones-1)].reshape([1, stop-start, nbones-1, 3, 2])
                Ytil_rvel = Ytil[:,:,15*(nbones-1)+0:15*(nbones-1)+3].reshape([1, stop-start, 3])
                Ytil_rang = Ytil[:,:,15*(nbones-1)+3:15*(nbones-1)+6].reshape([1, stop-start, 3])
            
                # Convert to quat and remove batch

                Ytil_rot = quat.from_xform_xy(Ytil_txy[0].detach().cpu().numpy())
                Ytil_pos = Ytil_pos[0].detach().cpu().numpy()
                Ytil_rvel = Ytil_rvel[0].detach().cpu().numpy()
                Ytil_rang = Ytil_rang[0].detach().cpu().numpy()
                # Integrate root displacement
                
                Ytil_rootrot = [Ygnd_rot[0,0,0].detach().cpu().numpy()]
                Ytil_rootpos = [Ygnd_pos[0,0,0].detach().cpu().numpy()]
                for i in range(1, Ygnd_pos.shape[1]):
                    Ytil_rootpos.append(Ytil_rootpos[-1] + quat.mul_vec(Ytil_rootrot[-1], Ytil_rvel[i-1]) * dt)
                    Ytil_rootrot.append(quat.mul(Ytil_rootrot[-1], quat.from_scaled_angle_axis(quat.mul_vec(Ytil_rootrot[-1], Ytil_rang[i-1]) * dt)))
                
                Ytil_rootrot = np.concatenate([r[np.newaxis] for r in Ytil_rootrot])
                Ytil_rootpos = np.concatenate([p[np.newaxis] for p in Ytil_rootpos])
                
                Ytil_rot = np.concatenate([Ytil_rootrot[:,np.newaxis], Ytil_rot], axis=1)
                Ytil_pos = np.concatenate([Ytil_rootpos[:,np.newaxis], Ytil_pos], axis=1)
                
                # Write BVH
                
                try:
                    bvh.save('decompressor_Ygnd.bvh', {
                        'rotations': np.degrees(quat.to_euler(Ygnd_rot[0].cpu().numpy())),
                        'positions': 100.0 * Ygnd_pos[0].cpu().numpy(),
                        'offsets': 100.0 * Ygnd_pos[0,0].cpu().numpy(),
                        'parents': parents,
                        'names': ['joint_%i' % i for i in range(nbones)],
                        'order': 'zyx'
                    })
                    
                    bvh.save('decompressor_Ytil.bvh', {
                        'rotations': np.degrees(quat.to_euler(Ytil_rot)),
                        'positions': 100.0 * Ytil_pos,
                        'offsets': 100.0 * Ytil_pos[0],
                        'parents': parents,
                        'names': ['joint_%i' % i for i in range(nbones)],
                        'order': 'zyx'
                    })
                except IOError as e:
                    print(e)
                
                fmin, fmax = Xgnd.detach.cpu().numpy().min(), Xgnd.detach.cpu().numpy().max()

                _, axs = plt.subplots(nfeatures, sharex=True, figsize=(12, 2*nfeatures))
                for i in range(nfeatures):
                    axs[i].plot(Xgnd[0,:500,i].detach().cpu().numpy())
                    axs[i].set_ylim(fmin, fmax)
                plt.tight_layout()

                try:
                    plt.savefig('decompressor_X.png')
                except IOError as e:
                    print(e)

                plt.close()
                
                # Write latent

                lmin, lmax = Zgnd.detach.cpu().numpy().min(), Zgnd.detach.cpu().numpy().max()

                _, axs = plt.subplots(nlatent, sharex=True, figsize=(12, 2*nlatent))
                for i in range(nlatent):
                    axs[i].plot(Zgnd[0,:500,i].detach().cpu().numpy())
                    axs[i].set_ylim(lmin, lmax)
                plt.tight_layout()

                try:
                    plt.savefig('decompressor_Z.png')
                except IOError as e:
                    print(e)
                
                plt.close()

        # Build batches respecting window size

        indicies = []
        for i in range(nframes - window):
            indicies.append(torch.arange(i, i + window))
        indicies = torch.as_tensor(indicies, dtype=torch.long, device=device)

        # Train

        writer = SummaryWriter()

        optimizer = torch.optim.AdamW(
            list(network_compressor.parameters()) +
            list(network_decompressor.parameters()),
            lr = lr,
            amsgrad=True,
            weight_decay=1e-3)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

        rolling_loss = None

        sys.stdout.write('Training...\n')

        for i in tqdm(range(niter)):

            optimizer.zero_grad()

            # Extract batch

            batch = indicies[torch.randint(0, len(indicies), size=[batchsize])]

            Xgnd = X[batch].to(device)

            Ygnd_pos = Ypos[batch]
            Ygnd_txy = Ytxy[batch]
            Ygnd_vel = Yvel[batch]
            Ygnd_ang = Yang[batch]
            
            Qgnd_pos = Qpos[batch]
            Qgnd_xfm = Qxfm[batch]
            Qgnd_txy = Qtxy[batch]
            Qgnd_vel = Qvel[batch]
            Qgnd_ang = Qang[batch]
            
            Ygnd_rvel = Yrvel[batch]
            Ygnd_rang = Yrang[batch]
            
            Ygnd_extra = Yextra[batch]

            # Encode

            Zgnd = network_compressor((torch.cat([
                Ygnd_pos[:,:,1:].reshape([batchsize, window, -1]),
                Ygnd_txy[:,:,1:].reshape([batchsize, window, -1]),
                Ygnd_vel[:,:,1:].reshape([batchsize, window, -1]),
                Ygnd_ang[:,:,1:].reshape([batchsize, window, -1]),
                Qgnd_pos[:,:,1:].reshape([batchsize, window, -1]),
                Qgnd_txy[:,:,1:].reshape([batchsize, window, -1]),
                Qgnd_vel[:,:,1:].reshape([batchsize, window, -1]),
                Qgnd_ang[:,:,1:].reshape([batchsize, window, -1]),
                Ygnd_rvel.reshape([batchsize, window, -1]),
                Ygnd_rang.reshape([batchsize, window, -1]),
                Ygnd_extra.reshape([batchsize, window, -1]),
            ], dim=-1) - compressor_mean_in) / compressor_std_in)

            # Decode

            Ytil = (network_decompressor(torch.cat([Xgnd, Zgnd], dim=-1)) * 
                decompressor_std_out + decompressor_mean_out)
            
            Ytil_pos = Ytil[:,:, 0*(nbones-1): 3*(nbones-1)].reshape([batchsize, window, nbones-1, 3])
            Ytil_txy = Ytil[:,:, 3*(nbones-1): 9*(nbones-1)].reshape([batchsize, window, nbones-1, 3, 2])
            Ytil_vel = Ytil[:,:, 9*(nbones-1):12*(nbones-1)].reshape([batchsize, window, nbones-1, 3])
            Ytil_ang = Ytil[:,:,12*(nbones-1):15*(nbones-1)].reshape([batchsize, window, nbones-1, 3])
            Ytil_rvel = Ytil[:,:,15*(nbones-1)+0:15*(nbones-1)+3].reshape([batchsize, window, 3])
            Ytil_rang = Ytil[:,:,15*(nbones-1)+3:15*(nbones-1)+6].reshape([batchsize, window, 3])
            Ytil_extra = Ytil[:,:,15*(nbones-1)+6:15*(nbones-1)+8].reshape([batchsize, window, nextra])
            
            # Add root bone from ground
            
            Ytil_pos = torch.cat([Ygnd_pos[:,:,0:1], Ytil_pos], dim=2)
            Ytil_txy = torch.cat([Ygnd_txy[:,:,0:1], Ytil_txy], dim=2)
            Ytil_vel = torch.cat([Ygnd_vel[:,:,0:1], Ytil_vel], dim=2)
            Ytil_ang = torch.cat([Ygnd_ang[:,:,0:1], Ytil_ang], dim=2)

                    # Do FK
        
            Ytil_xfm = txform.from_xy(Ytil_txy)

            Gtil_xfm, Gtil_pos, Gtil_vel, Gtil_ang = txform.fk_vel(
                Ytil_xfm, Ytil_pos, Ytil_vel, Ytil_ang, parents)
            
            # Compute Character Space
            
            Qtil_xfm = txform.inv_mul(Gtil_xfm[:,:,0:1], Gtil_xfm)
            Qtil_pos = txform.inv_mul_vec(Gtil_xfm[:,:,0:1], Gtil_pos - Gtil_pos[:,:,0:1])
            Qtil_vel = txform.inv_mul_vec(Gtil_xfm[:,:,0:1], Gtil_vel)
            Qtil_ang = txform.inv_mul_vec(Gtil_xfm[:,:,0:1], Gtil_ang)
            
            # Compute deltas
            
            Ygnd_dpos = (Ygnd_pos[:,1:] - Ygnd_pos[:,:-1]) / dt
            Ygnd_drot = (Ygnd_txy[:,1:] - Ygnd_txy[:,:-1]) / dt
            Qgnd_dpos = (Qgnd_pos[:,1:] - Qgnd_pos[:,:-1]) / dt
            Qgnd_drot = (Qgnd_xfm[:,1:] - Qgnd_xfm[:,:-1]) / dt
            
            Ytil_dpos = (Ytil_pos[:,1:] - Ytil_pos[:,:-1]) / dt
            Ytil_drot = (Ytil_txy[:,1:] - Ytil_txy[:,:-1]) / dt
            Qtil_dpos = (Qtil_pos[:,1:] - Qtil_pos[:,:-1]) / dt
            Qtil_drot = (Qtil_xfm[:,1:] - Qtil_xfm[:,:-1]) / dt
            
            Zdgnd = (Zgnd[:,1:] - Zgnd[:,:-1]) / dt
            
            # Compute losses
            
            loss_loc_pos = torch.mean(75.0 * torch.abs(Ygnd_pos - Ytil_pos))
            loss_loc_txy = torch.mean(10.0 * torch.abs(Ygnd_txy - Ytil_txy))
            loss_loc_vel = torch.mean(10.0 * torch.abs(Ygnd_vel - Ytil_vel))
            loss_loc_ang = torch.mean(1.25 * torch.abs(Ygnd_ang - Ytil_ang))
            loss_loc_rvel = torch.mean(2.0 * torch.abs(Ygnd_rvel - Ytil_rvel))
            loss_loc_rang = torch.mean(2.0 * torch.abs(Ygnd_rang - Ytil_rang))
            loss_loc_extra = torch.mean(2.0 * torch.abs(Ygnd_extra - Ytil_extra))
            
            loss_chr_pos = torch.mean(15.0 * torch.abs(Qgnd_pos - Qtil_pos))
            loss_chr_xfm = torch.mean( 5.0 * torch.abs(Qgnd_xfm - Qtil_xfm))
            loss_chr_vel = torch.mean( 2.0 * torch.abs(Qgnd_vel - Qtil_vel))
            loss_chr_ang = torch.mean(0.75 * torch.abs(Qgnd_ang - Qtil_ang))
            
            loss_lvel_pos = torch.mean(10.0 * torch.abs(Ygnd_dpos - Ytil_dpos))
            loss_lvel_rot = torch.mean(1.75 * torch.abs(Ygnd_drot - Ytil_drot))
            loss_cvel_pos = torch.mean(2.0  * torch.abs(Qgnd_dpos - Qtil_dpos))
            loss_cvel_rot = torch.mean(0.75 * torch.abs(Qgnd_drot - Qtil_drot))        
            
            loss_sreg = torch.mean(0.1  * torch.abs(Zgnd))
            loss_lreg = torch.mean(0.1  * torch.square(Zgnd))
            loss_vreg = torch.mean(0.01 * torch.abs(Zdgnd))
            
            loss = (
                loss_loc_pos + 
                loss_loc_txy + 
                loss_loc_vel + 
                loss_loc_ang + 
                loss_loc_rvel + 
                loss_loc_rang + 
                loss_loc_extra + 
                loss_chr_pos + 
                loss_chr_xfm + 
                loss_chr_vel + 
                loss_chr_ang + 
                loss_lvel_pos + 
                loss_lvel_rot + 
                loss_cvel_pos + 
                loss_cvel_rot + 
                loss_sreg + 
                loss_lreg +
                loss_vreg)
                    
            # Backprop
            
            loss.backward()

            optimizer.step()
        
            # Logging
            
            writer.add_scalar('decompressor/loss', loss.item(), i)
            
            writer.add_scalars('decompressor/loss_terms', {
                'loc_pos': loss_loc_pos.item(),
                'loc_txy': loss_loc_txy.item(),
                'loc_vel': loss_loc_vel.item(),
                'loc_ang': loss_loc_ang.item(),
                'loc_rvel': loss_loc_rvel.item(),
                'loc_rang': loss_loc_rang.item(),
                'loc_extra': loss_loc_extra.item(),
                'chr_pos': loss_chr_pos.item(),
                'chr_xfm': loss_chr_xfm.item(),
                'chr_vel': loss_chr_vel.item(),
                'chr_ang': loss_chr_ang.item(),
                'lvel_pos': loss_lvel_pos.item(),
                'lvel_rot': loss_lvel_rot.item(),
                'cvel_pos': loss_cvel_pos.item(),
                'cvel_rot': loss_cvel_rot.item(),
                'sreg': loss_sreg.item(),
                'lreg': loss_lreg.item(),
                'vreg': loss_vreg.item(),
            }, i)
            
            writer.add_scalars('decompressor/latent', {
                'mean': Zgnd.mean().item(),
                'std': Zgnd.std().item(),
            }, i)
            
            if rolling_loss is None:
                rolling_loss = loss.item()
            else:
                rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01
            
            if i % 10 == 0:
                sys.stdout.write('\rIter: %7i Loss: %5.3f' % (i, rolling_loss))
            
            if i % 1000 == 0:
                generate_animation()
                save_compressed_database()
                save_network('decompressor.bin', [
                    network_decompressor.linear0, 
                    network_decompressor.linear1],
                    decompressor_mean_in,
                    decompressor_std_in,
                    decompressor_mean_out,
                    decompressor_std_out)
                
            if i % 1000 == 0:
                scheduler.step()
                