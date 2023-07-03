import struct
import numpy as np
import torch

def load_database(filename):

    with open(filename, 'rb') as f:
        
        nframes, nbones = struct.unpack('II', f.read(8))
        temp = torch.frombuffer(f.read(nframes * nbones * 3 * 4), dtype = torch.float32, count = nframes * nbones * 3)
        bone_positions = torch.reshape(temp, [nframes, nbones, 3])

        nframes, nbones = struct.unpack('II', f.read(8))
        temp = torch.frombuffer(f.read(nframes * nbones * 3 * 4), dtype = torch.float32, count = nframes * nbones * 3)
        bone_velocities = torch.reshape(temp, [nframes, nbones, 3])

        nframes, nbones = struct.unpack('II', f.read(8))
        temp = torch.frombuffer(f.read(nframes * nbones * 3 * 4), dtype = torch.float32, count = nframes * nbones * 4)
        bone_rotations = torch.reshape(temp, [nframes, nbones, 3])

        nframes, nbones = struct.unpack('II', f.read(8))
        temp = torch.frombuffer(f.read(nframes * nbones * 3 * 4), dtype = torch.float32, count = nframes * nbones * 3)
        bone_angular_velocities = torch.reshape(temp, [nframes, nbones, 3])

        nbones = struct.unpack('I', f.read(4)[0])
        temp = torch.frombuffer(f.read(nbones * 4), dtype = torch.int32, count = nbones)
        bone_parents = torch.reshape(temp, [nbones])

        nranges = struct.unpack('I', f.read(4))[0]
        temp = torch.frombuffer(f.read(nranges * 4), dtype = torch.int32, count = nranges)
        range_starts = torch.reshape(temp, [nranges])

        nranges = struct.unpack('I', f.read(4))[0]
        temp = torch.frombuffer(f.read(nranges * 4), dtype = torch.int32, count = nranges)
        range_stops = torch.reshape(temp, [nranges])

        nframes, nbones = struct.unpack('II', f.read(8))
        temp = torch.frombuffer(f.read(nframes * nbones * 3 * 4), dtype = torch.int8, count = nframes * nbones * 3)
        contact_states = torch.reshape(temp, [nframes, nbones, 3])

    return {
        'bone_positions': bone_positions,
        'bone_rotations': bone_rotations,
        'bone_velocities': bone_velocities,
        'bone_angular_velocities': bone_angular_velocities,
        'bone_parents': bone_parents,
        'range_starts': range_starts,
        'range_stops': range_stops,
        'contact_states': contact_states,
    }

def load_features(filename):

    with open(filename, 'rb') as f:

        nframes, nfeatures = struct.unpack('II', f.read(8))
        temp = torch.frombuffer(f.read(nframes*nfeatures*4), dtype = torch.float32, count = nframes*nfeatures)
        features = torch.reshape(temp, [nframes, nfeatures])

        nfeatures = struct.unpack('I', f.read(4))[0]
        temp = torch.frombuffer(f.read(nfeatures*4), dtype = torch.float32, count = nfeatures)
        features_offset = torch.reshape(temp, [nfeatures])

        nfeatures = struct.unpack('I', f.read(4))[0]
        temp = torch.frombuffer(f.read(nfeatures*4), dtype = torch.float32, count = nfeatures)
        features_scale = torch.reshape(temp, [nfeatures])

        return {
            'features': features,
            'features_offset': features_offset,
            'features_scale': features_scale,
        }

def load_latent(filename):
    with open(filename, 'rb') as f:
        nframes, nfeatures = struct.unpack('II', f.read(8))
        temp = torch.frombuffer(f.read(nframes*nfeatures*4), dtype=torch.float32, count=nframes*nfeatures)
        latent = torch.reshape(temp, [nframes, nfeatures])
        
    return {
        'latent': latent,
    }

def save_network(filename, layers, mean_in, std_in, mean_out, std_out):
    with torch.no_grad():
        
        with open(filename, 'wb') as f:
            f.write(struct.pack('I', mean_in.shape[0]) + mean_in.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack('I', std_in.shape[0]) + std_in.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack('I', mean_out.shape[0]) + mean_out.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack('I', std_out.shape[0]) + std_out.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack('I', len(layers)))
            for layer in layers:
                f.write(struct.pack('II', *layer.weight.T.shape) + layer.weight.T.cpu().numpy().astype(np.float32).ravel().tobytes())
                f.write(struct.pack('I', *layer.bias.shape) + layer.bias.cpu().numpy().astype(np.float32).ravel().tobytes())
