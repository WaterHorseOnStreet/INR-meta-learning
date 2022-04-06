import h5py

with h5py.File('../model_saves/final_latents.h5', 'r') as f:
    keys = list(f.keys())
    dataset = f[keys[0]]
    print(dataset.shape)