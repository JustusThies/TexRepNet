import time
import math
import os.path
import copy
import torch
import torchvision.transforms as transforms
from options.base_options import BaseOptions
from data import CreateDataLoader
from models import create_model
import numpy as np
import trimesh

if __name__ == '__main__':
    # training dataset
    opt = BaseOptions().parse()
    opt.isTrain = False

    # model
    model = create_model(opt)
    model.setup(opt)

    # filenames
    mesh_filename = os.path.join(opt.dataroot, 'high-tess.ply')
    stat_filename = os.path.join(opt.dataroot, 'pos/mesh_stat.txt')
    mesh_output_filename = os.path.join(opt.results_dir, 'test-high-tess.ply')

    # load mesh
    print('Loading mesh...')
    mesh = trimesh.load(mesh_filename)
    vertices = mesh.vertices
    print('\tvertices: ', mesh.vertices.shape)
    n_vertices = mesh.vertices.shape[0]

    # mesh stats (note that it is important to use the same normalization strategy as used for training data generation!)
    print('Loading mesh stats...')
    with open(stat_filename, "r") as f:
        s_cog = f.readline()
        s_scale = f.readline()
        mesh_center_of_gravity = np.fromstring(s_cog, dtype=float, sep=' ')
        mesh_scale = np.fromstring(s_scale, dtype=float, sep=' ')

    print('\tcenter of gravity: ', mesh_center_of_gravity)
    print('\tscale: ', mesh_scale)


    # vertex data (positions are assumed to be in  [-1,1], uvs in [0,1])
    pos_np = (vertices - mesh_center_of_gravity) / mesh_scale
    uvs_np = np.array([[0.0,0.0,0.0], [1.0,1.0,1.0]]) # currently not used

    # convert to an 'image'
    pos_np = np.reshape(pos_np, (1, -1, 3))
    uvs_np = np.reshape(uvs_np, (1, -1, 2))

    # convert to tensors
    POS = transforms.ToTensor()(pos_np.astype(np.float32))
    UV = transforms.ToTensor()(uvs_np.astype(np.float32))
    UV = torch.where(UV > 1.0, torch.ones_like(UV), UV)
    UV = torch.where(UV < 0.0, torch.zeros_like(UV), UV)
    UV = 2.0 * UV - 1.0

    # run inferences on the mesh
    print('Running inference...')
    max_inference_size = 320*256
    if n_vertices < max_inference_size:
        data = { 'UV': UV[None, :,:,:], 'POS': POS[None, :,:,:]}
        model.set_input(data)
        model.test()
        colors = model.fake[0].clone().cpu().float().detach().numpy()[:,0,:]
        colors = np.transpose(colors)
        colors = np.clip(colors*0.5+0.5, 0.0, 1.0)
    else:
        n_chunks = math.ceil(n_vertices / max_inference_size)
        colors_chunk_list = []
        for c in range(n_chunks):
            start_idx = c * max_inference_size
            end_idx = (c+1) * max_inference_size
            if c == (n_chunks -1): end_idx = n_vertices

            data = { 'UV': UV[None, :,:,start_idx:end_idx], 'POS': POS[None, :,:,start_idx:end_idx]}
            model.set_input(data)
            model.test()
            colors_chunk = model.fake[0].clone().cpu().float().detach().numpy()[:,0,:]
            colors_chunk = np.transpose(colors_chunk)
            colors_chunk = np.clip(colors_chunk*0.5+0.5, 0.0, 1.0)
            colors_chunk_list.append(colors_chunk)
        colors=np.concatenate(colors_chunk_list, 0)


    print('colors: ', colors.shape)

    # export mesh with colors
    print('Writing mesh to file...')
    mesh.visual.vertex_colors = 255 * np.ones((n_vertices, 4))
    mesh.visual.vertex_colors[:,0:3] = 255 * colors[:,0:3] 
    mesh.export(mesh_output_filename)