import torch
from tqdm import tqdm
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from lazy_imports import itkwidgets
from lazy_imports import itkview
from lazy_imports import interactive
from lazy_imports import ipywidgets
from lazy_imports import pv
from mtch.RegistrationFunc3D import *
from mtch.SplitEbinMetric3D import *
from mtch.GeoPlot import *
# from Packages.disp.vis import show_2d, show_2d_tensors
from disp.vis import vis_tensors, vis_path, disp_scalar_to_file
from disp.vis import disp_vector_to_file, disp_tensor_to_file
from disp.vis import disp_gradG_to_file, disp_gradA_to_file
from disp.vis import view_3d_tensors, tensors_to_mesh
import algo.metricModSolver2d as mms
import algo.geodesic as geo
import algo.euler as euler
import algo.dijkstra as dijkstra

def phi_pullback(phi, g):
    #     input: phi.shape = [3, h, w, d]; g.shape = [h, w, d, 3, 3]
#     output: shape = [h, w, 2, 2]
#     torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    g = g.permute(3, 4, 0, 1, 2)
    idty = get_idty(*g.shape[-3:])
    #     four layers of scalar field, of all 1, all 0, all 1, all 0, where the shape of each layer is g.shape[-2:]?
    eye = torch.eye(3)
    ones = torch.ones(*g.shape[-3:])
    d_phi = get_jacobian_matrix(phi - idty) + torch.einsum("ij,mno->ijmno", eye, ones)
    g_phi = compose_function(g, phi)
    return torch.einsum("ij...,ik...,kl...->...jl", d_phi, g_phi, d_phi)


def energy_ebin(phi, g0, g1, f0, f1, sigma, dim, mask): 
#     input: phi.shape = [3, h, w, d]; g0/g1/f0/f1.shape = [h, w, d, 3, 3]; sigma/dim = scalar; mask.shape = [1, h, w, d]
#     output: scalar
# the phi here is identity
    phi_star_g1 = phi_pullback(phi, g1)
    phi_star_f1 = phi_pullback(phi, f1)# the compose operation in this step uses a couple of thousands MB of memory
    E1 = sigma * Squared_distance_Ebin(f0, phi_star_f1, 1./dim, mask)
    E2 = Squared_distance_Ebin(g0, phi_star_g1, 1./dim, mask)
    return E1 + E2



def energy_L2(phi, g0, g1, f0, f1, sigma, mask): 
#     input: phi.shape = [3, h, w, d]; g0/g1/f0/f1.shape = [h, w, d, 3, 3]; sigma = scalar; mask.shape = [1, h, w, d]
#     output: scalar
    phi_star_g1 = phi_pullback(phi, g1)
    phi_star_f1 = phi_pullback(phi, f1)
    E1 = sigma * torch.einsum("ijk...,lijk->", (f0 - phi_star_f1) ** 2, mask.unsqueeze(0))
    E2 = torch.einsum("ijk...,lijk->", (g0 - phi_star_g1) ** 2, mask.unsqueeze(0))
    # E = E1 + E2
#     del phi_star_g1, phi_star_f1
#     torch.cuda.empty_cache()
    return E1 + E2


def laplace_inverse(u):
#     input: u.shape = [3, h, w, d]
#     output: shape = [3, h, w, d]
    '''
    this function computes the laplacian inverse of a vector field u of size 3 x size_h x size_w x size_d
    '''
    size_h, size_w, size_d = u.shape[-3:]
    idty = get_idty(size_h, size_w, size_d).cpu().numpy()
    lap = 6. - 2. * (np.cos(2. * np.pi * idty[0] / size_h) +
                     np.cos(2. * np.pi * idty[1] / size_w) +
                     np.cos(2. * np.pi * idty[2] / size_d))
    lap[0, 0] = 1.
    lapinv = 1. / lap
    lap[0, 0] = 0.
    lapinv[0, 0] = 1.

    u = u.cpu().detach().numpy()
    fx = np.fft.fftn(u[0])
    fy = np.fft.fftn(u[1])
    fz = np.fft.fftn(u[2])
    fx *= lapinv
    fy *= lapinv
    fz *= lapinv
    vx = torch.from_numpy(np.real(np.fft.ifftn(fx)))
    vy = torch.from_numpy(np.real(np.fft.ifftn(fy)))
    vz = torch.from_numpy(np.real(np.fft.ifftn(fz)))

    return torch.stack((vx, vy, vz))#.to(device=torch.device('cuda'))

        
def metric_matching(gi, gm, height, width, depth, mask, iter_num, epsilon, sigma, dim):
    phi_inv = get_idty(height, width, depth)
    phi = get_idty(height, width, depth)
    idty = get_idty(height, width, depth)
    idty.requires_grad_()
    f0 = torch.eye(int(dim)).repeat(height, width, depth, 1, 1)
    f1 = torch.eye(int(dim)).repeat(height, width, depth, 1, 1)
    
    for j in range(iter_num):
        phi_actsg0 = phi_pullback(phi_inv, gi)
        phi_actsf0 = phi_pullback(phi_inv, f0)
        E = energy_ebin(idty, phi_actsg0, gm, phi_actsf0, f1, sigma, dim, mask) 
        E.backward()
        v = - laplace_inverse(idty.grad)
        with torch.no_grad():
            psi =  idty + epsilon*v  
            psi[0][psi[0] > height - 1] = height - 1
            psi[1][psi[1] > width - 1] = width - 1
            psi[2][psi[2] > depth - 1] = depth - 1
            psi[psi < 0] = 0
            psi_inv =  idty - epsilon*v
            psi_inv[0][psi_inv[0] > height - 1] = height - 1
            psi_inv[1][psi_inv[1] > width - 1] = width - 1
            psi_inv[2][psi_inv[2] > depth - 1] = depth - 1
            psi_inv[psi_inv < 0] = 0
            phi = compose_function(psi, phi)
            phi_inv = compose_function(phi_inv, psi_inv)
            idty.grad.data.zero_()
            
    gi = phi_pullback(phi_inv, gi)
    return gi, phi, phi_inv


def tensor_cleaning(g, det_threshold=1e-15):
    g[torch.det(g)<=det_threshold] = torch.eye((3))
    # Sylvester's criterion https://en.wikipedia.org/wiki/Sylvester%27s_criterion
    psd_map = torch.where(g[...,0,0]>0, 1, 0) + torch.where(torch.det(g[...,:2,:2])>0, 1, 0) + torch.where(torch.det(g)>0, 1, 0)
    nonpsd_idx = torch.where(psd_map!=3)
    # nonpsd_idx = torch.where(torch.isnan(torch.sum(batch_cholesky(g), (3,4))))
    for i in range(len(nonpsd_idx[0])):
        g[nonpsd_idx[0][i], nonpsd_idx[1][i], nonpsd_idx[2][i]] = torch.eye((3))
    return g

    
def fractional_anisotropy(g):
    e, _ = torch.symeig(g)
    lambd1 = e[:,:,:,0]
    lambd2 = e[:,:,:,1]
    lambd3 = e[:,:,:,2]
    mean = torch.mean(e,dim=len(e.shape)-1)
    return torch.sqrt(3.*(torch.pow((lambd1-mean),2)+torch.pow((lambd2-mean),2)+torch.pow((lambd3-mean),2)))/\
    torch.sqrt(2.*(torch.pow(lambd1,2)+torch.pow(lambd2,2)+torch.pow(lambd3,2)))


if __name__ == "__main__":
    # torch.set_default_tensor_type('torch.DoubleTensor')
    file_name = [108222, 102715, 105923, 107422, 100206, 104416]
    input_dir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/BallResults'
    output_dir = '/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallImgMetDirectRegSept10'
    # output_dir = '/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallMetSept11'
    # if not os.path.isdir(output_dir):
    #     os.mkdir(output_dir)
    height, width, depth = 145,174,145

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # after switch device, you need restart the script
    # torch.cuda.set_device('cuda:1')
    # torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')

    diffeo_list = []
    tensor_lin_list = []
    tensor_met_list = []
    mask_list = []
    brain_mask_list = []
    img_list = []

    for s in range(len(file_name)):
        # for s in range(1):
    #     read tensor and mask files
        print(s)
        tensor_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_scaled_orig_tensors_rreg_v2.nhdr'))
        mask_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_orig_mask_rreg.nhdr'))
        img_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_t1_to_reft1_rreg.nhdr'))
        
    #     rearrange tensor_lin to tensor_met
        tensor_met_zeros = torch.zeros(height,width,depth,3,3,dtype=torch.float64)
        tensor_lin_list.append(torch.from_numpy(tensor_np).double().permute(3,2,1,0))
        tensor_met_zeros[:,:,:,0,0] = tensor_lin_list[s][0]
        tensor_met_zeros[:,:,:,0,1] = tensor_lin_list[s][1]
        tensor_met_zeros[:,:,:,0,2] = tensor_lin_list[s][2]
        tensor_met_zeros[:,:,:,1,0] = tensor_lin_list[s][1]
        tensor_met_zeros[:,:,:,1,1] = tensor_lin_list[s][3]
        tensor_met_zeros[:,:,:,1,2] = tensor_lin_list[s][4]
        tensor_met_zeros[:,:,:,2,0] = tensor_lin_list[s][2]
        tensor_met_zeros[:,:,:,2,1] = tensor_lin_list[s][4]
        tensor_met_zeros[:,:,:,2,2] = tensor_lin_list[s][5]
    #     balance the background and subject by rescaling
        tensor_met_zeros = tensor_cleaning(tensor_met_zeros)
            
        diffeo_list.append(torch.from_numpy(sio.loadmat(f'{output_dir}/{file_name[s]}_800_phi_inv.mat')['diffeo']).double().to(device))
        tensor_met_list.append(torch.inverse(tensor_met_zeros).to(device))
        mask_list.append(torch.from_numpy(mask_np).double().permute(2,1,0).to(device))
    #     brain_mask_list.append(torch.from_numpy(brain_mask_np).double().permute(2,1,0).to(device))
        img_list.append(torch.from_numpy(img_np).double().permute(2,1,0).to(device))

        atlas_mask = torch.zeros_like(mask_list[0])
    # atlas_brain_mask = torch.zeros_like(brain_mask_list[0])
    atlas_img = torch.zeros_like(img_list[0])

    for s in range(len(file_name)):
            
        tensor_met_list[s] = phi_pullback(diffeo_list[s], tensor_met_list[s])
    #     tensor_met_list[s] = tensor_cleaning(tensor_met_list[s])
        atlas_mask += compose_function(mask_list[s].unsqueeze(0), diffeo_list[s]).squeeze()
    #     atlas_brain_mask += compose_function(brain_mask_list[s].unsqueeze(0), diffeo_list[s]).squeeze()
        atlas_img += compose_function(img_list[s].unsqueeze(0), diffeo_list[s]).squeeze()

    G = torch.stack(tuple(tensor_met_list))
#     G[0,64,83,100] = torch.eye(3)
#     G[1,64,83,100] = torch.eye(3)
#     G[0,72,41,38] = torch.eye(3)
#     G[1,72,41,38] = torch.eye(3)
#     G[0,56,77,40] = torch.eye(3)
#     G[1,56,77,40] = torch.eye(3)
    # G[torch.abs(torch.det(G)-1)<=2e-4] = torch.eye((3))

    atlas = get_karcher_mean(G, 1./3)
    atlas_mask = atlas_mask/6
    atlas_img = atlas_img/6

    atlas_lin = np.zeros((6,height,width,depth))
    atlas_inv = torch.inverse(atlas)
    atlas_lin[0] = atlas_inv[:,:,:,0,0].cpu()
    atlas_lin[1] = atlas_inv[:,:,:,0,1].cpu()
    atlas_lin[2] = atlas_inv[:,:,:,0,2].cpu()
    atlas_lin[3] = atlas_inv[:,:,:,1,1].cpu()
    atlas_lin[4] = atlas_inv[:,:,:,1,2].cpu()
    atlas_lin[5] = atlas_inv[:,:,:,2,2].cpu()

    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(atlas_lin,(3,2,1,0))), f'{output_dir}/atlas_tens_phi_inv_img_met_rreg_800.nhdr')
    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(atlas_mask.cpu().numpy(),(2,1,0))), f'{output_dir}/atlas_mask_phi_inv_img_met_rreg_800.nhdr')
    # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(atlas_brain_mask.cpu().numpy(),(2,1,0))), f'{output_dir}/atlas_brain_mask_phi_inv_800.nhdr')
    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(atlas_img.cpu().numpy(),(2,1,0))), f'{output_dir}/atlas_img_phi_inv_img_met_rreg_800.nhdr')

    # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(atlas_lin,(3,2,1,0))), f'{output_dir}/atlas_tens_phi_inv_met_rreg_800.nhdr')
    # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(atlas_mask.cpu().numpy(),(2,1,0))), f'{output_dir}/atlas_mask_phi_inv_met_rreg_800.nhdr')
    # # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(atlas_brain_mask.cpu().numpy(),(2,1,0))), f'{output_dir}/atlas_brain_mask_phi_inv_800.nhdr')
    # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(atlas_img.cpu().numpy(),(2,1,0))), f'{output_dir}/atlas_img_phi_inv_met_rreg_800.nhdr')