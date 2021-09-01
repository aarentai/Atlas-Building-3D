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

from mtch.RegistrationFunc3DCuda import *
from mtch.SplitEbinMetric3DCuda import *
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
from torch_sym3eig import Sym3Eig as se


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


def energy_ebin(phi, g0, g1, f0, f1, i0, i1, sigma, dim, mask, brain_mask): 
#     input: phi.shape = [3, h, w, d]; g0/g1/f0/f1.shape = [h, w, d, 3, 3]; sigma/dim = scalar; mask.shape = [1, h, w, d]
#     output: scalar
# the phi here is identity
    phi_star_g1 = phi_pullback(phi, g1)
    phi_star_f1 = phi_pullback(phi, f1)# the compose operation in this step uses a couple of thousands MB of memory
    phi_star_i1 = compose_function(i1.unsqueeze(0), phi).squeeze()# the compose operation in this step uses a couple of thousands MB of memory
    E1 = sigma * Squared_distance_Ebin(f0, phi_star_f1, 1./dim, mask)
    # E2 = Squared_distance_Ebin(g0, phi_star_g1, 1./dim, mask)
    E2 = Squared_distance_Ebin(g0, phi_star_g1, 1./dim, mask*brain_mask)
    # E3 = torch.einsum("ijk,ijk->", (i0 - phi_star_i1) ** 2, mask)
    E3 = torch.einsum("ijk,ijk->", (i0 - phi_star_i1) ** 2, (1-mask)*brain_mask)
    print(E2*1e0, E3*3e-8)
    return E1 + E2*1e0 + E3*3e-8


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

    return torch.stack((vx, vy, vz)).to(device=torch.device('cuda'))

        
def metric_matching(gi, gm, ii, im, height, width, depth, mask, brain_mask, iter_num, epsilon, sigma, dim):
    phi_inv = get_idty(height, width, depth)
    phi = get_idty(height, width, depth)
    idty = get_idty(height, width, depth)
    idty.requires_grad_()
    f0 = torch.eye(int(dim)).repeat(height, width, depth, 1, 1)
    f1 = torch.eye(int(dim)).repeat(height, width, depth, 1, 1)
    
    for j in range(iter_num):
        phi_actsg0 = phi_pullback(phi_inv, gi)
        phi_actsf0 = phi_pullback(phi_inv, f0)
        phi_actsi0 = compose_function(ii.unsqueeze(0), phi_inv).squeeze()
        E = energy_ebin(idty, phi_actsg0, gm, phi_actsf0, f1, phi_actsi0, im, sigma, dim, mask, brain_mask) 
        print(E.item())
        if torch.isnan(E):
            raise ValueError('NaN error')
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
    ii = compose_function(ii.unsqueeze(0), phi_inv)
    return gi, ii.squeeze(), phi, phi_inv


def tensor_cleaning(g, det_threshold=1e-11):
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


def get_framework(arr):
      # return np or torch depending on type of array
    # also returns framework name as "numpy" or "torch"
    fw = None
    fw_name = ''
    if type(arr) == np.ndarray:
        fw = np
        fw_name = 'numpy'
    else:
        fw = torch
        fw_name = 'torch'
    return (fw, fw_name)


def batch_cholesky(tens):
    # from https://stackoverflow.com/questions/60230464/pytorch-torch-cholesky-ignoring-exception
    # will get NaNs instead of exception where cholesky is invalid
    fw, fw_name = get_framework(tens)
    L = fw.zeros_like(tens)

    for i in range(tens.shape[-1]):
        for j in range(i+1):
            s = 0.0
        for k in range(j):
            s = s + L[...,i,k] * L[...,j,k]

        L[...,i,j] = fw.sqrt(tens[...,i,i] - s) if (i == j) else \
                      (1.0 / L[...,j,j] * (tens[...,i,j] - s))
    return L


def make_pos_def(tens, mask, small_eval = 0.00005):
  # make any small or negative eigenvalues slightly positive and then reconstruct tensors
  
    fw, fw_name = get_framework(tens)
    if fw_name == 'numpy':
        sym_tens = (tens + tens.transpose(0,1,2,4,3))/2
        evals, evecs = np.linalg.eig(sym_tens)
    else:
        sym_tens = (tens + torch.transpose(tens,3,4))/2
        # evals, evecs = torch.symeig(sym_tens,eigenvectors=True)
        evals, evecs = se.apply(sym_tens.reshape((-1,3,3)))
    evals = evals.reshape((*tens.shape[:-2],3))
    evecs = evecs.reshape((*tens.shape[:-2],3,3))
    #cmplx_evals, cmplx_evecs = fw.linalg.eig(sym_tens)
    #evals = fw.real(cmplx_evals)
    #evecs = fw.real(cmplx_evecs)
    #np.abs(evals, out=evals)
    idx = fw.where(evals < small_eval)
    small_map = fw.where(evals < small_eval,1,0)
    #idx = np.where(evals < 0)
    num_found = 0
    #print(len(idx[0]), 'tensors found with eigenvalues <', small_eval)
    for ee in range(len(idx[0])):
        if mask[idx[0][ee], idx[1][ee], idx[2][ee]]:
            num_found += 1
            # If largest eigenvalue is negative, replace with identity
            eval_2 = (idx[3][ee]+1) % 3
            eval_3 = (idx[3][ee]+2) % 3
        if ((evals[idx[0][ee], idx[1][ee], idx[2][ee], eval_2] < 0) and 
         (evals[idx[0][ee], idx[1][ee], idx[2][ee], eval_3] < 0)):
            evecs[idx[0][ee], idx[1][ee], idx[2][ee]] = fw.eye(3, dtype=tens.dtype)
            evals[idx[0][ee], idx[1][ee], idx[2][ee], idx[3][ee]] = small_eval
        else:
            # otherwise just set this eigenvalue to small_eval
            evals[idx[0][ee], idx[1][ee], idx[2][ee], idx[3][ee]] = small_eval

    print(num_found, 'tensors found with eigenvalues <', small_eval)
    #print(num_found, 'tensors found with eigenvalues < 0')
    mod_tens = fw.einsum('...ij,...jk,...k,...lk->...il',
                       evecs, fw.eye(3, dtype=tens.dtype), evals, evecs)
    #mod_tens = fw.einsum('...ij,...j,...jk->...ik',
    #                     evecs, evals, evecs)

    chol = batch_cholesky(mod_tens)
    idx_nan = torch.where(torch.isnan(chol))
    nan_map = torch.where(torch.isnan(chol),1,0)
    iso_tens = small_eval * torch.eye((3))
    for pt in range(len(idx_nan[0])):
        mod_tens[idx_nan[0][pt],idx_nan[1][pt],idx_nan[2][pt]] = iso_tens
    # if torch.norm(torch.transpose(mod_tens,3,4)-mod_tens)>0:
    #     print('asymmetric')
    mod_tens[:,:,:,1,0]=mod_tens[:,:,:,0,1]
    mod_tens[:,:,:,2,0]=mod_tens[:,:,:,0,2]
    mod_tens[:,:,:,2,1]=mod_tens[:,:,:,1,2]
    return(mod_tens)


def get_euclidean_mean(img_list):
    mean = torch.zeros_like(img_list[0])
    for i in range(len(img_list)):
        mean += img_list[i]

    return mean/len(img_list)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # after switch device, you need restart the script
    torch.cuda.set_device('cuda:1')
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')

    # file_name = []
    file_name = [108222, 102715, 105923, 107422, 100206, 104416]
    input_dir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/BallResults'
    output_dir = '/home/sci/hdai/Projects/Atlas3D/output/BrainAtlasUkfBallMetDominatedBrainMaskAug31'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    height, width, depth = 145,174,145
    sample_num = len(file_name)
    tensor_lin_list, tensor_met_list, mask_list, mask_thresh_list, fa_list, img_list, brain_mask_list = [], [], [], [], [], [], []
    mask_union = torch.zeros(height, width, depth).double().to(device)
    brain_mask_union = torch.zeros(height, width, depth).double().to(device)
    phi_inv_acc_list, phi_acc_list, energy_list = [], [], []
    resume = False
   
    start_iter = 0
    iter_num = 801

    for s in range(len(file_name)):
        # print(f'{s} is processing.')
        tensor_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_scaled_unsmoothed_tensors.nhdr'))
        mask_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_filt_mask.nhdr'))
        brain_mask_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_brain_mask.nhdr'))
        img_np = sitk.GetArrayFromImage(sitk.ReadImage(f'{input_dir}/{file_name[s]}_T1_flip_y.nhdr'))
        tensor_lin_list.append(torch.from_numpy(tensor_np).double().permute(3,2,1,0))
    #     create union of masks
        mask_union += torch.from_numpy(mask_np).double().permute(2,1,0).to(device)
        mask_list.append(torch.from_numpy(mask_np).double().permute(2,1,0))
        brain_mask_list.append(torch.from_numpy(brain_mask_np).double().permute(2,1,0))
        img_list.append(torch.from_numpy(img_np).double().permute(2,1,0))
    #     rearrange tensor_lin to tensor_met
        tensor_met_zeros = torch.zeros(height,width,depth,3,3,dtype=torch.float64)
        tensor_met_zeros[:,:,:,0,0] = tensor_lin_list[s][0]
        tensor_met_zeros[:,:,:,0,1] = tensor_lin_list[s][1]
        tensor_met_zeros[:,:,:,0,2] = tensor_lin_list[s][2]
        tensor_met_zeros[:,:,:,1,0] = tensor_lin_list[s][1]
        tensor_met_zeros[:,:,:,1,1] = tensor_lin_list[s][3]
        tensor_met_zeros[:,:,:,1,2] = tensor_lin_list[s][4]
        tensor_met_zeros[:,:,:,2,0] = tensor_lin_list[s][2]
        tensor_met_zeros[:,:,:,2,1] = tensor_lin_list[s][4]
        tensor_met_zeros[:,:,:,2,2] = tensor_lin_list[s][5]
        # tensor_met_zeros = make_pos_def(tensor_met_zeros, torch.ones((height, width, depth)))
    #     balance the background and subject by rescaling
        tensor_met_zeros = tensor_cleaning(tensor_met_zeros)
        # fa_list.append(fractional_anisotropy(tensor_met_zeros))
        tensor_met_list.append(torch.inverse(tensor_met_zeros))
        # fore_back_adaptor = torch.ones((height,width,depth))
        # fore_back_adaptor = torch.where(torch.det(tensor_met_list[s])>1e1, 5e-4, 1.)
        fore_back_adaptor = torch.where(torch.det(tensor_met_list[s])>1e2, 1e-3, 1.)#
        mask_thresh_list.append(fore_back_adaptor)
        tensor_met_list[s] = torch.einsum('ijk...,lijk->ijk...', tensor_met_list[s], mask_thresh_list[s].unsqueeze(0))
    #     initialize the accumulative diffeomorphism    
        if resume==False:
            print('start from identity')
            phi_inv_acc_list.append(get_idty(height, width, depth))
            phi_acc_list.append(get_idty(height, width, depth))
        else:
            print('start from checkpoint')
            phi_inv_acc_list.append(torch.from_numpy(sio.loadmat(f'{output_dir}/{file_name[s]}_{start_iter-1}_phi_inv.mat')['diffeo']))
            phi_acc_list.append(torch.from_numpy(sio.loadmat(f'{output_dir}/{file_name[s]}_{start_iter-1}_phi.mat')['diffeo']))
            tensor_met_list[s] = phi_pullback(phi_inv_acc_list[s], tensor_met_list[s])
        energy_list.append([])    
        
    # mask_union[mask_union>0] = 1

    print(f'file_name = {file_name}, iter_num = {iter_num}, epsilon = 5e-3')
    print(f'Starting from iteration {start_iter} to iteration {iter_num+start_iter}')

    for i in tqdm(range(start_iter, start_iter+iter_num)):
        G = torch.stack(tuple(tensor_met_list))
        dim, sigma, epsilon, iter_num = 3., 0, 5e-3, 1 # epsilon = 3e-3 for orig tensor
        atlas = get_karcher_mean(G, 1./dim)
        mean_img = get_euclidean_mean(img_list)

        phi_inv_list, phi_list = [], []
        mask_union = ((mask_list[0]+mask_list[1]+mask_list[2]+mask_list[3]+mask_list[4]+mask_list[5])/6).to(device)
        brain_mask_union = ((brain_mask_list[0]+brain_mask_list[1]+brain_mask_list[2]+brain_mask_list[3]+brain_mask_list[4]+brain_mask_list[5])/6).to(device)
        for s in range(sample_num):
            energy_list[s].append(torch.einsum("ijk...,lijk->",[(tensor_met_list[s] - atlas)**2, mask_union.unsqueeze(0)]).item())
            old = tensor_met_list[s]
            tensor_met_list[s], img_list[s], phi, phi_inv = metric_matching(tensor_met_list[s], atlas, img_list[s], mean_img, height, width, depth, mask_union, brain_mask_union, iter_num, epsilon, sigma, dim)
            phi_inv_list.append(phi_inv)
            phi_list.append(phi)
            phi_inv_acc_list[s] = compose_function(phi_inv_acc_list[s], phi_inv_list[s])
            phi_acc_list[s] = compose_function(phi_list[s], phi_acc_list[s])
            mask_list[s] = compose_function(mask_list[s], phi_inv_list[s])
            brain_mask_list[s] = compose_function(brain_mask_list[s], phi_inv_list[s])
    #         if i%1==0:
    #             plot_diffeo(phi_acc_list[s][1:, 50, :, :], step_size=2, show_axis=True)
    #             plot_diffeo(phi_acc_list[s][:2, :, :, 20], step_size=2, show_axis=True)
    #             plot_diffeo(torch.stack((phi_acc_list[s][0, :, 50, :],phi_acc_list[s][2, :, 50, :]),0), step_size=2, show_axis=True)
                
        '''check point'''
        if i%50==0:
            atlas_lin = np.zeros((6,height,width,depth))
            mask_acc = np.zeros((height,width,depth))
            atlas_inv = torch.inverse(atlas)
            atlas_lin[0] = atlas_inv[:,:,:,0,0].cpu()
            atlas_lin[1] = atlas_inv[:,:,:,0,1].cpu()
            atlas_lin[2] = atlas_inv[:,:,:,0,2].cpu()
            atlas_lin[3] = atlas_inv[:,:,:,1,1].cpu()
            atlas_lin[4] = atlas_inv[:,:,:,1,2].cpu()
            atlas_lin[5] = atlas_inv[:,:,:,2,2].cpu()
            for s in range(sample_num):
                sio.savemat(f'{output_dir}/{file_name[s]}_{i}_phi_inv.mat', {'diffeo': phi_inv_acc_list[s].cpu().detach().numpy()})
                sio.savemat(f'{output_dir}/{file_name[s]}_{i}_phi.mat', {'diffeo': phi_acc_list[s].cpu().detach().numpy()})
                sio.savemat(f'{output_dir}/{file_name[s]}_{i}_energy.mat', {'energy': energy_list[s]})
    #             plt.plot(energy_list[s])
                # mask_acc += mask_list[s].cpu().numpy()
            # mask_acc[mask_acc>0]=1
            sitk.WriteImage(sitk.GetImageFromArray(np.transpose(atlas_lin,(3,2,1,0))), f'{output_dir}/atlas_{i}_tens.nhdr')
            sitk.WriteImage(sitk.GetImageFromArray(np.transpose(mask_union.cpu(),(2,1,0))), f'{output_dir}/atlas_{i}_mask.nhdr')
            sitk.WriteImage(sitk.GetImageFromArray(np.transpose(mean_img.cpu(),(2,1,0))), f'{output_dir}/atlas_{i}_img.nhdr')

    atlas_lin = np.zeros((6,height,width,depth))
    # mask_acc = np.zeros((height,width,depth))

    for s in range(sample_num):
        sio.savemat(f'{output_dir}/{file_name[s]}_phi_inv.mat', {'diffeo': phi_inv_acc_list[s].cpu().detach().numpy()})
        sio.savemat(f'{output_dir}/{file_name[s]}_phi.mat', {'diffeo': phi_acc_list[s].cpu().detach().numpy()})
        sio.savemat(f'{output_dir}/{file_name[s]}_energy.mat', {'energy': energy_list[s]})
        
        plt.plot(energy_list[s])
        # mask_acc += mask_list[s].cpu().numpy()

    atlas = torch.inverse(atlas)
    atlas_lin[0] = atlas[:,:,:,0,0].cpu()
    atlas_lin[1] = atlas[:,:,:,0,1].cpu()
    atlas_lin[2] = atlas[:,:,:,0,2].cpu()
    atlas_lin[3] = atlas[:,:,:,1,1].cpu()
    atlas_lin[4] = atlas[:,:,:,1,2].cpu()
    atlas_lin[5] = atlas[:,:,:,2,2].cpu()
    # mask_acc[mask_acc>0]=1
    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(atlas_lin,(3,2,1,0))), f'{output_dir}/atlas_tens.nhdr')
    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(mask_union.cpu(),(2,1,0))), f'{output_dir}/atlas_mask.nhdr')
