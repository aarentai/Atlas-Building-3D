import os
from algo import geodesic
from lazy_imports import np
import scipy.io as sio
from lazy_imports import torch
from data.io import ReadTensors, ReadScalars, ReadVectors, WriteTensorNPArray, WriteScalarNPArray
from util.diffeo import coord_register_batch_3d, compose_function_3d, get_idty_3d
import gzip
import _pickle as pickle
import math
from functools import partial
import platform
import pathlib
import nibabel as nib

###################
# Begin patch for getting large amount of data back from pool.map
# Error received was:
#  File "/usr/lib64/python3.6/multiprocessing/connection.py", line 393, in _send_bytes
#    header = struct.pack("!i", n)
# struct.error: 'i' format requires -2147483648 <= number <= 2147483647
#
# patch from https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647
###################
import functools
import logging
import struct
import sys

logger = logging.getLogger()

def patch_mp_connection_bpo_17560():
    """Apply PR-10305 / bpo-17560 connection send/receive max size update

    See the original issue at https://bugs.python.org/issue17560 and 
    https://github.com/python/cpython/pull/10305 for the pull request.

    This only supports Python versions 3.3 - 3.7, this function
    does nothing for Python versions outside of that range.

    """
    patchname = "Multiprocessing connection patch for bpo-17560"
    if not (3, 3) < sys.version_info < (3, 8):
        logger.info(
            patchname + " not applied, not an applicable Python version: %s",
            sys.version
        )
        return

    from multiprocessing.connection import Connection

    orig_send_bytes = Connection._send_bytes
    orig_recv_bytes = Connection._recv_bytes
    if (
        orig_send_bytes.__code__.co_filename == __file__
        and orig_recv_bytes.__code__.co_filename == __file__
    ):
        logger.info(patchname + " already applied, skipping")
        return

    @functools.wraps(orig_send_bytes)
    def send_bytes(self, buf):
        n = len(buf)
        if n > 0x7fffffff:
            pre_header = struct.pack("!i", -1)
            header = struct.pack("!Q", n)
            self._send(pre_header)
            self._send(header)
            self._send(buf)
        else:
            orig_send_bytes(self, buf)

    @functools.wraps(orig_recv_bytes)
    def recv_bytes(self, maxsize=None):
        buf = self._recv(4)
        size, = struct.unpack("!i", buf.getvalue())
        if size == -1:
            buf = self._recv(8)
            size, = struct.unpack("!Q", buf.getvalue())
        if maxsize is not None and size > maxsize:
            return None
        return self._recv(size)

    Connection._send_bytes = send_bytes
    Connection._recv_bytes = recv_bytes

    logger.info(patchname + " applied")

patch_mp_connection_bpo_17560()
###################
# End patch code
###################

import multiprocessing

if multiprocessing.connection.Connection._send_bytes.__code__.co_filename == __file__:
  print("Think patch worked")
else:
  print("Patch not detected")

def apply_transform_to_img(in_img, diffeo):
  # use phi_inv and compose_function to take subj image into atlas space
  # use phi and compose_function to take atlas image into subj space
  try:
    torch.set_default_tensor_type('torch.DoubleTensor')
    print('Applying transform to image')

    with torch.no_grad():
      img_tfm_space = compose_function_3d(torch.from_numpy(in_img), torch.from_numpy(diffeo)).detach().numpy()

  except Exception as err:
    print('Caught', err, 'while applying transform to image')
  return(img_tfm_space)
  

def get_paths(tens=None, mask=None, start_coords=None, Gamma1=None, Gamma2=None, Gamma3=None, fileprefix=None):
  atlas_geos = []
  init_velocity = None

  max_coords_at_once = 4000 #161000

  print('Computing ', len(start_coords), 'geodesic paths.')

  atlas_geos = []
  num_paths = len(start_coords)
  num_blocks = math.floor(num_paths / max_coords_at_once)

  geo_delta_t = 0.1#0.01#0.005
  geo_iters = 3000 # 22000 for Kris annulus(delta_t=0.005), 32000 for cubic (delta_t=0.005)

  for block in range(num_blocks):
    geox, geoy, geoz = geodesic.batch_geodesicpath_3d(tens, mask,
                                                      start_coords[block*max_coords_at_once:(block+1)*max_coords_at_once], init_velocity,
                                                      geo_delta_t, iter_num=geo_iters, both_directions=True,
                                                      Gamma1=Gamma1, Gamma2=Gamma2, Gamma3=Gamma3)
    atlas_geos.append((geox, geoy, geoz))
  # now get last partial block
  geox, geoy, geoz = geodesic.batch_geodesicpath_3d(tens, mask,
                                                    start_coords[num_blocks*max_coords_at_once:], init_velocity,
                                                    geo_delta_t, iter_num=geo_iters, both_directions=True,
                                                    Gamma1=Gamma1, Gamma2=Gamma2, Gamma3=Gamma3)
  atlas_geos.append((geox, geoy, geoz))

  # Write to file
  fname = f'{fileprefix}.pkl.gz'
  with gzip.open(fname,'wb') as f:
    print('writing results to file:', fname)
    pickle.dump(atlas_geos, f)

  del atlas_geos
  return([])
# end get_paths

def get_paths_subj_space(tens=None, mask=None, start_coords=None, Gamma1=None, Gamma2=None, Gamma3=None, fileprefix_subjspace=None,
                         diffeo=None, fileprefix_atlasspace=None):
  atlas_geos = []
  init_velocity = None

  max_coords_at_once = 4000 #161000

  print('Computing ', len(start_coords), 'geodesic paths.')

  subj_geos = []
  atlas_geos = []
  num_paths = len(start_coords)
  num_blocks = math.floor(num_paths / max_coords_at_once)

  geo_delta_t = 0.1#0.01#0.005
  geo_iters = 3000 # 22000 for Kris annulus(delta_t=0.005), 32000 for cubic (delta_t=0.005)

  
  for block in range(num_blocks):
    geox, geoy, geoz = geodesic.batch_geodesicpath_3d(tens, mask,
                                                      start_coords[block*max_coords_at_once:(block+1)*max_coords_at_once], init_velocity,
                                                      geo_delta_t, iter_num=geo_iters, both_directions=True,
                                                      Gamma1=Gamma1, Gamma2=Gamma2, Gamma3=Gamma3)
    subj_geos.append((geox, geoy, geoz))
    num_p_in_block = len(geox)
    atlasx = []
    atlasy = []
    atlasz = []
    for p in range(num_p_in_block):
      ax, ay, az = coord_register_batch_3d(geox[p], geoy[p], geoz[p], diffeo)
      atlasx.append(ax)
      atlasy.append(ay)
      atlasz.append(az)
    atlas_geos.append((atlasx,atlasy,atlasz))

  # now get last partial block
  geox, geoy, geoz = geodesic.batch_geodesicpath_3d(tens, mask,
                                                    start_coords[num_blocks*max_coords_at_once:], init_velocity,
                                                    geo_delta_t, iter_num=geo_iters, both_directions=True,
                                                    Gamma1=Gamma1, Gamma2=Gamma2, Gamma3=Gamma3)
  subj_geos.append((geox, geoy, geoz))
  num_p_in_block = len(geox)
  atlasx = []
  atlasy = []
  atlasz = []
  for p in range(num_p_in_block):
    ax, ay, az = coord_register_batch_3d(geox[p], geoy[p], geoz[p], diffeo)
    atlasx.append(ax)
    atlasy.append(ay)
    atlasz.append(az)
  atlas_geos.append((atlasx,atlasy,atlasz))


  # Write to file
  fname = f'{fileprefix_subjspace}.pkl.gz'
  with gzip.open(fname,'wb') as f:
    print('writing results to file:', fname)
    pickle.dump(subj_geos, f)

  fname = f'{fileprefix_atlasspace}.pkl.gz'
  with gzip.open(fname,'wb') as f:
    print('writing results to file:', fname)
    pickle.dump(atlas_geos, f)

  del subj_geos
  del atlas_geos
  return([])
# end get_paths_subj_space

def collect_result(result):
  # Right now, empty list expected.
  print('collected result')

def compute_geodesics(atlas_tens_4_path, mask, start_coords, fileprefix, pool):
  # Since calling batch_geodesic_3d multiple times, precompute gammas
  Gamma1, Gamma2, Gamma3 = geodesic.compute_gammas_3d(atlas_tens_4_path, mask)

  ars = []
  #get_paths_tens = partial(get_paths, tens=atlas_tens_4_path, mask=mask, start_coords=start_coords,
  #                         Gamma1=Gamma1, Gamma2=Gamma2, Gamma3=Gamma3, fileprefix=fileprefix)
  for sc, fp in zip(start_coords, fileprefix):
    get_paths_tens = partial(get_paths, tens=atlas_tens_4_path, mask=mask, start_coords=sc,
                             Gamma1=Gamma1, Gamma2=Gamma2, Gamma3=Gamma3, fileprefix=fp)

    ar = pool.apply_async(get_paths_tens, args=(), callback=collect_result)
    ars.append(ar)
  return(ars)
# end compute_geodesics

def compute_geodesics_subj_space(atlas_tens_4_path, mask, start_coords, fileprefix_subjspace, diffeo, fileprefix_atlasspace, pool):
  # Since calling batch_geodesic_3d multiple times, precompute gammas
  Gamma1, Gamma2, Gamma3 = geodesic.compute_gammas_3d(atlas_tens_4_path, mask)

  ars = []
  #get_paths_tens = partial(get_paths_subj_space, tens=atlas_tens_4_path, mask=mask, start_coords=start_coords,
  #                         Gamma1=Gamma1, Gamma2=Gamma2, Gamma3=Gamma3, fileprefix_subjspace=fileprefix_subjspace,
  #                         diffeo_fname=diffeo_fname, fileprefix_atlasspace=fileprefix_atlasspace)
  for sc, fps, fpa in zip(start_coords, fileprefix_subjspace, fileprefix_atlasspace):
    get_paths_tens = partial(get_paths_subj_space, tens=atlas_tens_4_path, mask=mask, start_coords=sc,
                             Gamma1=Gamma1, Gamma2=Gamma2, Gamma3=Gamma3, fileprefix_subjspace=fps,
                             diffeo=diffeo, fileprefix_atlasspace=fpa)

    ar = pool.apply_async(get_paths_tens, args=(), callback=collect_result)
    ars.append(ar)
  return(ars)
# end compute_geodesics_subj_space

def create_seeds(mask, filename, max_coords=None,offsets=None):
  xseedmin = 0
  xseedmax = mask.shape[0]-1
  yseedmin = 0
  yseedmax = mask.shape[1]-1
  zseedmin = 0
  zseedmax = mask.shape[2]-1
  vals = np.where(mask > 0)
  xseedmin = np.min(vals[0])
  xseedmax = np.max(vals[0])+1
  yseedmin = np.min(vals[1])
  yseedmax = np.max(vals[1])+1
  zseedmin = np.min(vals[2])
  zseedmax = np.max(vals[2])+1
  
  start_coords = gen_seeds([xseedmin, xseedmax],[yseedmin, yseedmax],[zseedmin, zseedmax],
                                 mask, max_coords, offsets)

  if filename:
    with open(filename, 'wb') as f:
      print('writing seeds to file:', filename)
      pickle.dump(start_coords, f)
    
  return(start_coords)
# end create_seeds


def gen_seeds(xrng, yrng, zrng, mask, max_coords=None,offsets=None):
  if offsets is None:
    offsets = [0]

  numx = int(xrng[1]-xrng[0])*10
  numy = int(yrng[1]-yrng[0])*10
  numz = int(zrng[1]-zrng[0])*10
  
  if max_coords is None:
    max_coords = numx * numy * numz
    
  start_coords = []
#  for offs in offsets:
#    for xx in np.linspace(xrng[0], xrng[1], num=numx):
#      for yy in np.linspace(yrng[0], yrng[1], num=numy):
#        for zz in np.linspace(zrng[0], zrng[1], num=numz):
  max_times_through = 10
  num_times_through = 0
  while num_times_through < max_times_through:
    pts = np.random.uniform(0,1,size=[numx,numy,numz,3])
    for xx in range(numx):
      for yy in range(numy):
        for zz in range(numz):
          xcoord = xrng[0] + pts[xx,yy,zz,0]*(xrng[1]-xrng[0])
          ycoord = yrng[0] + pts[xx,yy,zz,1]*(yrng[1]-yrng[0])
          zcoord = zrng[0] + pts[xx,yy,zz,2]*(zrng[1]-zrng[0])
          if mask[math.floor(xcoord),math.floor(ycoord),math.floor(zcoord)] > 0.5:
            start_coords.append([xcoord,ycoord,zcoord])
            if len(start_coords) > max_coords:
              return(start_coords)
    num_times_through += 1
  print('Took', num_times_through, 'attempts to find', max_coords,
        'seeds. Still found only', len(start_coords), 'seeds.  Returning the seeds we found. xrng:', xrng,
        'yrng:', yrng,'zrng:', zrng, 'mask shape:', mask.shape, 'num mask pts > 0.5:', np.sum(mask>0.5))
  return(start_coords)
# end gen_seeds

def convert_seeds(atlas_start_coords, diffeo, fname):
  atlas_xcoords = np.array([c[0] for c in atlas_start_coords])
  atlas_ycoords = np.array([c[1] for c in atlas_start_coords])
  atlas_zcoords = np.array([c[2] for c in atlas_start_coords])
  
  subjx, subjy, subjz = coord_register_batch_3d(atlas_xcoords,atlas_ycoords,atlas_zcoords,diffeo)
  subj_start_coords = [ [x,y,z] for x,y,z in zip(subjx,subjy,subjz) ]

  with open(fname, 'wb') as f:
    print('writing subj seeds to file:', fname)
    pickle.dump(subj_start_coords, f)

  return(subj_start_coords)
# end convert_seeds


if __name__ == "__main__":
  # Create seed points in atlas space
  # Then transform seed points to subj space to compute geos there
  max_atlas_coords_total = 10000 #100000
  max_seed_coords_total = 1000 #1000
  steps = [0.00,0.33,0.67]
  #steps = [0.00,0.33]
  offsets = []
  for xoffs in steps:
    for yoffs in steps:
      for zoffs in steps:
        #if ((xoffs == 0.33 and yoffs == 0.33 and zoffs == 0.67) or
        #    (xoffs == 0.33 and yoffs == 0.67 and zoffs == 0.33) or
        #    (xoffs == 0.67 and yoffs == 0.67 and zoffs == 0.33)):
        offsets.append((xoffs,yoffs,zoffs))
  
  host = platform.node()
  if ('ardashir' in host) or ('lakota' in host) or ('kourosh' in host):
    pool = multiprocessing.Pool(48) # 8 or 16 for atlas, 48 for subjs
  elif 'beast' in host:
    pool = multiprocessing.Pool(6) # split into 4 batches to avoid hitting swap space
  else:
    print('Unknown host,', host, ' defaulting to 6 processes.  Increase if on capable host')
    pool = multiprocessing.Pool(6) # split into 4 batches to avoid hitting swap space
    
  hd_atlasname = 'BrainAtlasUkfBallImgMetDirectRegSept10'
  dtitk_atlasname = 'DTITKReg'
  kc_atlasname = 'Ball_met_img_rigid_6subj'
  indir = '/usr/sci/projects/HCP/Kris/NSFCRCNS/prepped_UKF_data_with_grad_dev/'
  atlasdir = f'/home/sci/hdai/Projects/Atlas3D/output/{hd_atlasname}/'
  dtitkatlasdir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/BallResults/{dtitk_atlasname}/'
  outatlasdir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/{hd_atlasname}/'
  outatlassubjspacedir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/{hd_atlasname}/subjspace/'
  outdtitkatlasdir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/atlases/{dtitk_atlasname}/'
  out_tract_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/{kc_atlasname}_and_subj_tracts/'
  out_atlas_tract_dir = f'{out_tract_dir}atlas_tracts/'
  out_dtitk_atlas_tract_dir = f'{out_tract_dir}dtitk_atlas_tracts/'
  out_subj_tract_dir = f'{out_tract_dir}subj_tracts/'
  out_subj_tracts_to_atlas_space_dir = f'{out_tract_dir}subj_tracts_deformed_to_atlas_space/'
  out_subj_tracts_in_atlas_space_dir = f'{out_tract_dir}subj_tracts_computed_in_atlas_space/'
  out_dtitk_subj_tract_dir = f'{out_tract_dir}dtitk_subj_tracts/'
  out_subj_tracts_to_dtitkatlas_space_dir = f'{out_tract_dir}subj_tracts_deformed_to_dtitk_atlas_space/'
  out_subj_tracts_in_dtitkatlas_space_dir = f'{out_tract_dir}subj_tracts_computed_in_dtitk_atlas_space/'
  outsubjdir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/UKF_experiments/BallResults/'

  pathlib.Path(out_tract_dir).mkdir(exist_ok=True) 
  pathlib.Path(out_atlas_tract_dir).mkdir(exist_ok=True) 
  pathlib.Path(out_dtitk_atlas_tract_dir).mkdir(exist_ok=True) 
  pathlib.Path(out_subj_tract_dir).mkdir(exist_ok=True) 
  pathlib.Path(out_dtitk_subj_tract_dir).mkdir(exist_ok=True) 
  pathlib.Path(out_subj_tracts_to_atlas_space_dir).mkdir(exist_ok=True) 
  pathlib.Path(out_subj_tracts_in_atlas_space_dir).mkdir(exist_ok=True) 
  pathlib.Path(out_subj_tracts_to_dtitkatlas_space_dir).mkdir(exist_ok=True) 
  pathlib.Path(out_subj_tracts_in_dtitkatlas_space_dir).mkdir(exist_ok=True) 

  subjs = []
  subjs.append('105923')
  subjs.append('108222')
  subjs.append('102715')
  subjs.append('100206')
  subjs.append('104416')
  subjs.append('107422')
  bval = 'all'
  
  region_masks = []
  single_masks = []
  region_seeds = {}
  single_seeds = {}
  all_ars = []

  region_masks.append('CST_v3')
  region_masks.append('Cing_cor_v3')
  region_masks.append('SLF_v3')
  single_masks.append('AC_v3_seed')
  single_masks.append('CC_v3_seed')
  single_masks.append('CC_genu_thick_seed')
  single_masks.append('CC_genu_thin_seed')
  niters = 800
  if kc_atlasname == 'Ball_joint_img_6subj':
    niters = 500
  elif kc_atlasname == 'Ball_met_dom_6subj':
    niters = 700
  elif kc_atlasname == 'Ball_met_dom_brainmask_iter0_6subj':
    niters = 0
  elif kc_atlasname == 'Ball_met_img_brainmask_iter0_6subj':
    niters = 0

  if ((kc_atlasname == 'Ball_met_img_brainmask_6subj') or
      (kc_atlasname == 'Ball_scaledorig_6subj') or
      (kc_atlasname == 'Ball_met_img_rigid_6subj')):
    print('Computing geodesics for', kc_atlasname, 'using tensors from', atlasdir + f'atlas_tens_phi_inv_img_met_rreg_800.nhdr')
    #atlas_tens = ReadTensors(atlasdir + f'atlas_tens_phi_inv.nhdr')
    #atlas_mask = ReadScalars(atlasdir + f'atlas_mask_phi_inv.nhdr')
    atlas_tens = ReadTensors(atlasdir + f'atlas_tens_phi_inv_img_met_rreg_800.nhdr')
    atlas_mask = ReadScalars(atlasdir + f'atlas_mask_phi_inv_img_met_rreg_800.nhdr')
  else:
    print('Computing geodesics for', kc_atlasname, 'using tensors from', atlasdir + f'atlas_{niters}_tens.nhdr')
    atlas_tens = ReadTensors(atlasdir + f'atlas_{niters}_tens.nhdr')
    atlas_mask = ReadScalars(atlasdir + f'atlas_{niters}_mask.nhdr')
  atlas_mask[atlas_mask < 0.3] = 0
  atlas_mask[atlas_mask > 0] = 1
  end_mask = atlas_mask
  atlas_tens_4_path = np.transpose(atlas_tens,(3,0,1,2))

  # Now compute tracts for dtitk atlas
  #dtitk_tens_nib = nib.load(f'{dtitkatlasdir}/dtitk_atlas_aff_diffeo_orig_dims.nii.gz')
  dtitk_tens_nib = nib.load(f'{dtitkatlasdir}/mean_diffeomorphic_initial6_orig_dims_scaled.nii.gz')
  dtitk_tens = dtitk_tens_nib.get_fdata().squeeze()
  print('Swapping index 2 and 3 in tensor to match DTITK expectations.',  'DO NOT DO THIS if starting w/ NIFTI file w/ expected tensor ordering!')
  # Note dtitk is also off by one voxel in the x direction.  dtitk[0] = atlas[1]
  dtitk_swap = dtitk_tens.copy()
  #dtitk_swap[1:] = dtitk_tens[0:-1]
  dtitk_swap[:,:,:,2] = dtitk_tens[:,:,:,3]
  dtitk_swap[:,:,:,3] = dtitk_tens[:,:,:,2]


  dtitk_atlas_mask = ReadScalars(f'{dtitkatlasdir}/mean_diffeomorphic_initial6_orig_dims_tr_mask.nii.gz')
  dtitk_atlas_mask[dtitk_atlas_mask < 0.3] = 0
  dtitk_atlas_mask[dtitk_atlas_mask > 0] = 1
  end_dtitk_mask = dtitk_atlas_mask
  
  dtitk_tens_4_path = np.transpose(dtitk_swap,(3,0,1,2))

  atlas_coord_list = []
  atlas_fpref_list = []
  dtitk_atlas_coord_list = []
  dtitk_atlas_fpref_list = []
  subj_in_atlas_coord_list = {}
  subj_in_atlas_fpref_list = {}
  subj_in_dtitk_atlas_coord_list = {}
  subj_in_dtitk_atlas_fpref_list = {}
  subj_coord_list = {}
  subj_fpref_list = {}
  subj_to_atlas_fpref_list = {}
  subj_dtitk_coord_list = {}
  subj_dtitk_fpref_list = {}
  subj_to_dtitk_atlas_fpref_list = {}
  for subj in subjs:
    subj_in_atlas_coord_list[subj] = []
    subj_in_atlas_fpref_list[subj] = []
    subj_in_dtitk_atlas_coord_list[subj] = []
    subj_in_dtitk_atlas_fpref_list[subj] = []
    subj_coord_list[subj] = []
    subj_fpref_list[subj] = []
    subj_to_atlas_fpref_list[subj] = []
    subj_dtitk_coord_list[subj] = []
    subj_dtitk_fpref_list[subj] = []
    subj_to_dtitk_atlas_fpref_list[subj] = []
        
  # Set up seed points for everything
  fname = f'{out_tract_dir}atlas_seeds.pkl'
  atlas_start_coords = create_seeds(atlas_mask, fname, max_atlas_coords_total, offsets)

  fname = f'{out_tract_dir}dtitk_atlas_seeds.pkl'
  dtitk_atlas_start_coords = create_seeds(dtitk_atlas_mask, fname, max_atlas_coords_total, offsets)

  atlas_coord_list.append(atlas_start_coords)
  atlas_fpref_list.append(f'{out_atlas_tract_dir}{kc_atlasname}_geos_all_geodesics')
  dtitk_atlas_coord_list.append(dtitk_atlas_start_coords)
  dtitk_atlas_fpref_list.append(f'{out_dtitk_atlas_tract_dir}dtitk_all_geos_geodesics')
  
  #ar = compute_geodesics(atlas_tens_4_path, atlas_mask, atlas_start_coords, f'{out_atlas_tract_dir}{kc_atlasname}_geos_all_geodesics', pool)
  #ars.append(ar)
  #ar = compute_geodesics(dtitk_tens_4_path, atlas_mask, atlas_start_coords, f'{out_dtitk_atlas_tract_dir}dtitk_all_geos_geodesics',pool)
  #ars.append(ar)

  for rmask in region_masks:
    mask1_start = ReadScalars(outatlasdir + f'/{rmask}_hemi1_start.nhdr')
    mask2_start = ReadScalars(outatlasdir + f'/{rmask}_hemi2_start.nhdr')

    fname = f'{out_tract_dir}atlas_seeds_{rmask}_hemi1.pkl'
    start_coords = create_seeds(mask1_start, fname, max_seed_coords_total, offsets)
    region_seeds[f'{rmask}_hemi1'] = start_coords

    dtitk_mask1_start = ReadScalars(outdtitkatlasdir + f'/{rmask}_hemi1_start.nhdr')
    dtitk_mask2_start = ReadScalars(outdtitkatlasdir + f'/{rmask}_hemi2_start.nhdr')

    fname = f'{out_tract_dir}dtitk_atlas_seeds_{rmask}_hemi1.pkl'
    dtitk_start_coords = create_seeds(dtitk_mask1_start, fname, max_seed_coords_total, offsets)
    region_seeds[f'dtitk_{rmask}_hemi1'] = dtitk_start_coords

    atlas_coord_list.append(start_coords)
    atlas_fpref_list.append(f'{out_atlas_tract_dir}{kc_atlasname}_geos_{rmask}_hemi1_geodesics')
    dtitk_atlas_coord_list.append(dtitk_start_coords)
    dtitk_atlas_fpref_list.append(f'{out_dtitk_atlas_tract_dir}dtitk_geos_{rmask}_hemi1_geodesics')
    #ar = compute_geodesics(atlas_tens_4_path, end_mask, start_coords, f'{out_atlas_tract_dir}{kc_atlasname}_geos_{rmask}_hemi1_geodesics', pool)
    #ars.append(ar)
    #ar = compute_geodesics(dtitk_tens_4_path, end_mask, start_coords, f'{out_dtitk_atlas_tract_dir}dtitk_geos_{rmask}_hemi1_geodesics', pool)
    #ars.append(ar)

    fname = f'{out_tract_dir}atlas_seeds_{rmask}_hemi2.pkl'
    start_coords = create_seeds(mask2_start, fname, max_seed_coords_total, offsets)
    region_seeds[f'{rmask}_hemi2'] = start_coords

    fname = f'{out_tract_dir}dtitk_atlas_seeds_{rmask}_hemi2.pkl'
    dtitk_start_coords = create_seeds(dtitk_mask2_start, fname, max_seed_coords_total, offsets)
    region_seeds[f'dtitk_{rmask}_hemi2'] = dtitk_start_coords


    atlas_coord_list.append(start_coords)
    atlas_fpref_list.append(f'{out_atlas_tract_dir}{kc_atlasname}_geos_{rmask}_hemi2_geodesics')
    dtitk_atlas_coord_list.append(dtitk_start_coords)
    dtitk_atlas_fpref_list.append(f'{out_dtitk_atlas_tract_dir}dtitk_geos_{rmask}_hemi2_geodesics')
    #ar = compute_geodesics(atlas_tens_4_path, end_mask, start_coords, f'{out_atlas_tract_dir}{kc_atlasname}_geos_{rmask}_hemi2_geodesics', pool)
    #ars.append(ar)
    #ar = compute_geodesics(dtitk_tens_4_path, end_mask, start_coords, f'{out_dtitk_atlas_tract_dir}dtitk_geos_{rmask}_hemi2_geodesics', pool)
    #ars.append(ar)
    
  for smask in single_masks:
    mask = ReadScalars(outatlasdir + f'{smask}.nhdr')
    fname = f'{out_tract_dir}atlas_seeds_{smask}.pkl'
    start_coords = create_seeds(mask, fname, max_seed_coords_total, offsets)
    single_seeds[f'{smask}'] = start_coords

    dtitk_mask = ReadScalars(outdtitkatlasdir + f'{smask}.nhdr')
    fname = f'{out_tract_dir}dtitk_atlas_seeds_{smask}.pkl'
    dtitk_start_coords = create_seeds(dtitk_mask, fname, max_seed_coords_total, offsets)
    single_seeds[f'dtitk_{smask}'] = dtitk_start_coords

    atlas_coord_list.append(start_coords)
    atlas_fpref_list.append(f'{out_atlas_tract_dir}{kc_atlasname}_geos_{smask}_geodesics')
    dtitk_atlas_coord_list.append(dtitk_start_coords)
    dtitk_atlas_fpref_list.append(f'{out_dtitk_atlas_tract_dir}dtitk_geos_{smask}_geodesics')
    #ar = compute_geodesics(atlas_tens_4_path, end_mask, start_coords, f'{out_atlas_tract_dir}{kc_atlasname}_geos_{smask}_geodesics', pool)
    #ars.append(ar)
    #ar = compute_geodesics(dtitk_tens_4_path, end_mask, start_coords, f'{out_dtitk_atlas_tract_dir}dtitk_geos_{smask}_geodesics', pool)
    #ars.append(ar)

  ars = compute_geodesics(atlas_tens_4_path, end_mask, atlas_coord_list, atlas_fpref_list, pool)
  for ar in ars:
    all_ars.append(ar)
  ars = compute_geodesics(dtitk_tens_4_path, end_dtitk_mask, dtitk_atlas_coord_list, dtitk_atlas_fpref_list, pool)
  for ar in ars:
    all_ars.append(ar)

  
  for subj in subjs:
    subjname = subj + f'_{bval}_'

    # Compute tracts for subjs in atlas space
    subj_tens = ReadTensors(f'{outatlasdir}/{subj}_scaled_orig_tensors_rreg_atlas_space.nhdr')
    #subj_mask = ReadScalars(f'{outatlasdir}/{subj}_FA_mask_0.20_rreg_atlas_space.nhdr')
    subj_tens_4_path = np.transpose(subj_tens,(3,0,1,2))
    subj_dtitk_tens = nib.load(f'{dtitkatlasdir}/{subj}_padded_aff_aff_diffeo_orig_dims_scaled.nii.gz').get_fdata().squeeze()
    print('Swapping index 2 and 3 in tensor to match DTITK expectations.',  'DO NOT DO THIS if starting w/ NIFTI file w/ expected tensor ordering!')
    # Note dtitk is also off by one voxel in the x direction.  dtitk[0] = atlas[1]
    subj_dtitk_swap = subj_dtitk_tens.copy()
    #dtitk_swap[1:] = dtitk_tens[0:-1]
    subj_dtitk_swap[:,:,:,2] = subj_dtitk_tens[:,:,:,3]
    subj_dtitk_swap[:,:,:,3] = subj_dtitk_tens[:,:,:,2]
    subj_dtitk_tens_4_path = np.transpose(subj_dtitk_swap,(3,0,1,2))

    #subj_mask[subj_mask < 0.3] = 0
    #subj_mask[subj_mask > 0] = 1
    subj_in_atlas_coord_list[subj].append(atlas_start_coords)
    subj_in_atlas_fpref_list[subj].append(f'{out_subj_tracts_in_atlas_space_dir}{subjname}_all_geos_geodesics')
    subj_in_dtitk_atlas_coord_list[subj].append(dtitk_atlas_start_coords)
    subj_in_dtitk_atlas_fpref_list[subj].append(f'{out_subj_tracts_in_dtitkatlas_space_dir}{subjname}_all_geos_geodesics')
    #ar = compute_geodesics(subj_tens_4_path, subj_mask, atlas_start_coords, f'{out_subj_tracts_in_atlas_space_dir}{subjname}_all_geos_geodesics', pool)
    #ars.append(ar)

    for rmask in region_masks:
      subj_in_atlas_coord_list[subj].append(region_seeds[f'{rmask}_hemi1'])
      subj_in_atlas_fpref_list[subj].append(f'{out_subj_tracts_in_atlas_space_dir}{subjname}_geos_{rmask}_hemi1_geodesics')
      subj_in_atlas_coord_list[subj].append(region_seeds[f'{rmask}_hemi2'])
      subj_in_atlas_fpref_list[subj].append(f'{out_subj_tracts_in_atlas_space_dir}{subjname}_geos_{rmask}_hemi2_geodesics')
      subj_in_dtitk_atlas_coord_list[subj].append(region_seeds[f'dtitk_{rmask}_hemi1'])
      subj_in_dtitk_atlas_fpref_list[subj].append(f'{out_subj_tracts_in_dtitkatlas_space_dir}{subjname}_geos_{rmask}_hemi1_geodesics')
      subj_in_dtitk_atlas_coord_list[subj].append(region_seeds[f'dtitk_{rmask}_hemi2'])
      subj_in_dtitk_atlas_fpref_list[subj].append(f'{out_subj_tracts_in_dtitkatlas_space_dir}{subjname}_geos_{rmask}_hemi2_geodesics')
      #ar = compute_geodesics(subj_tens_4_path, subj_mask, region_seeds[f'{rmask}_hemi1'], f'{out_subj_tracts_in_atlas_space_dir}{subjname}_geos_{rmask}_hemi1_geodesics', pool)
      #ars.append(ar)
      #ar = compute_geodesics(subj_tens_4_path, subj_mask, region_seeds[f'{rmask}_hemi2'], f'{out_subj_tracts_in_atlas_space_dir}{subjname}_geos_{rmask}_hemi2_geodesics', pool)
      #ars.append(ar)
    for smask in single_masks:
      subj_in_atlas_coord_list[subj].append(single_seeds[f'{smask}'])
      subj_in_atlas_fpref_list[subj].append(f'{out_subj_tracts_in_atlas_space_dir}{subjname}_geos_{smask}_geodesics')
      subj_in_dtitk_atlas_coord_list[subj].append(single_seeds[f'dtitk_{smask}'])
      subj_in_dtitk_atlas_fpref_list[subj].append(f'{out_subj_tracts_in_dtitkatlas_space_dir}{subjname}_geos_{smask}_geodesics')
      #ar = compute_geodesics(subj_tens_4_path, subj_mask, single_seeds[f'{smask}'], f'{out_subj_tracts_in_atlas_space_dir}{subjname}_geos_{smask}_geodesics', pool)
      #ars.append(ar)

    #ars = compute_geodesics(subj_tens_4_path, subj_mask, subj_in_atlas_coord_list[subj], subj_in_atlas_fpref_list[subj], pool)
    ars = compute_geodesics(subj_tens_4_path, atlas_mask, subj_in_atlas_coord_list[subj], subj_in_atlas_fpref_list[subj], pool)
    for ar in ars:
      all_ars.append(ar)
    ars = compute_geodesics(subj_dtitk_tens_4_path, dtitk_atlas_mask, subj_in_dtitk_atlas_coord_list[subj], subj_in_dtitk_atlas_fpref_list[subj], pool)
    for ar in ars:
      all_ars.append(ar)

    ##################
    # Transform seeds to subj space and compute tracts for subjs in subj space
    # Convert subj tracts from subj space to atlas space


    # use phi_inv to transform images and metrics from subj to atlas space
    # use phi to transform images and metrics from atlas to subj space
    # use phi to transform points from subj to atlas space
    # use phi_inv to transform points from atlas to subj space
    diffeo_img_to_atlas_fname = atlasdir + f'{subj}_phi_inv.mat'
    diffeo_img_to_subj_fname = atlasdir + f'{subj}_phi.mat'
    diffeo_to_atlas_fname = atlasdir + f'{subj}_phi.mat'
    diffeo_to_subj_fname = atlasdir + f'{subj}_phi_inv.mat'
    diffeo = sio.loadmat(diffeo_to_subj_fname)['diffeo']
    img_diffeo = sio.loadmat(diffeo_img_to_subj_fname)['diffeo']
    diffeo_to_atlas = sio.loadmat(diffeo_to_atlas_fname)['diffeo']

    # For DTITK use df to transform images and metrics from subj to atlas space
    # For DTITK use df_inv to transform images and metrics from atlas to subj space
    # For DTITK use df_inv to transform points from subj to atlas space
    # For DTITK use df to transform points from atlas to subj space
    diffeo_img_to_dtitk_atlas_fname = dtitkatlasdir + f'{subj}_padded_combined_aff_aff_diffeo_orig_dims.df.nii.gz'
    diffeo_img_to_dtitk_subj_fname = dtitkatlasdir + f'{subj}_padded_combined_aff_aff_diffeo_orig_dims.df_inv.nii.gz'
    diffeo_to_dtitk_atlas_fname = dtitkatlasdir + f'{subj}_padded_combined_aff_aff_diffeo_orig_dims.df_inv.nii.gz'
    diffeo_to_dtitk_subj_fname = dtitkatlasdir + f'{subj}_padded_combined_aff_aff_diffeo_orig_dims.df.nii.gz'
    disp = nib.load(diffeo_to_dtitk_subj_fname).get_fdata().squeeze()
    if disp.shape[-1] == 3:
      disp = disp.transpose((3,0,1,2))
    dtitk_diffeo = (get_idty_3d(disp.shape[1],disp.shape[2], disp.shape[3]) + torch.Tensor(disp)).detach().numpy()
    disp = nib.load(diffeo_img_to_dtitk_subj_fname).get_fdata().squeeze()
    if disp.shape[-1] == 3:
      disp = disp.transpose((3,0,1,2))
    dtitk_img_diffeo = (get_idty_3d(disp.shape[1],disp.shape[2], disp.shape[3]) + torch.Tensor(disp)).detach().numpy()
    disp = nib.load(diffeo_to_dtitk_atlas_fname).get_fdata().squeeze()
    if disp.shape[-1] == 3:
      disp = disp.transpose((3,0,1,2))
    dtitk_diffeo_to_atlas = (get_idty_3d(disp.shape[1],disp.shape[2], disp.shape[3]) + torch.Tensor(disp)).detach().numpy()
    
    try:
      fname = f'{out_tract_dir}{subj}_atlas_seeds_in_subj_space.pkl'
      subj_start_coords = convert_seeds(atlas_start_coords, diffeo, fname)
    except Exception as err:
      print('Caught', err, 'while trying to transform atlas_start_coords for subj', subj,
            'using diffeo', diffeo_to_subj_fname, '. Stopping processing for this subject.')
      continue
    try:
      fname = f'{out_tract_dir}{subj}_dtitk_atlas_seeds_in_subj_space.pkl'
      dtitk_subj_start_coords = convert_seeds(dtitk_atlas_start_coords, dtitk_diffeo, fname)
    except Exception as err:
      print('Caught', err, 'while trying to transform dtitk_atlas_start_coords for subj', subj,
            'using diffeo', diffeo_to_dtitk_subj_fname, '. Stopping processing for this subject.')
      continue

    subj_tens = ReadTensors(f'{outsubjdir}/{subj}_scaled_orig_tensors_rreg.nhdr')
    #end_mask = ReadScalars(f'{indir}/{subj}/dti_{bval}_FA_mask_0.20_rreg.nhdr')
    end_mask_subj_space = apply_transform_to_img(end_mask, img_diffeo)
    subj_tens_4_path = np.transpose(subj_tens,(3,0,1,2))
        
    end_mask_subj_space[end_mask_subj_space < 0.3] = 0
    end_mask_subj_space[end_mask_subj_space > 0] = 1

    subj_coord_list[subj].append(subj_start_coords)
    subj_fpref_list[subj].append(f'{out_subj_tract_dir}{subjname}_all_geos_geodesics')
    subj_to_atlas_fpref_list[subj].append(f'{out_subj_tracts_to_atlas_space_dir}{subjname}_all_geos_geodesics')

    subj_dtitk_tens = nib.load(f'{dtitkatlasdir}/{subj}_orig_tensors.nii.gz').get_fdata().squeeze()
    print('Swapping index 2 and 3 in tensor to match DTITK expectations.',  'DO NOT DO THIS if starting w/ NIFTI file w/ expected tensor ordering!')
    # Note dtitk is also off by one voxel in the x direction.  dtitk[0] = atlas[1]
    subj_dtitk_swap = subj_dtitk_tens.copy()
    #dtitk_swap[1:] = dtitk_tens[0:-1]
    subj_dtitk_swap[:,:,:,2] = subj_dtitk_tens[:,:,:,3]
    subj_dtitk_swap[:,:,:,3] = subj_dtitk_tens[:,:,:,2]
    #end_mask = ReadScalars(f'{indir}/{subj}/dti_{bval}_FA_mask_0.20_rreg.nhdr')
    end_dtitk_mask_subj_space = apply_transform_to_img(end_dtitk_mask, dtitk_img_diffeo)
    subj_dtitk_tens_4_path = np.transpose(subj_dtitk_swap,(3,0,1,2))
        
    end_dtitk_mask_subj_space[end_dtitk_mask_subj_space < 0.3] = 0
    end_dtitk_mask_subj_space[end_dtitk_mask_subj_space > 0] = 1

    subj_dtitk_coord_list[subj].append(subj_start_coords)
    subj_dtitk_fpref_list[subj].append(f'{out_dtitk_subj_tract_dir}{subjname}_all_geos_geodesics')
    subj_to_dtitk_atlas_fpref_list[subj].append(f'{out_subj_tracts_to_dtitkatlas_space_dir}{subjname}_all_geos_geodesics')

    
    #ar = compute_geodesics_subj_space(subj_tens_4_path, end_mask, subj_start_coords, f'{out_subj_tract_dir}{subjname}_all_geos_geodesics', diffeo_to_atlas_fname, f'{out_subj_tracts_to_atlas_space_dir}{subjname}_all_geos_geodesics',pool)
    #ars.append(ar)
    
    for rmask in region_masks:
      # hemi 1
      try:
        fname = f'{out_tract_dir}{subj}_atlas_seeds_{rmask}_hemi1_in_subj_space.pkl'
        subj_start_coords = convert_seeds(region_seeds[f'{rmask}_hemi1'], diffeo, fname)
      except Exception as err:
        print('Caught', err, 'while trying to transform start_coords for subj', subj, 'seed', f'{rmask}_hemi1',
              'using diffeo', diffeo_to_subj_fname, '. Stopping processing for this region.')
        continue
      try:
        fname = f'{out_tract_dir}{subj}_dtitk_atlas_seeds_{rmask}_hemi1_in_subj_space.pkl'
        dtitk_subj_start_coords = convert_seeds(region_seeds[f'dtitk_{rmask}_hemi1'], dtitk_diffeo, fname)
      except Exception as err:
        print('Caught', err, 'while trying to transform dtitk_start_coords for subj', subj, 'seed', f'dtitk_{rmask}_hemi1',
              'using diffeo', diffeo_to_dtitk_subj_fname, '. Stopping processing for this region.')
        continue

      subj_coord_list[subj].append(subj_start_coords)
      subj_fpref_list[subj].append(f'{out_subj_tract_dir}{subjname}_geos_{rmask}_hemi1_geodesics')
      subj_to_atlas_fpref_list[subj].append(f'{out_subj_tracts_to_atlas_space_dir}{subjname}_geos_{rmask}_hemi1_geodesics')
      subj_dtitk_coord_list[subj].append(dtitk_subj_start_coords)
      subj_dtitk_fpref_list[subj].append(f'{out_dtitk_subj_tract_dir}{subjname}_geos_{rmask}_hemi1_geodesics')
      subj_to_dtitk_atlas_fpref_list[subj].append(f'{out_subj_tracts_to_dtitkatlas_space_dir}{subjname}_geos_{rmask}_hemi1_geodesics')
      #ar = compute_geodesics_subj_space(subj_tens_4_path, end_mask, subj_start_coords, f'{out_subj_tract_dir}{subjname}_geos_{rmask}_hemi1_geodesics', diffeo_to_atlas_fname, f'{out_subj_tracts_to_atlas_space_dir}{subjname}_geos_{rmask}_hemi1_geodesics',pool)
      #ars.append(ar)

      # hemi 2
      try:
        fname = f'{out_tract_dir}{subj}_atlas_seeds_{rmask}_hemi2_in_subj_space.pkl'
        subj_start_coords = convert_seeds(region_seeds[f'{rmask}_hemi2'], diffeo, fname)
      except Exception as err:
        print('Caught', err, 'while trying to transform start_coords for subj', subj, 'seed', f'{rmask}_hemi2',
              'using diffeo', diffeo_to_subj_fname, '. Stopping processing for this region.')
        continue
      try:
        fname = f'{out_tract_dir}{subj}_dtitk_atlas_seeds_{rmask}_hemi2_in_subj_space.pkl'
        dtitk_subj_start_coords = convert_seeds(region_seeds[f'dtitk_{rmask}_hemi2'], dtitk_diffeo, fname)
      except Exception as err:
        print('Caught', err, 'while trying to transform dtitk_start_coords for subj', subj, 'seed', f'dtitk_{rmask}_hemi2',
              'using diffeo', diffeo_to_dtitk_subj_fname, '. Stopping processing for this region.')
        continue
      
      subj_coord_list[subj].append(subj_start_coords)
      subj_fpref_list[subj].append(f'{out_subj_tract_dir}{subjname}_geos_{rmask}_hemi2_geodesics')
      subj_to_atlas_fpref_list[subj].append(f'{out_subj_tracts_to_atlas_space_dir}{subjname}_geos_{rmask}_hemi2_geodesics')
      subj_dtitk_coord_list[subj].append(dtitk_subj_start_coords)
      subj_dtitk_fpref_list[subj].append(f'{out_dtitk_subj_tract_dir}{subjname}_geos_{rmask}_hemi2_geodesics')
      subj_to_dtitk_atlas_fpref_list[subj].append(f'{out_subj_tracts_to_dtitkatlas_space_dir}{subjname}_geos_{rmask}_hemi2_geodesics')
      #ar = compute_geodesics_subj_space(subj_tens_4_path, end_mask, subj_start_coords, f'{out_subj_tract_dir}{subjname}_geos_{rmask}_hemi2_geodesics', diffeo_to_atlas_fname, f'{out_subj_tracts_to_atlas_space_dir}{subjname}_geos_{rmask}_hemi2_geodesics', pool)
      #ars.append(ar)

      
    for smask in single_masks:
      try:
        fname = f'{out_tract_dir}{subj}_atlas_seeds_{smask}_in_subj_space.pkl'
        subj_start_coords = convert_seeds(single_seeds[f'{smask}'], diffeo, fname)
      except Exception as err:
        print('Caught', err, 'while trying to transform start_coords for subj', subj, 'seed', f'{smask}',
              'using diffeo', diffeo_to_subj_fname, '. Stopping processing for this region.')
        continue
      try:
        fname = f'{out_tract_dir}{subj}_dtitk_atlas_seeds_{smask}_in_subj_space.pkl'
        dtitk_subj_start_coords = convert_seeds(single_seeds[f'dtitk_{smask}'], dtitk_diffeo, fname)
      except Exception as err:
        print('Caught', err, 'while trying to transform dtitk_start_coords for subj', subj, 'seed', f'dtitk_{smask}',
              'using diffeo', diffeo_to_dtitk_subj_fname, '. Stopping processing for this region.')
        continue

      subj_coord_list[subj].append(subj_start_coords)
      subj_fpref_list[subj].append(f'{out_subj_tract_dir}{subjname}_geos_{smask}_geodesics')
      subj_to_atlas_fpref_list[subj].append(f'{out_subj_tracts_to_atlas_space_dir}{subjname}_geos_{smask}_geodesics')
      subj_dtitk_coord_list[subj].append(dtitk_subj_start_coords)
      subj_dtitk_fpref_list[subj].append(f'{out_dtitk_subj_tract_dir}{subjname}_geos_{smask}_geodesics')
      subj_to_dtitk_atlas_fpref_list[subj].append(f'{out_subj_tracts_to_dtitkatlas_space_dir}{subjname}_geos_{smask}_geodesics')
      #ar = compute_geodesics_subj_space(subj_tens_4_path, end_mask, subj_start_coords, f'{out_subj_tract_dir}{subjname}_geos_{smask}_geodesics', diffeo_to_atlas_fname, f'{out_subj_tracts_to_atlas_space_dir}{subjname}_geos_{smask}_geodesics', pool)
      #ars.append(ar)  
    ars = compute_geodesics_subj_space(subj_tens_4_path, end_mask_subj_space, subj_coord_list[subj], subj_fpref_list[subj],
                                       diffeo_to_atlas, subj_to_atlas_fpref_list[subj], pool)
    for ar in ars:
      all_ars.append(ar)
    ars = compute_geodesics_subj_space(subj_dtitk_tens_4_path, end_dtitk_mask_subj_space, subj_dtitk_coord_list[subj], subj_dtitk_fpref_list[subj],
                                       dtitk_diffeo_to_atlas, subj_to_dtitk_atlas_fpref_list[subj], pool)
    for ar in ars:
      all_ars.append(ar)
  # end for each subject

  for ar in all_ars:
    ar.wait()
    
  pool.close()
  pool.join()
      


  
  
