from lazy_imports import np
from lazy_imports import LinearOperator, gmres
from lazy_imports import sitk
import math
import time
import gc

from util import diff
from util import tensors
from util import maskops as mo
from util.riemann import riem_vec_norm
from data.convert import GetNPArrayFromSITK, GetSITKImageFromNP

# starting function will make a copy of the input data in order to keep the originals clean.

def precondition_tensors(tens, mask):
  # get rid of nans, poor conditioned tensors
  # Only keep elements of mask that are large enough to take accurate derivatives
  tens[np.isnan(tens)] = 0

def compute_g_derivs_new(g, g_inv, sqrt_det_g, bdry_idx, bdry_map):
  xsz = g.shape[0]
  ysz = g.shape[1]
  zsz = g.shape[2]
  gradG = np.zeros((xsz,ysz, zsz, 3, 3, 3))
  grad_g_inv = np.zeros((xsz,ysz, zsz, 3, 3, 3))
  grad_sqrt_det_g = np.zeros((xsz,ysz,zsz,3))

 # correct center derivatives (np.gradient) on all 4 boundaries

  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue
    
    b_idx = bdry_idx[bnum]

    xdiff, ydiff, zdiff = diff.select_diffs_3d(btype,zero_shifts=False)

    if btype == "outside":
      # outside mask, skip
      continue
    else:
      for ii in range(3):
        for jj in range(3):
          gradG[b_idx[0],b_idx[1],b_idx[2],ii,jj,0] = xdiff(g[:,:,:,ii,jj], b_idx)
          gradG[b_idx[0],b_idx[1],b_idx[2],ii,jj,1] = ydiff(g[:,:,:,ii,jj], b_idx)
          gradG[b_idx[0],b_idx[1],b_idx[2],ii,jj,2] = zdiff(g[:,:,:,ii,jj], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],b_idx[2],ii,jj,0] = xdiff(g_inv[:,:,:,ii,jj], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],b_idx[2],ii,jj,1] = ydiff(g_inv[:,:,:,ii,jj], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],b_idx[2],ii,jj,2] = zdiff(g_inv[:,:,:,ii,jj], b_idx)
    
      grad_sqrt_det_g[b_idx[0], b_idx[1], b_idx[2], 0] = xdiff(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1], b_idx[2], 1] = ydiff(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1], b_idx[2], 2] = zdiff(sqrt_det_g, b_idx)
        
  return(gradG, grad_g_inv, grad_sqrt_det_g)
# end compute_g_derivs_new
  
def compute_g_derivs(g, g_inv, sqrt_det_g, bdry_idx, bdry_map):
  print("compute_g_derivs not correct for 3D yet, use compute_g_derivs_new or edit compute_g_derivs")
  return([],[],[])
  xsz = g.shape[0]
  ysz = g.shape[1]
  zsz = g.shape[2]
  gradG = np.zeros((xsz,ysz, zsz, 3, 3, 3, 3))
  grad_g_inv = np.zeros((xsz,ysz, zsz, 3, 3, 3, 3))
  grad_sqrt_det_g = np.zeros((xsz,ysz,zsz,3))

 # correct center derivatives (np.gradient) on all 4 boundaries

  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue
    
    b_idx = bdry_idx[bnum]
    
    if btype[0:8] == "interior":
      for xx in range(3):
        for yy in range(3):
          for zz in range(3):
            gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,0] = diff.gradx_idx_3d(g[:,:,xx,yy,zz], b_idx)
            gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.grady_idx_3d(g[:,:,xx,yy,zz], b_idx)
            gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,2] = diff.gradz_idx_3d(g[:,:,xx,yy,zz], b_idx)
            grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,0] = diff.gradx_idx_3d(g_inv[:,:,xx,yy,zz], b_idx)
            grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.grady_idx_3d(g_inv[:,:,xx,yy,zz], b_idx)
            grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,2] = diff.gradz_idx_3d(g_inv[:,:,xx,yy,zz], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1], b_idx[2],0] = diff.gradx_idx_3d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1], b_idx[2],1] = diff.grady_idx_3d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1], b_idx[2],2] = diff.gradz_idx_3d(sqrt_det_g, b_idx)
                                                              
    elif btype == "left":
      for xx in range(3):
        for yy in range(3):
          for zz in range(3):
            gradG[b_idx[0],b_idx[2],xx,yy,zz,0] = diff.left_diff_idx_3d(g[:,:,xx,yy,zz], b_idx)
            gradG[b_idx[0],b_idx[2],xx,yy,zz,1] = diff.grady_idx_3d(g[:,:,xx,yy,zz], b_idx)
            gradG[b_idx[0],b_idx[2],xx,yy,zz,2] = diff.gradz_idx_3d(g[:,:,xx,yy,zz], b_idx)
            grad_g_inv[b_idx[0],b_idx[2],xx,yy,zz,0] = diff.left_diff_idx_3d(g_inv[:,:,xx,yy,zz], b_idx)
            grad_g_inv[b_idx[0],b_idx[2],xx,yy,zz,1] = diff.grady_idx_3d(g_inv[:,:,xx,yy,zz], b_idx)
            grad_g_inv[b_idx[0],b_idx[2],xx,yy,zz,2] = diff.gradz_idx_3d(g_inv[:,:,xx,yy,zz], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1], b_idx[2],0] = diff.left_diff_idx_3d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1], b_idx[2],1] = diff.grady_idx_3d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1], b_idx[2],2] = diff.gradz_idx_3d(sqrt_det_g, b_idx)
      
    elif btype == "bottomleft":
      for xx in range(3):
        for yy in range(3):
          for zz in range(3):
            gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,0] = diff.left_diff_idx_3d(g[:,:,xx,yy,zz], b_idx)
            gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.bottom_diff_idx_3d(g[:,:,xx,yy,zz], b_idx)
            gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,2] = diff.gradz_idx_3d(g[:,:,xx,yy,zz], b_idx)
            grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,0] = diff.left_diff_idx_3d(g_inv[:,:,xx,yy,zz], b_idx)
            grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.bottom_diff_idx_3d(g_inv[:,:,xx,yy,zz], b_idx)
            grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,2] = diff.gradz_idx_3d(g_inv[:,:,xx,yy,zz], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1], b_idx[2],0] = diff.left_diff_idx_3d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1], b_idx[2],1] = diff.bottom_diff_idx_3d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1], b_idx[2],2] = diff.gradz_idx_3d(sqrt_det_g, b_idx)
  
    elif btype == "topleft":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.left_diff_idx_2d(g[:,:,xx,yy,zz], b_idx)
          gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.top_diff_idx_2d(g[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.left_diff_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.top_diff_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],0] = diff.left_diff_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],1] = diff.top_diff_idx_2d(sqrt_det_g, b_idx)
          
    elif btype == "notright":
      # TODO confirm we want 0 in the y direction in this case
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.left_diff_idx_2d(g[:,:,xx,yy,zz], b_idx)
          gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = 0
          grad_g_inv[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.left_diff_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = 0

      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],0] = diff.left_diff_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],1] = 0

    elif btype == "right":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.right_diff_idx_2d(g[:,:,xx,yy,zz], b_idx)
          gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.grady_idx_2d(g[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.right_diff_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.grady_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],0] = diff.right_diff_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],1] = diff.grady_idx_2d(sqrt_det_g, b_idx)
      
    elif btype == "bottomright":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.right_diff_idx_2d(g[:,:,xx,yy,zz], b_idx)
          gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.bottom_diff_idx_2d(g[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.right_diff_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.bottom_diff_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],0] = diff.right_diff_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],1] = diff.bottom_diff_idx_2d(sqrt_det_g, b_idx)
  
    elif btype == "topright":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.right_diff_idx_2d(g[:,:,xx,yy,zz], b_idx)
          gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.top_diff_idx_2d(g[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.right_diff_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.top_diff_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],0] = diff.right_diff_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],1] = diff.top_diff_idx_2d(sqrt_det_g, b_idx)
          
    elif btype == "notleft":
      # TODO confirm we want 0 in the y direction in this case
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.right_diff_idx_2d(g[:,:,xx,yy,zz], b_idx)
          gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = 0
          grad_g_inv[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.right_diff_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = 0

      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],0] = diff.right_diff_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],1] = 0
      
    elif btype == "bottom":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.gradx_idx_2d(g[:,:,xx,yy,zz], b_idx)
          gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.bottom_diff_idx_2d(g[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.gradx_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.bottom_diff_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],0] = diff.gradx_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],1] = diff.bottom_diff_idx_2d(sqrt_det_g, b_idx)

    elif btype == "nottop":
      # TODO confirm want 0 in x direction
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = 0
          gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.bottom_diff_idx_2d(g[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = 0
          grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.bottom_diff_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],0] = 0
      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],1] = diff.bottom_diff_idx_2d(sqrt_det_g, b_idx)
  
    elif btype == "top":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.gradx_idx_2d(g[:,:,xx,yy,zz], b_idx)
          gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.top_diff_idx_2d(g[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = diff.gradx_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.top_diff_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],0] = diff.gradx_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],1] = diff.top_diff_idx_2d(sqrt_det_g, b_idx)

    elif btype == "notbottom":
      # TODO confirm want 0 in x direction
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = 0
          gradG[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.top_diff_idx_2d(g[:,:,xx,yy,zz], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],b_idx[2],xx,yy,zz,0] = 0
          grad_g_inv[b_idx[0],b_idx[1],b_idx[2],xx,yy,zz,1] = diff.top_diff_idx_2d(g_inv[:,:,xx,yy,zz], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],0] = 0
      grad_sqrt_det_g[b_idx[0], b_idx[1],b_idx[2],1] = diff.top_diff_idx_2d(sqrt_det_g, b_idx)
      
              
    elif btype == "outside":
      # outside mask, skip
      pass

    else:
      # unrecognized type
      print(btype, "unrecognized.  Skipping")
      
  return(gradG, grad_g_inv, grad_sqrt_det_g)
# end compute_g_derivs

def compute_T_derivs_new(TDirx, TDiry, TDirz, bdry_idx, bdry_map):
  xsz = TDirx.shape[0]
  ysz = TDirx.shape[1]
  zsz = TDirx.shape[2]
  gradTx_delx = np.zeros((xsz,ysz,zsz))
  gradTx_dely = np.zeros((xsz,ysz,zsz))
  gradTx_delz = np.zeros((xsz,ysz,zsz))
  gradTy_delx = np.zeros((xsz,ysz,zsz))
  gradTy_dely = np.zeros((xsz,ysz,zsz))
  gradTy_delz = np.zeros((xsz,ysz,zsz))
  gradTz_delx = np.zeros((xsz,ysz,zsz))
  gradTz_dely = np.zeros((xsz,ysz,zsz))
  gradTz_delz = np.zeros((xsz,ysz,zsz))

  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue

    b_idx = bdry_idx[bnum]

    xeigvdiff, yeigvdiff, zeigvdiff = diff.select_eigv_diffs_3d(btype)

    if btype == "outside":
      # outside mask, skip
      continue
    else:
      gradTx_delx[b_idx[0], b_idx[1],b_idx[2]], gradTy_delx[b_idx[0], b_idx[1],b_idx[2]], gradTz_delx[b_idx[0], b_idx[1],b_idx[2]] = xeigvdiff(TDirx, b_idx)
      gradTx_dely[b_idx[0], b_idx[1],b_idx[2]], gradTy_dely[b_idx[0], b_idx[1],b_idx[2]], gradTz_dely[b_idx[0], b_idx[1],b_idx[2]] = yeigvdiff(TDiry, b_idx)
      gradTx_delz[b_idx[0], b_idx[1],b_idx[2]], gradTy_delz[b_idx[0], b_idx[1],b_idx[2]], gradTz_delz[b_idx[0], b_idx[1],b_idx[2]] = zeigvdiff(TDirz, b_idx)
      
  return(gradTx_delx, gradTx_dely, gradTx_delz, gradTy_delx, gradTy_dely, gradTy_delz, gradTz_delx, gradTz_dely, gradTz_delz)
# end compute_T_derivs_new

def compute_T_derivs(TDir, bdry_idx, bdry_map):
  print("compute_T_derivs not correct for 3D yet, use compute_T_derivs_new or edit compute_T_derivs")
  return([],[],[],[],[],[],[],[],[])
  xsz = TDir.shape[0]
  ysz = TDir.shape[1]
  gradTx_delx = np.zeros((xsz,ysz))
  gradTx_dely = np.zeros((xsz,ysz))
  gradTy_delx = np.zeros((xsz,ysz))
  gradTy_dely = np.zeros((xsz,ysz))

  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue

    b_idx = bdry_idx[bnum]
    
    if btype[0:8] == "interior":
      gradTx_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_2d(TDir[:,:,0], b_idx)
      gradTx_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_2d(TDir[:,:,0], b_idx)
      gradTy_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_2d(TDir[:,:,1], b_idx)
      gradTy_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_2d(TDir[:,:,1], b_idx)
      
    elif btype == "left":
      gradTx_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTx_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_2d(TDir[:,:,0], b_idx)
      gradTy_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTy_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_2d(TDir[:,:,1], b_idx)

    elif btype == "bottomleft":
      gradTx_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTx_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTy_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTy_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(TDir[:,:,1], b_idx)

    elif btype == "topleft":
      gradTx_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTx_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTy_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTy_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(TDir[:,:,1], b_idx)

    elif btype == "notright":
      gradTx_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTx_dely[b_idx[0], b_idx[1],b_idx[2]] = 0
      gradTy_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTy_dely[b_idx[0], b_idx[1],b_idx[2]] = 0

    elif btype == "right":
      gradTx_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTx_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_2d(TDir[:,:,0], b_idx)
      gradTy_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTy_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_2d(TDir[:,:,1], b_idx)

    elif btype == "bottomright":
      gradTx_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTx_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTy_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTy_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(TDir[:,:,1], b_idx)

    elif btype == "topright":
      gradTx_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTx_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTy_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTy_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(TDir[:,:,1], b_idx)

    elif btype == "notleft":
      gradTx_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTx_dely[b_idx[0], b_idx[1],b_idx[2]] = 0
      gradTy_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTy_dely[b_idx[0], b_idx[1],b_idx[2]] = 0      

    elif btype == "bottom":
      gradTx_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_2d(TDir[:,:,0], b_idx)
      gradTx_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTy_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_2d(TDir[:,:,1], b_idx)
      gradTy_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(TDir[:,:,1], b_idx)

    elif btype == "nottop":
      gradTx_delx[b_idx[0], b_idx[1],b_idx[2]] = 0
      gradTx_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTy_delx[b_idx[0], b_idx[1],b_idx[2]] = 0
      gradTy_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(TDir[:,:,1], b_idx)
      
    elif btype == "top":
      gradTx_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_2d(TDir[:,:,0], b_idx)
      gradTx_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTy_delx[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_2d(TDir[:,:,1], b_idx)
      gradTy_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(TDir[:,:,1], b_idx)

    elif btype == "notbottom":
      gradTx_delx[b_idx[0], b_idx[1],b_idx[2]] = 0
      gradTx_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(TDir[:,:,0], b_idx)
      gradTy_delx[b_idx[0], b_idx[1],b_idx[2]] = 0
      gradTy_dely[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(TDir[:,:,1], b_idx)      

    elif btype == "outside":
      # outside mask, skip
      pass

    else:
      # unrecognized type
      print(btype, "unrecognized.  Skipping")
      
  return(gradTx_delx, gradTx_dely, gradTy_delx, gradTy_dely)
# end compute_T_derivs

def compute_nabla_derivs_new(nabla_TT, sqrt_det_nabla_TT, bdry_idx, bdry_map):
  xsz = nabla_TT.shape[0]
  ysz = nabla_TT.shape[1]
  zsz = nabla_TT.shape[2]
  grad_nabla_TT = np.zeros((xsz,ysz, zsz, 3, 3))
  grad_sqrt_det_nabla_TT = np.zeros((xsz,ysz,zsz, 3, 3))

  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue

    b_idx = bdry_idx[bnum]
    
    xdiff,ydiff,zdiff = diff.select_diffs_3d(btype,zero_shifts=False)

    if btype == "outside":
      # outside mask, skip
      continue
    else:
      grad_nabla_TT[b_idx[0],b_idx[1],b_idx[2],0,0] = xdiff(nabla_TT[:,:,:,0], b_idx)
      grad_nabla_TT[b_idx[0],b_idx[1],b_idx[2],0,1] = ydiff(nabla_TT[:,:,:,0], b_idx)
      grad_nabla_TT[b_idx[0],b_idx[1],b_idx[2],1,0] = xdiff(nabla_TT[:,:,:,1], b_idx)
      grad_nabla_TT[b_idx[0],b_idx[1],b_idx[2],0,2] = zdiff(nabla_TT[:,:,:,0], b_idx)
      grad_nabla_TT[b_idx[0],b_idx[1],b_idx[2],2,0] = xdiff(nabla_TT[:,:,:,2], b_idx)
      grad_nabla_TT[b_idx[0],b_idx[1],b_idx[2],1,1] = ydiff(nabla_TT[:,:,:,1], b_idx)
      grad_nabla_TT[b_idx[0],b_idx[1],b_idx[2],1,2] = zdiff(nabla_TT[:,:,:,1], b_idx)
      grad_nabla_TT[b_idx[0],b_idx[1],b_idx[2],2,1] = ydiff(nabla_TT[:,:,:,2], b_idx)
      grad_nabla_TT[b_idx[0],b_idx[1],b_idx[2],2,2] = zdiff(nabla_TT[:,:,:,2], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0],b_idx[1],b_idx[2],0,0] = xdiff(sqrt_det_nabla_TT[:,:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0],b_idx[1],b_idx[2],0,1] = ydiff(sqrt_det_nabla_TT[:,:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0],b_idx[1],b_idx[2],1,0] = xdiff(sqrt_det_nabla_TT[:,:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0],b_idx[1],b_idx[2],0,2] = zdiff(sqrt_det_nabla_TT[:,:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0],b_idx[1],b_idx[2],2,0] = xdiff(sqrt_det_nabla_TT[:,:,:,2], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0],b_idx[1],b_idx[2],1,1] = ydiff(sqrt_det_nabla_TT[:,:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0],b_idx[1],b_idx[2],1,2] = zdiff(sqrt_det_nabla_TT[:,:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0],b_idx[1],b_idx[2],2,1] = ydiff(sqrt_det_nabla_TT[:,:,:,2], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0],b_idx[1],b_idx[2],2,2] = zdiff(sqrt_det_nabla_TT[:,:,:,2], b_idx)
      
  return(grad_nabla_TT, grad_sqrt_det_nabla_TT)
# end compute_nabla_derivs_new

def compute_nabla_derivs(nabla_TT, sqrt_det_nabla_TT, bdry_idx, bdry_map):
  print("compute_nabla_derivs not correct for 3D yet, use compute_nabla_derivs_new or edit compute_nabla_derivs")
  return([],[])

  xsz = nabla_TT.shape[0]
  ysz = nabla_TT.shape[1]
  grad_nabla_TT = np.zeros((xsz,ysz, 2, 2))
  grad_sqrt_det_nabla_TT = np.zeros((xsz,ysz, 2, 2))

  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue

    b_idx = bdry_idx[bnum]
    
    if btype[0:8] == "interior":
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.gradx_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.grady_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.gradx_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.grady_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      
    elif btype == "left":
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.left_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.grady_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.left_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.grady_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "bottomleft":
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.left_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.left_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "topleft":
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.left_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.top_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.left_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.top_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "notright":
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.left_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = 0
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.left_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = 0

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = 0
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = 0

    elif btype == "right":
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.right_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.grady_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.right_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.grady_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "bottomright":
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.right_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.right_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "topright":
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.right_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.top_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.right_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.top_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "notleft":
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.right_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = 0
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.right_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = 0

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = 0
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = 0

    elif btype == "bottom":
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.gradx_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.gradx_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      
    elif btype == "nottop":
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = 0
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = 0
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = 0
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = 0
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      
    elif btype == "top":
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.gradx_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.top_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.gradx_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.top_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "notbottom":
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = 0
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.top_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = 0
      grad_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.top_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,0] = 0
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],0,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,0] = 0
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],b_idx[2],1,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "outside":
      # outside mask, skip
      pass

    else:
      # unrecognized type
      print(btype, "unrecognized.  Skipping")
      
  return(grad_nabla_TT, grad_sqrt_det_nabla_TT)
# end compute_nabla_derivs


def compute_alpha_derivs_orig(alpha, mask):
  xsz = alpha.shape[0]
  ysz = alpha.shape[1]
  zsz = alpha.shape[2]
  alpha_gradX, alpha_gradY, alpha_gradZ = np.gradient(alpha)
  alpha_gradX_left = diff.left_diff_3d(alpha)
  alpha_gradX_right = diff.right_diff_3d(alpha)
  alpha_gradY_bottom = diff.bottom_diff_3d(alpha)
  alpha_gradY_top = diff.top_diff_3d(alpha)
  alpha_gradZ_rear = diff.rear_diff_3d(alpha)
  alpha_gradZ_front = diff.front_diff_3d(alpha)

  for ii in range(xsz):
    for jj in range(ysz):
      for kk in range(zsz):
        if mask[ii,jj,kk]:
          if (not mask[ii-1,jj,kk]) and (not mask[ii+1,jj,kk]):
            alpha_gradX[ii,jj,kk] = 0
          if (not mask[ii-1,jj,kk]):
            alpha_gradX[ii,jj,kk] = alpha_gradX_left[ii,jj,kk]
          if (not mask[ii+1,jj,kk]):
            alpha_gradX[ii,jj,kk] = alpha_gradX_right[ii,jj,kk]
          if (not mask[ii,jj-1,kk]) and (not mask[ii,jj+1,kk]):
            alpha_gradY[ii,jj,kk] = 0
          if (not mask[ii,jj-1,kk]):
            alpha_gradY[ii,jj,kk] = alpha_gradY_bottom[ii,jj,kk]
          if (not mask[ii,jj+1,kk]):
            alpha_gradY[ii,jj,kk] = alpha_gradY_top[ii,jj,kk]
          if (not mask[ii,jj,kk-1]) and (not mask[ii,jj,kk+1]):
            alpha_gradZ[ii,jj,kk] = 0
          if (not mask[ii,jj,kk-1]):
            alpha_gradZ[ii,jj,kk] = alpha_gradZ_rear[ii,jj,kk]
          if (not mask[ii,jj,kk+1]):
            alpha_gradZ[ii,jj,kk] = alpha_gradZ_front[ii,jj,kk]

  return(alpha_gradX, alpha_gradY, alpha_gradZ)
# end compute_alpha_derivs_orig

def compute_alpha_derivs(alpha, bdry_idx, bdry_map):
  xsz = alpha.shape[0]
  ysz = alpha.shape[1]
  zsz = alpha.shape[2]
  alpha_gradX = np.zeros((xsz,ysz,zsz))
  alpha_gradY = np.zeros((xsz,ysz,zsz))
  alpha_gradZ = np.zeros((xsz,ysz,zsz))

  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue

    b_idx = bdry_idx[bnum]

    xdiff, ydiff, zdiff = diff.select_diffs_3d(btype,zero_shifts=False)

    if btype == "outside":
      # outside mask, skip
      continue
    else:
      alpha_gradX[b_idx[0], b_idx[1],b_idx[2]] = xdiff(alpha, b_idx)
      alpha_gradY[b_idx[0], b_idx[1],b_idx[2]] = ydiff(alpha, b_idx)
      alpha_gradZ[b_idx[0], b_idx[1],b_idx[2]] = zdiff(alpha, b_idx)
      
  return(alpha_gradX, alpha_gradY, alpha_gradZ)
# end compute_alpha_derivs

def compute_div_grad_alpha_orig(alpha_gradX, alpha_gradY, grad_det_g_x_g_inv, grad_g_inv_x_det_g, det_g_x_g_inv, sqrt_det_g, mask):
  xsz = alpha_gradX.shape[0]
  ysz = alpha_gradX.shape[1]
  alpha_div = np.zeros((xsz,ysz))

  # use np.gradient to calculate cross-derivatives for central point
  alpha_gradX_Y, alpha_gradY_Y = np.gradient(alpha_gradY)
  alpha_gradX_X, alpha_gradY_X = np.gradient(alpha_gradX)

  # use accurate one-sided diffs of central diff for all second derivatives, including cross-derivatives on the boundary 
  # Remember, we can't do this in the Ax calculation, because the Ax calc needs to account for Neumann Boundary Conditions
  # here, for a straight div grad ln_img we don't need to work about the boundary adjustment
  # cross derivatives
  alpha_gradY_left = diff.left_diff_2d(alpha_gradY)
  alpha_gradY_right = diff.right_diff_2d(alpha_gradY)
  alpha_gradX_bottom = diff.bottom_diff_2d(alpha_gradX)
  alpha_gradX_top = diff.top_diff_2d(alpha_gradX)

  # second derivs
  alpha_gradX_left = diff.left_diff_2d(alpha_gradX)
  alpha_gradX_right = diff.right_diff_2d(alpha_gradX)
  alpha_gradY_bottom = diff.bottom_diff_2d(alpha_gradY)
  alpha_gradY_top = diff.top_diff_2d(alpha_gradY)

  for ii in range(xsz):
    for jj in range(ysz):
      if mask[ii,jj]:
        if (not mask[ii-1,jj]) and mask[ii+1,jj] and mask[ii,jj-1] and mask[ii,jj+1]: # left boundary
          alpha_gradX_X[ii,jj] = alpha_gradX_left[ii,jj]
          alpha_gradX_Y[ii,jj] = alpha_gradY_left[ii,jj]

        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottomleft boundary
          alpha_gradX_X[ii,jj] = alpha_gradX_left[ii,jj]
          alpha_gradX_Y[ii,jj] = alpha_gradY_left[ii,jj]
          alpha_gradY_Y[ii,jj] = alpha_gradY_bottom[ii,jj]

        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and mask[ii,jj-1] and (not mask[ii,jj+1]): # topleft boundary
          alpha_gradX_X[ii,jj] = alpha_gradX_left[ii,jj]
          alpha_gradX_Y[ii,jj] = alpha_gradY_left[ii,jj]
          alpha_gradY_Y[ii,jj] = alpha_gradY_top[ii,jj]

        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and (not mask[ii,jj-1]) and (not mask[ii,jj+1]): # notright boundary
          alpha_gradX_X[ii,jj] = alpha_gradX_left[ii+1,jj]
          alpha_gradX_Y[ii,jj] = alpha_gradY_left[ii+1,jj] # correct?
          alpha_gradY_Y[ii,jj] = alpha_gradY_Y[ii+1,jj]
                   
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and mask[ii,jj-1] and mask[ii,jj+1]: # right boundary
          alpha_gradX_X[ii,jj] = alpha_gradX_right[ii,jj]
          alpha_gradX_Y[ii,jj] = alpha_gradY_right[ii,jj]

        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottomright boundary
          alpha_gradX_X[ii,jj] = alpha_gradX_right[ii,jj]
          alpha_gradX_Y[ii,jj] = alpha_gradY_right[ii,jj]
          alpha_gradY_Y[ii,jj] = alpha_gradY_bottom[ii,jj]
                   
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and mask[ii,jj-1] and (not mask[ii,jj+1]): # topright boundary
          alpha_gradX_X[ii,jj] = alpha_gradX_right[ii,jj]
          alpha_gradX_Y[ii,jj] = alpha_gradY_right[ii,jj]
          alpha_gradY_Y[ii,jj] = alpha_gradY_top[ii,jj]
                   
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and (not mask[ii,jj+1]): # notleft boundary
          alpha_gradX_X[ii,jj] = alpha_gradX_right[ii-1,jj]
          alpha_gradX_Y[ii,jj] = alpha_gradY_right[ii-1,jj] # correct?
          alpha_gradY_Y[ii,jj] = alpha_gradY_Y[ii-1,jj]

        elif mask[ii-1,jj] and mask[ii+1,jj] and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottom boundary
          alpha_gradX_Y[ii,jj] = alpha_gradX_bottom[ii,jj]
          alpha_gradY_Y[ii,jj] = alpha_gradY_bottom[ii,jj]

        elif (not mask[ii-1,jj]) and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and mask[ii,jj+1]: # nottop boundary
          alpha_gradX_X[ii,jj] = alpha_gradX_X[ii,jj+1]  # correct?
          alpha_gradX_Y[ii,jj] = alpha_gradX_bottom[ii,jj+1]
          alpha_gradY_Y[ii,jj] = alpha_gradY_bottom[ii,jj+1]
                   
        elif mask[ii-1,jj] and mask[ii+1,jj] and mask[ii,jj-1] and (not mask[ii,jj+1]): # top boundary
          alpha_gradX_Y[ii,jj] = alpha_gradX_top[ii,jj]
          alpha_gradY_Y[ii,jj] = alpha_gradY_top[ii,jj]

        elif (not mask[ii-1,jj]) and (not mask[ii+1,jj]) and mask[ii,jj-1] and (not mask[ii,jj+1]): # notbottom boundary
          alpha_gradX_X[ii,jj] = alpha_gradX_X[ii,jj-1]  # correct?
          alpha_gradX_Y[ii,jj] = alpha_gradX_top[ii,jj-1]
          alpha_gradY_Y[ii,jj] = alpha_gradY_top[ii,jj-1]

        else:
          # interior point
          if mask[ii-1,jj-1] and mask[ii+1,jj-1] and mask[ii-1,jj+1] and mask[ii+1,jj+1]:
            pass
          elif mask[ii-1,jj-1] and mask[ii-1,jj+1]:
            # same as for right
            alpha_gradX_Y[ii,jj] = alpha_gradY_right[ii,jj]
            
          elif mask[ii+1,jj-1] and mask[ii+1,jj+1]:
            # same as for left
            alpha_gradX_Y[ii,jj] = alpha_gradY_left[ii,jj]
            
          elif mask[ii-1,jj+1] and mask[ii+1,jj+1]:
            # same as for bottom
            alpha_gradX_Y[ii,jj] = alpha_gradX_bottom[ii,jj]
            
          elif mask[ii-1,jj-1] and mask[ii+1,jj-1]:
            # same as for top
            alpha_gradX_Y[ii,jj] = alpha_gradX_top[ii,jj]
            
          else:
            # not sure yet whether we need to worry about these other cases, note if we get one
            print(ii,jj,mask[ii-1,jj-1],mask[ii-1,jj+1],
                  mask[ii+1,jj-1],mask[ii+1,jj+1],'unexpected interior point case')

        alpha_div[ii,jj] = (grad_det_g_x_g_inv[ii,jj,0,0,0] + grad_g_inv_x_det_g[ii,jj,0,0,0])* alpha_gradX[ii,jj] \
                           + det_g_x_g_inv[ii,jj,0,0] * alpha_gradX_X[ii,jj] \
                           + (grad_det_g_x_g_inv[ii,jj,0,0,1] + grad_g_inv_x_det_g[ii,jj,0,1,0])* alpha_gradY[ii,jj] \
                           + det_g_x_g_inv[ii,jj,0,1] * alpha_gradX_Y[ii,jj] \
                           + (grad_det_g_x_g_inv[ii,jj,1,1,0] + grad_g_inv_x_det_g[ii,jj,1,0,1])* alpha_gradX[ii,jj] \
                           + det_g_x_g_inv[ii,jj,1,0] * alpha_gradX_Y[ii,jj] \
                           + (grad_det_g_x_g_inv[ii,jj,1,1,1] + grad_g_inv_x_det_g[ii,jj,1,1,1])* alpha_gradY[ii,jj] \
                           + det_g_x_g_inv[ii,jj,1,1] * alpha_gradY_Y[ii,jj] / sqrt_det_g[ii,jj]
  return(alpha_div)
# end compute_div_grad_alpha_orig

def compute_div_grad_alpha(alpha_gradX, alpha_gradY, grad_det_g_x_g_inv, grad_g_inv_x_det_g, det_g_x_g_inv, sqrt_det_g, bdry_idx, bdry_map):
  xsz = alpha_gradX.shape[0]
  ysz = alpha_gradX.shape[1]
  zsz = alpha_gradX.shape[2]
  alpha_div = np.zeros((xsz,ysz,zsz))
  alpha_gradX_X = np.zeros((xsz,ysz,zsz))
  alpha_gradX_Y = np.zeros((xsz,ysz,zsz))
  #alpha_gradY_X = np.zeros((xsz,ysz))
  alpha_gradY_Y = np.zeros((xsz,ysz,zsz))
  
  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue

    b_idx = bdry_idx[bnum]
    
    if btype[0:8] == "interior":
      alpha_gradX_X[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_2d(alpha_gradY, b_idx)
      
    elif btype == "left":
      alpha_gradX_X[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_2d(alpha_gradY, b_idx)

    elif btype == "bottomleft":
      alpha_gradX_X[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(alpha_gradY, b_idx)

    elif btype == "topleft":
      alpha_gradX_X[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(alpha_gradY, b_idx)
    elif btype == "notright":
      alpha_gradX_X[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1],b_idx[2]] = 0
      #alpha_gradY_X[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1],b_idx[2]] = 0

    elif btype == "right":
      alpha_gradX_X[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_2d(alpha_gradY, b_idx)

    elif btype == "bottomright":
      alpha_gradX_X[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(alpha_gradY, b_idx)

    elif btype == "topright":
      alpha_gradX_X[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(alpha_gradY, b_idx)

    elif btype == "notleft":
      alpha_gradX_X[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1],b_idx[2]] = 0
      #alpha_gradY_X[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1],b_idx[2]] = 0

    elif btype == "bottom":
      alpha_gradX_X[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(alpha_gradY, b_idx)

    elif btype == "nottop":
      alpha_gradX_X[b_idx[0], b_idx[1],b_idx[2]] = 0
      alpha_gradX_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1],b_idx[2]] = 0
      alpha_gradY_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_2d(alpha_gradY, b_idx)
      
    elif btype == "top":
      alpha_gradX_X[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(alpha_gradY, b_idx)

    elif btype == "notbottom":
      alpha_gradX_X[b_idx[0], b_idx[1],b_idx[2]] = 0
      alpha_gradX_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1],b_idx[2]] = 0
      alpha_gradY_Y[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_2d(alpha_gradY, b_idx)

    elif btype == "outside":
      # outside mask, skip
      pass

    else:
      # unrecognized type
      print(btype, "unrecognized.  Skipping")

  alpha_div[:,:] = ( (grad_det_g_x_g_inv[:,:,0,0,0] + grad_g_inv_x_det_g[:,:,0,0,0])* alpha_gradX[:,:] \
                     + det_g_x_g_inv[:,:,0,0] * alpha_gradX_X[:,:] \
                     + (grad_det_g_x_g_inv[:,:,0,0,1] + grad_g_inv_x_det_g[:,:,0,1,0])* alpha_gradY[:,:] \
                     + det_g_x_g_inv[:,:,0,1] * alpha_gradX_Y[:,:] \
                     + (grad_det_g_x_g_inv[:,:,1,1,0] + grad_g_inv_x_det_g[:,:,1,0,1])* alpha_gradX[:,:] \
                     + det_g_x_g_inv[:,:,1,0] * alpha_gradX_Y[:,:] \
                     + (grad_det_g_x_g_inv[:,:,1,1,1] + grad_g_inv_x_det_g[:,:,1,1,1])* alpha_gradY[:,:] \
                     + det_g_x_g_inv[:,:,1,1] * alpha_gradY_Y[:,:] ) / sqrt_det_g[:,:]    
  return(alpha_div)
# end compute_div_grad_alpha

def neumann_conditions_rhs(nabla_TT, g, grad_det_g_x_g_inv, grad_g_inv_x_det_g, det_g_x_g_inv, sqrt_det_g, bdry_idx, bdry_map):
  xsz = nabla_TT.shape[0]
  ysz = nabla_TT.shape[1]
  zsz = nabla_TT.shape[2]
  bdry_term = np.zeros((xsz,ysz,zsz))
  n = np.zeros((3))

  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue

    b_idx = bdry_idx[bnum]
    
    if btype[0:8] == "interior":
      # No boundary adjustments for interior points
      continue

    if ("shiftx" in btype) or ("shiftleft" in btype) or ("shiftright" in btype):
      n[0] = 0
      xsign = 0
    elif "left" in btype:
      n[0] = -1.0
      xsign = -1
    elif "right" in btype:
      n[0] = 1.0
      xsign = 1
    else:
      n[0] = 0
      xsign = 0
    if ("shifty" in btype) or ("shiftbottom" in btype) or ("shifttop" in btype):
      n[1] = 0
      ysign = 0
    elif "bottom" in btype:
      n[1] = -1.0 # TODO Check that this sign is correct, will change things from 2d version
      ysign = -1
    elif "top" in btype:
      n[1] = 1.0
      ysign = 1
    else:
      n[1] = 0
      ysign = 0
    if ("shiftz" in btype) or ("shiftrear" in btype) or ("shiftfront" in btype):
      n[2] = 0
      zsign = 0
    elif "rear" in btype:
      n[2] = -1.0
      zsign = -1
    elif "front" in btype:
      n[2] = 1.0
      zsign = 1
    else:
      n[2] = 0
      zsign = 0

    nmag = np.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
    
    if nmag > 0:
      n = n / nmag

    # This way is easier to code, but is also less efficient as many of these terms will be 0 based on values in n.
    
    if np.abs(n[0]) > 0.00001:
      Bx = 2*(nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] * n[0]*n[0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] * n[0]*n[1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] * n[0]*n[2] 
              + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] * n[0]*n[0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] * n[0]*n[1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] * n[0]*n[2]
              + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] * n[0]*n[0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] * n[0]*n[1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2] * n[0]*n[2])
    else:
      Bx = 0

    if np.abs(n[1]) > 0.00001:
      By = 2*(nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] * n[1]*n[0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] * n[1]*n[1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] * n[1]*n[2]
              + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] * n[1]*n[0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] * n[1]*n[1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] * n[1]*n[2] 
              + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] * n[1]*n[0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] * n[1]*n[1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2] * n[1]*n[2])
    else:
      By = 0
      
    if np.abs(n[2]) > 0.00001:
      Bz = 2*(nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] * n[2]*n[0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] * n[2]*n[1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] * n[2]*n[2] 
              + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] * n[2]*n[0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] * n[2]*n[1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] * n[2]*n[2] 
              + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] * n[2]*n[0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] * n[2]*n[1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2] * n[2]*n[2])
    else:
      Bz = 0

    # if 'shift' not in btype:
    bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
                                              + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
                                              + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
                                              + xsign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
                                              +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
                                              + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
                                              + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
                                              + ysign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By \
                                              +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
                                              + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
                                              + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
                                              + zsign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
    # else:
    #   # handle the shift cases (correspond in 2D to notleft, notright, etc)
    #   xdyy = 0
    #   xdzz = 0
    #   ydxx = 0
    #   ydzz = 0
    #   zdxx = 0
    #   zdyy = 0
    #   if 'shiftx' in btype:
    #     if 'top' in btype:
    #       ydxx = ysign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]
    #     elif 'bottom' in btype:
    #       ydxx = ysign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]
    #     elif 'rear' in btype:
    #       zdxx = zsign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]
    #     elif 'front' in btype:
    #       zdxx = zsign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]
    #   if 'shifty' in btype:
    #     if 'left' in btype:
    #       xdyy = xsign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]
    #     elif 'right' in btype:
    #       xdyy = xsign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]
    #     elif 'rear' in btype:
    #       zdyy = zsign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]
    #     elif 'front' in btype:
    #       zdyy = zsign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]
    #   if 'shiftz' in btype:
    #     if 'left' in btype:
    #       xdzz = xsign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]
    #     elif 'right' in btype:
    #       xdzz = xsign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]
    #     elif 'top' in btype:
    #       ydzz = ysign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]
    #     elif 'bottom' in btype:
    #       ydzz = ysign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]

    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                             + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                             + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                             + xsign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0] \
    #                                             + xdyy + xdzz) * Bx \
    #                                             # + xsign * dxdy + xsign * dxdz) * Bx \
    #                                             # + dxdy + dxdz) * Bx \
    #                                             +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                             + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                             + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                             + ysign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1] \
    #                                             + ydxx + ydzz) * By \
    #                                             # + ysign * dxdy + ysign * dydz) * By \
    #                                             # + dxdy + dydz) * By \
    #                                             +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                             + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                             + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                             + zsign * 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2] \
    #                                             + zdxx + zdyy) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
    #                                             # + zsign * dxdz + zsign * dydz) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
    #                                             # + dxdz + dydz) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
        
    
    # if btype == "left":
    #   Bx = 2 * nabla_TT[b_idx[0],b_idx[1],b_idx[2],0] * g[b_idx[0],b_idx[1],b_idx[2],0,0] \
    #      + 2 * nabla_TT[b_idx[0],b_idx[1],b_idx[2],1] * g[b_idx[0],b_idx[1],b_idx[2],1,0] \
    #      + 2 * nabla_TT[b_idx[0],b_idx[1],b_idx[2],2] * g[b_idx[0],b_idx[1],b_idx[2],2,0]

    #   bdry_term[b_idx[0],b_idx[1],b_idx[2]] = (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                    + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                    + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                    - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
      
    # elif btype == "bottomleft":
    #   Bx = nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1]
    #   By = Bx # TODO skip copy as it's wasteful
      
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                       - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "topleft":
    #   Bx = nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1]
    #   By = -Bx
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                       + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
     
    # elif btype == "notright":
    #   Bx = 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0]
    #   # By = 0
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      - det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,1] - det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,0]) * Bx ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "right":
    #   Bx = 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],1,2]                     
    #   # By = 0
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                    + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                    + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                    + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx /  sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "bottomright":
    #   Bx = nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1]
    #   By = -Bx
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                       - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "topright":
    #   Bx = nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1]
                     
    #   By = Bx # TODO skip copy as it's wasteful
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                       + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "notleft":
    #   Bx = 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0]
    #   # By = 0
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      + det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,1] + det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,0]) * Bx ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "bottom":
    #   # Bx = 0
    #   By = 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1]
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                    + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                    + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                    - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "nottop":
    #   # Bx = 0
    #   By = 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1]

    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                      - det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,1] - det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,0]) * By ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
      
    # elif btype == "top":
    #    # Bx = 0
    #   By = 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1]
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                    + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                    + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                    + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "notbottom":
    #   # Bx = 0
    #   By = 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1]
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                      + det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,1] + det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,0]) * By ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "rear":
    #   Bz = 2 * nabla_TT[b_idx[0],b_idx[1],b_idx[2],0] * g[b_idx[0],b_idx[1],b_idx[2],0,2] \
    #      + 2 * nabla_TT[b_idx[0],b_idx[1],b_idx[2],1] * g[b_idx[0],b_idx[1],b_idx[2],1,2] \
    #      + 2 * nabla_TT[b_idx[0],b_idx[1],b_idx[2],2] * g[b_idx[0],b_idx[1],b_idx[2],2,2]

    #   bdry_term[b_idx[0],b_idx[1],b_idx[2]] = (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                    + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                    + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                    - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
      
    # elif btype == "rearleft":
    #   Bx = nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2]
    #   Bz = Bx # TODO skip copy as it's wasteful
      
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                       - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "rearright":
    #   Bx = nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2]
    #   Bz = -Bx
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                     + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                     + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                     + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                    +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                      + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
     
    # elif btype == "rearbottom":
    #   By = nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2]
    #   Bz = By # TODO skip copy as it's wasteful
      
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                      - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                       - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "reartop":
    #   By = nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2]
    #   Bz = -By
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                      + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                       - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
    # elif btype == "rearbottomleft":
    #   Bx = 2.0/3.0 * (nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2])
    #   By = Bx
    #   Bz = Bx
      
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                      - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                       - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "reartopleft":
    #   Bx = 2.0/3.0 * (nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2])
    #   By = -Bx
    #   Bz = Bx
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                      + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                       - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
    # elif btype == "rearbottomright":
    #   Bx = 2.0/3.0 * (nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2])
    #   By = -Bx
    #   Bz = -Bx
      
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                      - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                       - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "reartopright":
    #   Bx = 2.0/3.0 * (nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2])
    #   By = Bx
    #   Bz = -Bx
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                      + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                       - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
    # elif btype == "notfront":
    #   Bz = 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2]
    #   # By = 0
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                      - det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,1] - det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
                
    # elif btype == "front":
    #   Bz = 2 * nabla_TT[b_idx[0],b_idx[1],b_idx[2],0] * g[b_idx[0],b_idx[1],b_idx[2],0,2] \
    #      + 2 * nabla_TT[b_idx[0],b_idx[1],b_idx[2],1] * g[b_idx[0],b_idx[1],b_idx[2],1,2] \
    #      + 2 * nabla_TT[b_idx[0],b_idx[1],b_idx[2],2] * g[b_idx[0],b_idx[1],b_idx[2],2,2]

    #   bdry_term[b_idx[0],b_idx[1],b_idx[2]] = (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                    + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                    + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                    + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
      
    # elif btype == "frontleft":
    #   Bx = nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2]
    #   Bz = Bx # TODO skip copy as it's wasteful
      
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                       + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "frontright":
    #  Bx = nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #       + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #       + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2]
    #  Bz = -Bx
    #  bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                     + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                     + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                     + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                    +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                      + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
     
    # elif btype == "frontbottom":
    #   By = nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #        + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2]
    #   Bz = By # TODO skip copy as it's wasteful
      
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                      - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                       + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "fronttop":
    #  By = nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #       + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #       + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2]
    #  Bz = -By
    #  bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                      + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                       + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
    # elif btype == "frontbottomleft":
    #   Bx = 2.0/3.0 * (nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2])
    #   By = Bx
    #   Bz = -Bx
      
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                      - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                       + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "fronttopleft":
    #   Bx = 2.0/3.0 * (nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2])
    #   By = -Bx
    #   Bz = -Bx
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                      + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                       + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
    # elif btype == "frontbottomright":
    #   Bx = 2.0/3.0 * (nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] - nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2])
    #   By = -Bx
    #   Bz = Bx
      
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                      - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                       + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "fronttopright":
    #   Bx = 2.0/3.0 * (nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #                 + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,0] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,1] + nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2])
    #   By = Bx
    #   Bz = Bx
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,0,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,0,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,0,2] \
    #                                      + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0]) * Bx \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,1,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,1,0] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,1,2] \
    #                                      + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1]) * By \
    #                                     +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                       + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
    # elif btype == "notrear":
    #   Bz = 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],0] * g[b_idx[0], b_idx[1],b_idx[2],0,2] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],1] * g[b_idx[0], b_idx[1],b_idx[2],1,2] \
    #      + 2 * nabla_TT[b_idx[0], b_idx[1],b_idx[2],2] * g[b_idx[0], b_idx[1],b_idx[2],2,2]
    #   # By = 0
    #   bdry_term[b_idx[0], b_idx[1],b_idx[2]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,2,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],2,2,2] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,1,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],1,2,1] \
    #                                      + grad_det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],0,0,2] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],b_idx[2],0,2,0] \
    #                                      + det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],2,1] + det_g_x_g_inv[b_idx[0], b_idx[1],b_idx[2],1,2]) * Bz ) / sqrt_det_g[b_idx[0], b_idx[1],b_idx[2]]
                     
    # elif btype == "outside":
    #   # outside mask, skip
    #   pass

    # else:
    #   # unrecognized type
    #   print(btype, "unrecognized.  Skipping")

  return(bdry_term)
# end neumann_conditions_rhs

def Ax(x, args):
# Note, need to fix with args prior to making a LinearOperator
  # args['sqrt_det_g'], args['mask']

  alpha = np.zeros((args['xsz'],args['ysz'],args['zsz']))
  alpha[args['mask']>0] = x

  gradX, gradY, gradZ = compute_alpha_derivs(alpha, args['bdry_idx'], args['bdry_map'])
  
  div = np.zeros((args['xsz'],args['ysz'],args['zsz']))

  gradXX = np.zeros((args['xsz'],args['ysz'],args['zsz']))
  gradXY = np.zeros((args['xsz'],args['ysz'],args['zsz']))
  gradXZ = np.zeros((args['xsz'],args['ysz'],args['zsz']))
  gradYY = np.zeros((args['xsz'],args['ysz'],args['zsz']))
  gradYZ = np.zeros((args['xsz'],args['ysz'],args['zsz']))
  gradZZ = np.zeros((args['xsz'],args['ysz'],args['zsz']))
  
  # use np.gradient to calculate cross-derivatives for central point
  #gradX_Y, gradY_Y = np.gradient(gradY)
  # use accurate one-sided diffs of central diff for cross-derivatives on the boundary 
  #gradY_left = diff.left_diff_2d(gradY)
  #gradY_right = diff.right_diff_2d(gradY)
  #gradX_bottom = diff.bottom_diff_2d(gradX)
  #gradX_top = diff.top_diff_2d(gradX)
   
  # grad_det_g_x_g_inv[i,j, gradient-direction, g-component, g-component] = gradient of sqrt_det_g .* g_inv
  # grad_g_inv_x_det_g[i,j, g-component, g-component, gradient-direction] = gradient of g_inv .* sqrt_det_g
  # det_g_x_g_inv[i,j, g-component, g-component] = sqrt_det_g .* g_inv
  for btype, bnum in args['bdry_map'].items():
    if bnum == 0:
      # skip, since outside
      continue

    b_idx = args['bdry_idx'][bnum]
    
    if btype[0:8] == "interior":
      gradXX[b_idx[0], b_idx[1],b_idx[2]] = diff.gradxx_idx_3d(alpha, b_idx)
      gradYY[b_idx[0], b_idx[1],b_idx[2]] = diff.gradyy_idx_3d(alpha, b_idx)
      gradZZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradzz_idx_3d(alpha, b_idx)

      if btype == "interior":
        gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_3d(gradX, b_idx)
        gradXZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradz_idx_3d(gradX, b_idx)
        gradYZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradz_idx_3d(gradY, b_idx)
        
      elif btype == "interiorleft":
        gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_3d(gradX, b_idx)
        gradXZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradz_idx_3d(gradX, b_idx)
        gradYZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradz_idx_3d(gradY, b_idx)

      elif btype == "interiorright":
        gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_3d(gradX, b_idx)
        gradXZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradz_idx_3d(gradX, b_idx)
        gradYZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradz_idx_3d(gradY, b_idx)

      elif btype == "interiorbottom":
        gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_3d(gradY, b_idx)
        gradXZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradz_idx_3d(gradX, b_idx)
        gradYZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradz_idx_3d(gradY, b_idx)

      elif btype == "interiortop":
        gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_3d(gradY, b_idx)
        gradXZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradz_idx_3d(gradX, b_idx)
        gradYZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradz_idx_3d(gradY, b_idx)

      elif btype == "interiorrear":
        gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_3d(gradX, b_idx)
        gradXZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_3d(gradZ, b_idx)
        gradYZ[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_3d(gradZ, b_idx)

      elif btype == "interiorfront":
        gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_3d(gradX, b_idx)
        gradXZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradx_idx_3d(gradZ, b_idx)
        gradYZ[b_idx[0], b_idx[1],b_idx[2]] = diff.grady_idx_3d(gradZ, b_idx)
  
      else:
        # TODO How valid/necessary is this case?
        gradXY[b_idx[0], b_idx[1],b_idx[2]] = 0
        gradXZ[b_idx[0], b_idx[1],b_idx[2]] = 0
        gradYZ[b_idx[0], b_idx[1],b_idx[2]] = 0
        

      div[b_idx[0], b_idx[1],b_idx[2]] = (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,0,0])* gradX[b_idx[0], b_idx[1],b_idx[2]] \
                                        + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * gradXX[b_idx[0], b_idx[1],b_idx[2]] \
                                       + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,1,0])* gradY[b_idx[0], b_idx[1],b_idx[2]] \
                                        + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1] * gradXY[b_idx[0], b_idx[1],b_idx[2]] \
                                       + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,2,0])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
                                        + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,2] * gradXZ[b_idx[0], b_idx[1],b_idx[2]] \
                                       + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,0,1])* gradX[b_idx[0], b_idx[1],b_idx[2]] \
                                        + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] * gradXY[b_idx[0], b_idx[1],b_idx[2]] \
                                       + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,1,1])* gradY[b_idx[0], b_idx[1],b_idx[2]] \
                                        + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * gradYY[b_idx[0], b_idx[1],b_idx[2]] \
                                       + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,2,1])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
                                        + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,2] * gradYZ[b_idx[0], b_idx[1],b_idx[2]] \
                                       + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],2,0,2])* gradX[b_idx[0], b_idx[1],b_idx[2]] \
                                        + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,0] * gradXZ[b_idx[0], b_idx[1],b_idx[2]] \
                                       + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],2,1,2])* gradY[b_idx[0], b_idx[1],b_idx[2]] \
                                        + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,1] * gradYZ[b_idx[0], b_idx[1],b_idx[2]] \
                                       + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],2,2,2])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
                                        + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2] * gradZZ[b_idx[0], b_idx[1],b_idx[2]]

    else:
      # set to None at beginning of each case to ensure all cases assign all values appropriately, and that values aren't overwritten
      xdiff = None
      ydiff = None
      zdiff = None
      xxdiff = 0
      xydiff = 0
      xzdiff = 0
      yydiff = 0
      yzdiff = 0
      zzdiff = 0
      xxset = 0
      yyset = 0
      zzset = 0
      xyset = 0
      xzset = 0
      yzset = 0
      if "shiftx" in btype:
        # TODO shift
        if ("shifty" in btype) or ("shiftbottom" in btype) or ("shifttop" in btype):
          xdiff = None # Set to None, because this term is in RHS Neumann condition
          if "tore" in btype:
            xxdiff = (-4*alpha[b_idx[0], b_idx[1],b_idx[2]+1] + 2*alpha[b_idx[0], b_idx[1],b_idx[2]+2]
                     + 2*alpha[b_idx[0]-1, b_idx[1],b_idx[2]+1] - alpha[b_idx[0]-1, b_idx[1],b_idx[2]+2]
                     + 2*alpha[b_idx[0]+1, b_idx[1],b_idx[2]+1] - alpha[b_idx[0]+1, b_idx[1],b_idx[2]+2])
            xzdiff = (-1.5*alpha[b_idx[0],b_idx[1],b_idx[2]+1] + 3*alpha[b_idx[0],b_idx[1],b_idx[2]+2] - 1.5*alpha[b_idx[0],b_idx[1],b_idx[2]+3]
                       + 2*alpha[b_idx[0]-1,b_idx[1],b_idx[2]+1] - 3.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]+2] + 1.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]+3]
                     - 0.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]+1] + 0.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]+2])
          elif "tof" in btype:
            xxdiff = (-4*alpha[b_idx[0], b_idx[1],b_idx[2]-1] + 2*alpha[b_idx[0], b_idx[1],b_idx[2]-2]
                     + 2*alpha[b_idx[0]-1, b_idx[1],b_idx[2]-1] - alpha[b_idx[0]-1, b_idx[1],b_idx[2]-2]
                     + 2*alpha[b_idx[0]+1, b_idx[1],b_idx[2]-1] - alpha[b_idx[0]+1, b_idx[1],b_idx[2]-2])
            xzdiff = (1.5*alpha[b_idx[0],b_idx[1],b_idx[2]-1] - 3*alpha[b_idx[0],b_idx[1],b_idx[2]-2] + 1.5*alpha[b_idx[0],b_idx[1],b_idx[2]-3]
                       - 2*alpha[b_idx[0]-1,b_idx[1],b_idx[2]-1] + 3.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]-2] - 1.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]-3]
                     + 0.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]-1] - 0.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]-2])
        elif "tob" in btype or "bottom" in btype:
          xdiff = None # Set to None, because this term is in RHS Neumann condition
          xxdiff = (-4*alpha[b_idx[0], b_idx[1]+1,b_idx[2]] + 2*alpha[b_idx[0], b_idx[1]+2,b_idx[2]]
                   + 2*alpha[b_idx[0]-1, b_idx[1]+1,b_idx[2]] - alpha[b_idx[0]-1, b_idx[1]+2,b_idx[2]]
                   + 2*alpha[b_idx[0]+1, b_idx[1]+1,b_idx[2]] - alpha[b_idx[0]+1, b_idx[1]+2,b_idx[2]]) 
          xydiff =  (-1.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]] + 3*alpha[b_idx[0],b_idx[1]+2,b_idx[2]] - 1.5*alpha[b_idx[0],b_idx[1]+3,b_idx[2]]
                     + 2*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]] - 3.5*alpha[b_idx[0]-1,b_idx[1]+2,b_idx[2]] + 1.5*alpha[b_idx[0]-1,b_idx[1]+3,b_idx[2]]
                   - 0.5*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]] + 0.5*alpha[b_idx[0]+1,b_idx[1]+2,b_idx[2]])
        elif "top" in btype or "top" in btype:
          xdiff = None # Set to None, because this term is in RHS Neumann condition
          xxdiff = (-4*alpha[b_idx[0], b_idx[1]-1,b_idx[2]] + 2*alpha[b_idx[0], b_idx[1]-2,b_idx[2]]
                   + 2*alpha[b_idx[0]-1, b_idx[1]-1,b_idx[2]] - alpha[b_idx[0]-1, b_idx[1]-2,b_idx[2]]
                   + 2*alpha[b_idx[0]+1, b_idx[1]-1,b_idx[2]] - alpha[b_idx[0]+1, b_idx[1]-2,b_idx[2]])
          xydiff = (1.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]] - 3*alpha[b_idx[0],b_idx[1]-2,b_idx[2]] + 1.5*alpha[b_idx[0],b_idx[1]-3,b_idx[2]]
                    - 2*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]] + 3.5*alpha[b_idx[0]-1,b_idx[1]-2,b_idx[2]] - 1.5*alpha[b_idx[0]-1,b_idx[1]-3,b_idx[2]]
                  + 0.5*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]] - 0.5*alpha[b_idx[0]+1,b_idx[1]-2,b_idx[2]])
        if ("shiftz" in btype) or ("shiftrear" in btype) or ("shiftfront" in btype):
          xdiff = None # Set to None, because this term is in RHS Neumann condition
          if "tob" in btype:
            xxdiff = (-4*alpha[b_idx[0], b_idx[1]+1,b_idx[2]] + 2*alpha[b_idx[0], b_idx[1]+2,b_idx[2]]
                     + 2*alpha[b_idx[0]-1, b_idx[1]+1,b_idx[2]] - alpha[b_idx[0]-1, b_idx[1]+2,b_idx[2]]
                     + 2*alpha[b_idx[0]+1, b_idx[1]+1,b_idx[2]] - alpha[b_idx[0]+1, b_idx[1]+2,b_idx[2]]) 
            xydiff =  (-1.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]] + 3*alpha[b_idx[0],b_idx[1]+2,b_idx[2]] - 1.5*alpha[b_idx[0],b_idx[1]+3,b_idx[2]]
                       + 2*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]] - 3.5*alpha[b_idx[0]-1,b_idx[1]+2,b_idx[2]] + 1.5*alpha[b_idx[0]-1,b_idx[1]+3,b_idx[2]]
                     - 0.5*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]] + 0.5*alpha[b_idx[0]+1,b_idx[1]+2,b_idx[2]])
          elif "tot" in btype:
            xxdiff = (-4*alpha[b_idx[0], b_idx[1]-1,b_idx[2]] + 2*alpha[b_idx[0], b_idx[1]-2,b_idx[2]]
                     + 2*alpha[b_idx[0]-1, b_idx[1]-1,b_idx[2]] - alpha[b_idx[0]-1, b_idx[1]-2,b_idx[2]]
                     + 2*alpha[b_idx[0]+1, b_idx[1]-1,b_idx[2]] - alpha[b_idx[0]+1, b_idx[1]-2,b_idx[2]])
            xydiff = (1.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]] - 3*alpha[b_idx[0],b_idx[1]-2,b_idx[2]] + 1.5*alpha[b_idx[0],b_idx[1]-3,b_idx[2]]
                      - 2*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]] + 3.5*alpha[b_idx[0]-1,b_idx[1]-2,b_idx[2]] - 1.5*alpha[b_idx[0]-1,b_idx[1]-3,b_idx[2]]
                    + 0.5*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]] - 0.5*alpha[b_idx[0]+1,b_idx[1]-2,b_idx[2]])
        elif "tore" in btype or "rear" in btype:
          xdiff = None # Set to None, because this term is in RHS Neumann condition
          xxdiff = (-4*alpha[b_idx[0], b_idx[1],b_idx[2]+1] + 2*alpha[b_idx[0], b_idx[1],b_idx[2]+2]
                   + 2*alpha[b_idx[0]-1, b_idx[1],b_idx[2]+1] - alpha[b_idx[0]-1, b_idx[1],b_idx[2]+2]
                   + 2*alpha[b_idx[0]+1, b_idx[1],b_idx[2]+1] - alpha[b_idx[0]+1, b_idx[1],b_idx[2]+2])
          xzdiff = (-1.5*alpha[b_idx[0],b_idx[1],b_idx[2]+1] + 3*alpha[b_idx[0],b_idx[1],b_idx[2]+2] - 1.5*alpha[b_idx[0],b_idx[1],b_idx[2]+3]
                     + 2*alpha[b_idx[0]-1,b_idx[1],b_idx[2]+1] - 3.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]+2] + 1.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]+3]
                   - 0.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]+1] + 0.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]+2])
        elif "tof" in btype or "front" in btype:
          xdiff = None # Set to None, because this term is in RHS Neumann condition
          xxdiff = (-4*alpha[b_idx[0], b_idx[1],b_idx[2]-1] + 2*alpha[b_idx[0], b_idx[1],b_idx[2]-2]
                   + 2*alpha[b_idx[0]-1, b_idx[1],b_idx[2]-1] - alpha[b_idx[0]-1, b_idx[1],b_idx[2]-2]
                   + 2*alpha[b_idx[0]+1, b_idx[1],b_idx[2]-1] - alpha[b_idx[0]+1, b_idx[1],b_idx[2]-2])
          xzdiff = (1.5*alpha[b_idx[0],b_idx[1],b_idx[2]-1] - 3*alpha[b_idx[0],b_idx[1],b_idx[2]-2] + 1.5*alpha[b_idx[0],b_idx[1],b_idx[2]-3]
                     - 2*alpha[b_idx[0]-1,b_idx[1],b_idx[2]-1] + 3.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]-2] - 1.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]-3]
                   + 0.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]-1] - 0.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]-2])
        else:
          #print('in else case 2, btype=', btype)
          xdiff = None # Set to None, because this term is in RHS Neumann condition
          if "tob" in btype or "bottom" in btype:
            xzdiff = (0.25*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]+1] - 0.25*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]-1]
                    - 0.25*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]+1] + 0.25*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]-1])
          elif "tot" in btype or "top" in btype:
            xzdiff = (0.25*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]+1] - 0.25*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]-1]
                    - 0.25*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]+1] + 0.25*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]-1])
      elif "shiftleft" in btype:
        pass
        #print("shiftleft not implemented in Ax yet")
      elif "shiftright" in btype:
        pass
        #print("shiftright not implemented in Ax yet")
      elif "left" in btype:
        xdiff = None # Set to None, because this term is in RHS Neumann condition
        xxdiff = (4*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] - 0.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]] - 3.5*alpha[b_idx[0],b_idx[1],b_idx[2]]) # The rest of this term is RHS Neumann condition
        if "shifty" in btype:
          xydiff = (-1.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] + 3*alpha[b_idx[0]+2,b_idx[1],b_idx[2]] - 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]]
                     + 2*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]] - 3.5*alpha[b_idx[0]+2,b_idx[1]-1,b_idx[2]] + 1.5*alpha[b_idx[0]+3,b_idx[1]-1,b_idx[2]]
                   - 0.5*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]] + 0.5*alpha[b_idx[0]+2,b_idx[1]+1,b_idx[2]])
        elif "shiftbottom" in btype:
          xydiff = (3*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] - 4.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]] + 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]]
                - 3.5*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]] + 5*alpha[b_idx[0]+2,b_idx[1]+1,b_idx[2]] - 1.5*alpha[b_idx[0]+3,b_idx[1]+1,b_idx[2]]
                + 0.5*alpha[b_idx[0]+1,b_idx[1]+2,b_idx[2]] - 0.5*alpha[b_idx[0]+2,b_idx[1]+2,b_idx[2]])
        elif "shifttop" in btype:
          xydiff = (-3*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] + 4.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]] - 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]]
                 + 3.5*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]] - 5*alpha[b_idx[0]+2,b_idx[1]-1,b_idx[2]] + 1.5*alpha[b_idx[0]+3,b_idx[1]-1,b_idx[2]]
                 - 0.5*alpha[b_idx[0]+1,b_idx[1]-2,b_idx[2]] + 0.5*alpha[b_idx[0]+2,b_idx[1]-2,b_idx[2]])
        elif "bottom" in btype:
          xydiff = (2*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]] - 0.25*alpha[b_idx[0]+2,b_idx[1]+2,b_idx[2]] + 1.75*alpha[b_idx[0],b_idx[1],b_idx[2]]
                  - 2*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] + 0.25*alpha[b_idx[0]+2,b_idx[1],b_idx[2]]
                  - 2*alpha[b_idx[0],b_idx[1]+1,b_idx[2]] + 0.25*alpha[b_idx[0],b_idx[1]+2,b_idx[2]])
        elif "top" in btype:
          xydiff = (-2*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]] + 0.25*alpha[b_idx[0]+2,b_idx[1]-2,b_idx[2]] - 1.75*alpha[b_idx[0],b_idx[1],b_idx[2]]
                  + 2*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] - 0.25*alpha[b_idx[0]+2,b_idx[1],b_idx[2]]
                  + 2*alpha[b_idx[0],b_idx[1]-1,b_idx[2]] - 0.25*alpha[b_idx[0],b_idx[1]-2,b_idx[2]])
        else:
          xydiff = (-alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]] + alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]]
              + 0.25*alpha[b_idx[0]+2, b_idx[1]-1,b_idx[2]] - 0.25*alpha[b_idx[0]+2, b_idx[1]+1,b_idx[2]]
              + 0.75*alpha[b_idx[0], b_idx[1]-1,b_idx[2]] - 0.75*alpha[b_idx[0], b_idx[1]+1,b_idx[2]])
        if "shiftz" in btype:
          xzdiff = (-1.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] + 3*alpha[b_idx[0]+2,b_idx[1],b_idx[2]] - 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]]
                     + 2*alpha[b_idx[0]+1,b_idx[1],b_idx[2]-1] - 3.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]-1] + 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]-1]
                   - 0.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]+1] + 0.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]+1])
        elif "shiftrear" in btype:
          xzdiff = (3*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] - 4.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]] + 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]]
                - 3.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]+1] + 5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]+1] - 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]+1]
                + 0.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]+2] - 0.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]+2])
        elif "shiftfront" in btype:
          xzdiff = (-3*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] + 4.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]] - 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]]
                 + 3.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]-1] - 5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]-1] + 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]-1]
                 - 0.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]-2] + 0.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]-2])
        elif "rear" in btype:
          xzdiff = (2*alpha[b_idx[0]+1,b_idx[1],b_idx[2]+1] - 0.25*alpha[b_idx[0]+2,b_idx[1],b_idx[2]+2] + 1.75*alpha[b_idx[0],b_idx[1],b_idx[2]]
                  - 2*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] + 0.25*alpha[b_idx[0]+2,b_idx[1],b_idx[2]]
                  - 2*alpha[b_idx[0],b_idx[1],b_idx[2]+1] + 0.25*alpha[b_idx[0],b_idx[1],b_idx[2]+2])
        elif "front" in btype:
          xzdiff = (-2*alpha[b_idx[0]+1,b_idx[1],b_idx[2]-1] + 0.25*alpha[b_idx[0]+2,b_idx[1],b_idx[2]-2] - 1.75*alpha[b_idx[0],b_idx[1],b_idx[2]]
                  + 2*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] - 0.25*alpha[b_idx[0]+2,b_idx[1],b_idx[2]]
                  + 2*alpha[b_idx[0],b_idx[1],b_idx[2]-1] - 0.25*alpha[b_idx[0],b_idx[1],b_idx[2]-2])
        else:
          xzdiff = (-alpha[b_idx[0]+1,b_idx[1],b_idx[2]-1] + alpha[b_idx[0]+1,b_idx[1],b_idx[2]+1]
              + 0.25*alpha[b_idx[0]+2, b_idx[1],b_idx[2]-1] - 0.25*alpha[b_idx[0]+2, b_idx[1],b_idx[2]+1]
              + 0.75*alpha[b_idx[0], b_idx[1],b_idx[2]-1] - 0.75*alpha[b_idx[0], b_idx[1],b_idx[2]+1])
      elif "right" in btype:
        xdiff = None # Set to None, because this term is in RHS Neumann condition
        xxdiff = (4*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] - 0.5*alpha[b_idx[0]-2,b_idx[1],b_idx[2]] - 3.5*alpha[b_idx[0],b_idx[1],b_idx[2]]) # The rest of this term is RHS Neumann condition
        if "shifty" in btype:
          xydiff = (1.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] - 3*alpha[b_idx[0]-2,b_idx[1],b_idx[2]] + 1.5*alpha[b_idx[0]-3,b_idx[1],b_idx[2]]
                    - 2*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]] + 3.5*alpha[b_idx[0]-2,b_idx[1]-1,b_idx[2]] - 1.5*alpha[b_idx[0]-3,b_idx[1]-1,b_idx[2]]
                  + 0.5*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]] - 0.5*alpha[b_idx[0]-2,b_idx[1]+1,b_idx[2]])
        elif "shiftbottom" in btype:
          xydiff = (-3*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] + 4.5*alpha[b_idx[0]-2,b_idx[1],b_idx[2]] - 1.5*alpha[b_idx[0]-3,b_idx[1],b_idx[2]]
                + 3.5*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]] - 5*alpha[b_idx[0]-2,b_idx[1]+1,b_idx[2]] + 1.5*alpha[b_idx[0]-3,b_idx[1]+1,b_idx[2]]
                - 0.5*alpha[b_idx[0]-1,b_idx[1]+2,b_idx[2]] + 0.5*alpha[b_idx[0]-2,b_idx[1]+2,b_idx[2]])
        elif "shifttop" in btype:
          xydiff = (3*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] - 4.5*alpha[b_idx[0]-2,b_idx[1],b_idx[2]] + 1.5*alpha[b_idx[0]-3,b_idx[1],b_idx[2]]
                 - 3.5*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]] + 5*alpha[b_idx[0]-2,b_idx[1]-1,b_idx[2]] - 1.5*alpha[b_idx[0]-3,b_idx[1]-1,b_idx[2]]
                 + 0.5*alpha[b_idx[0]-1,b_idx[1]-2,b_idx[2]] - 0.5*alpha[b_idx[0]-2,b_idx[1]-2,b_idx[2]])
        elif "bottom" in btype:
          xydiff = (-2*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]] + 0.25*alpha[b_idx[0]-2,b_idx[1]+2,b_idx[2]] - 1.75*alpha[b_idx[0],b_idx[1],b_idx[2]]
                  + 2*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] - 0.25*alpha[b_idx[0]-2,b_idx[1],b_idx[2]]
                  + 2*alpha[b_idx[0],b_idx[1]+1,b_idx[2]] - 0.25*alpha[b_idx[0],b_idx[1]+2,b_idx[2]])
        elif "top" in btype:
          xydiff = (+2*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]] - 0.25*alpha[b_idx[0]-2,b_idx[1]-2,b_idx[2]] + 1.75*alpha[b_idx[0],b_idx[1],b_idx[2]]
                  - 2*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] + 0.25*alpha[b_idx[0]-2,b_idx[1],b_idx[2]]
                  - 2*alpha[b_idx[0],b_idx[1]-1,b_idx[2]] + 0.25*alpha[b_idx[0],b_idx[1]-2,b_idx[2]])
        else:
          xydiff = (alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]] - alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]]
              - 0.25*alpha[b_idx[0]-2, b_idx[1]-1,b_idx[2]] + 0.25*alpha[b_idx[0]-2, b_idx[1]+1,b_idx[2]]
              - 0.75*alpha[b_idx[0], b_idx[1]-1,b_idx[2]] + 0.75*alpha[b_idx[0], b_idx[1]+1,b_idx[2]])
        if "shiftz" in btype:
          xzdiff = (1.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] - 3*alpha[b_idx[0]-2,b_idx[1],b_idx[2]] + 1.5*alpha[b_idx[0]-3,b_idx[1],b_idx[2]]
                    - 2*alpha[b_idx[0]-1,b_idx[1],b_idx[2]-1] + 3.5*alpha[b_idx[0]-2,b_idx[1],b_idx[2]-1] - 1.5*alpha[b_idx[0]-3,b_idx[1],b_idx[2]-1]
                  + 0.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]+1] - 0.5*alpha[b_idx[0]-2,b_idx[1],b_idx[2]+1])
        elif "shiftrear" in btype:
          xzdiff = (-3*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] + 4.5*alpha[b_idx[0]-2,b_idx[1],b_idx[2]] - 1.5*alpha[b_idx[0]-3,b_idx[1],b_idx[2]]
                 + 3.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]+1] - 5*alpha[b_idx[0]-2,b_idx[1],b_idx[2]+1] + 1.5*alpha[b_idx[0]-3,b_idx[1],b_idx[2]+1]
                 - 0.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]+2] + 0.5*alpha[b_idx[0]-2,b_idx[1],b_idx[2]+2])
        elif "shiftfront" in btype:
          xzdiff = (3*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] - 4.5*alpha[b_idx[0]-2,b_idx[1],b_idx[2]] + 1.5*alpha[b_idx[0]-3,b_idx[1],b_idx[2]]
                - 3.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]-1] + 5*alpha[b_idx[0]-2,b_idx[1],b_idx[2]-1] - 1.5*alpha[b_idx[0]-3,b_idx[1],b_idx[2]-1]
                + 0.5*alpha[b_idx[0]-1,b_idx[1],b_idx[2]-2] - 0.5*alpha[b_idx[0]-2,b_idx[1],b_idx[2]-2])
        elif "rear" in btype:
          xzdiff = (-2*alpha[b_idx[0]-1,b_idx[1],b_idx[2]+1] + 0.25*alpha[b_idx[0]-2,b_idx[1],b_idx[2]+2] - 1.75*alpha[b_idx[0],b_idx[1],b_idx[2]]
                  + 2*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] - 0.25*alpha[b_idx[0]-2,b_idx[1],b_idx[2]]
                  + 2*alpha[b_idx[0],b_idx[1],b_idx[2]+1] - 0.25*alpha[b_idx[0],b_idx[1],b_idx[2]+2])
        elif "front" in btype:
          xzdiff = (2*alpha[b_idx[0]-1,b_idx[1],b_idx[2]-1] - 0.25*alpha[b_idx[0]-2,b_idx[1],b_idx[2]-2] + 1.75*alpha[b_idx[0],b_idx[1],b_idx[2]]
                  - 2*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] + 0.25*alpha[b_idx[0]-2,b_idx[1],b_idx[2]]
                  - 2*alpha[b_idx[0],b_idx[1],b_idx[2]-1] + 0.25*alpha[b_idx[0],b_idx[1],b_idx[2]-2])
        else:
          xzdiff = (alpha[b_idx[0]-1,b_idx[1],b_idx[2]-1] - alpha[b_idx[0]-1,b_idx[1],b_idx[2]+1]
              - 0.25*alpha[b_idx[0]-2,b_idx[1],b_idx[2]-1] + 0.25*alpha[b_idx[0]-2, b_idx[1],b_idx[2]+1]
              - 0.75*alpha[b_idx[0],b_idx[1],b_idx[2]-1] + 0.75*alpha[b_idx[0], b_idx[1],b_idx[2]+1])
      else:
        xdiff = gradX[b_idx[0], b_idx[1],b_idx[2]]
        xxdiff = diff.gradxx_idx_3d(alpha, b_idx)
        if "shifty" in btype:
          if "tore" in btype:
            xydiff = (0.25*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]+1] - 0.25*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]+1]
                    - 0.25*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]+1] + 0.25*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]+1])
          elif "tof" in btype:
            xydiff = (0.25*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]-1] - 0.25*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]-1]
                      - 0.25*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]-1] + 0.25*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]-1])
          else:
            xydiff = (0.25*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]] - 0.25*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]]
                    - 0.25*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]] + 0.25*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]])
        elif "shiftbottom" in btype:
          # TODO figure out this case
          xydiff = (3*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] - 4.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]] + 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]]
                - 3.5*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]] + 5*alpha[b_idx[0]+2,b_idx[1]+1,b_idx[2]] - 1.5*alpha[b_idx[0]+3,b_idx[1]+1,b_idx[2]]
                + 0.5*alpha[b_idx[0]+1,b_idx[1]+2,b_idx[2]] - 0.5*alpha[b_idx[0]+2,b_idx[1]+2,b_idx[2]])
        elif "shifttop" in btype:
          # TODO figure out this case
          xydiff = (-3*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] + 4.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]] - 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]]
                + 3.5*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]] - 5*alpha[b_idx[0]+2,b_idx[1]-1,b_idx[2]] + 1.5*alpha[b_idx[0]+3,b_idx[1]-1,b_idx[2]]
                - 0.5*alpha[b_idx[0]+1,b_idx[1]-2,b_idx[2]] + 0.5*alpha[b_idx[0]+2,b_idx[1]-2,b_idx[2]])
        elif "bottom" in btype:
          xydiff = (-alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]] + alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]]
              + 0.25*alpha[b_idx[0]-1,b_idx[1]+2,b_idx[2]] - 0.25*alpha[b_idx[0]+1,b_idx[1]+2,b_idx[2]]
              + 0.75*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] - 0.75*alpha[b_idx[0]+1,b_idx[1],b_idx[2]])
        elif "top" in btype:
          xydiff = (alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]] - alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]]
              - 0.25*alpha[b_idx[0]-1,b_idx[1]-2,b_idx[2]] + 0.25*alpha[b_idx[0]+1,b_idx[1]-2,b_idx[2]]
              - 0.75*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] + 0.75*alpha[b_idx[0]+1,b_idx[1],b_idx[2]])
        else:  
          xydiff = diff.gradx_idx_3d(gradY,b_idx)

        if "shiftz" in btype:
          if "tob" in btype or "bottom" in btype:
            xzdiff = (0.25*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]+1] - 0.25*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]+1]
                    - 0.25*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]-1] + 0.25*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]-1])
          elif "tot" in btype or "top" in btype:
            xzdiff = (0.25*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]+1] - 0.25*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]+1]
                    - 0.25*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]-1] + 0.25*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]-1])
          else:
            xzdiff = (0.25*alpha[b_idx[0]+1,b_idx[1],b_idx[2]+1] - 0.25*alpha[b_idx[0]-1,b_idx[1],b_idx[2]+1]
                    - 0.25*alpha[b_idx[0]+1,b_idx[1],b_idx[2]-1] + 0.25*alpha[b_idx[0]-1,b_idx[1],b_idx[2]-1])
        elif "shiftrear" in btype:
          # TODO figure out this case
          xzdiff = (3*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] - 4.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]] + 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]]
                - 3.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]+1] + 5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]+1] - 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]+1]
                + 0.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]+2] - 0.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]+2])
        elif "shiftfront" in btype:
          # TODO figure out this case
          xzdiff = (-3*alpha[b_idx[0]+1,b_idx[1],b_idx[2]] + 4.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]] - 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]]
                + 3.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]-1] - 5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]-1] + 1.5*alpha[b_idx[0]+3,b_idx[1],b_idx[2]-1]
                - 0.5*alpha[b_idx[0]+1,b_idx[1],b_idx[2]-2] + 0.5*alpha[b_idx[0]+2,b_idx[1],b_idx[2]-2])
        elif "rear" in btype:
          xzdiff = (-alpha[b_idx[0]-1,b_idx[1],b_idx[2]+1] + alpha[b_idx[0]+1,b_idx[1],b_idx[2]+1]
              + 0.25*alpha[b_idx[0]-1,b_idx[1],b_idx[2]+2] - 0.25*alpha[b_idx[0]+1,b_idx[1],b_idx[2]+2]
              + 0.75*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] - 0.75*alpha[b_idx[0]+1,b_idx[1],b_idx[2]])
        elif "front" in btype:
          xzdiff = (alpha[b_idx[0]-1,b_idx[1],b_idx[2]-1] - alpha[b_idx[0]+1,b_idx[1],b_idx[2]-1]
              - 0.25*alpha[b_idx[0]-1,b_idx[1],b_idx[2]-2] + 0.25*alpha[b_idx[0]+1,b_idx[1],b_idx[2]-2]
              - 0.75*alpha[b_idx[0]-1,b_idx[1],b_idx[2]] + 0.75*alpha[b_idx[0]+1,b_idx[1],b_idx[2]])
        else:    
          xzdiff = diff.gradx_idx_3d(gradZ,b_idx)
      if "shifty" in btype:
        if ("shiftx" in btype) or ("shiftleft" in btype) or ("shiftright" in btype):
          ydiff = None # Set to None, because this term is in RHS Neumann condition
          if "rear" in btype:
            yydiff = (-4*alpha[b_idx[0], b_idx[1],b_idx[2]+1] + 2*alpha[b_idx[0], b_idx[1],b_idx[2]+2]
                   + 2*alpha[b_idx[0], b_idx[1]-1,b_idx[2]+1] - alpha[b_idx[0], b_idx[1]-1,b_idx[2]+2]
                   + 2*alpha[b_idx[0], b_idx[1]+1,b_idx[2]+1] - alpha[b_idx[0], b_idx[1]+1,b_idx[2]+2])
            yzdiff = (-1.5*alpha[b_idx[0],b_idx[1],b_idx[2]+1] + 3*alpha[b_idx[0],b_idx[1],b_idx[2]+2] - 1.5*alpha[b_idx[0],b_idx[1],b_idx[2]+3]
                       + 2*alpha[b_idx[0],b_idx[1]-1,b_idx[2]+1] - 3.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]+2] + 1.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]+3]
                     - 0.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]+1] + 0.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]+2])
            # TODO figure out xydiff in this case
            #xydiff = diff.gradx_idx_3d(gradY, [b_idx[0],b_idx[1],b_idx[2]+1])
            xydiff = (-0.25*alpha[b_idx[0]-1, b_idx[1]+1,b_idx[2]+1] - 0.25*alpha[b_idx[0]+1, b_idx[1]-1,b_idx[2]+1]
                     + 0.25*alpha[b_idx[0]+1, b_idx[1]+1,b_idx[2]+1] + 0.25*alpha[b_idx[0]-1, b_idx[1]-1,b_idx[2]+1])
          elif "tof" in btype:
            yydiff = (-4*alpha[b_idx[0], b_idx[1],b_idx[2]-1] + 2*alpha[b_idx[0], b_idx[1],b_idx[2]-2]
                   + 2*alpha[b_idx[0], b_idx[1]-1,b_idx[2]-1] - alpha[b_idx[0], b_idx[1]-1,b_idx[2]-2]
                   + 2*alpha[b_idx[0], b_idx[1]+1,b_idx[2]-1] - alpha[b_idx[0], b_idx[1]+1,b_idx[2]-2])
            yzdiff = (1.5*alpha[b_idx[0],b_idx[1],b_idx[2]-1] - 3*alpha[b_idx[0],b_idx[1],b_idx[2]-2] + 1.5*alpha[b_idx[0],b_idx[1],b_idx[2]-3]
                      - 2*alpha[b_idx[0],b_idx[1]-1,b_idx[2]-1] + 3.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]-2] - 1.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]-3]
                     + 0.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]-1] - 0.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]-2])
            # TODO figure out xydiff in this case
            #xydiff = diff.gradx_idx_3d(gradY, [b_idx[0],b_idx[1],b_idx[2]-1])
            xydiff = (-0.25*alpha[b_idx[0]-1, b_idx[1]+1,b_idx[2]+1] - 0.25*alpha[b_idx[0]+1, b_idx[1]-1,b_idx[2]+1]
                     + 0.25*alpha[b_idx[0]+1, b_idx[1]+1,b_idx[2]+1] + 0.25*alpha[b_idx[0]-1, b_idx[1]-1,b_idx[2]+1])
        elif "tol" in btype or "left" in btype:
          ydiff = None # Set to None, because this term is in RHS Neumann condition
          yydiff = (-4*alpha[b_idx[0]+1, b_idx[1],b_idx[2]] + 2*alpha[b_idx[0]+2, b_idx[1],b_idx[2]]
                   + 2*alpha[b_idx[0]+1, b_idx[1]-1,b_idx[2]] - alpha[b_idx[0]+2, b_idx[1]-1,b_idx[2]]
                   + 2*alpha[b_idx[0]+1, b_idx[1]+1,b_idx[2]] - alpha[b_idx[0]+2, b_idx[1]+1,b_idx[2]])
          #xydiff taken care of already
        elif "tori" in btype or "right" in btype:
          ydiff = None # Set to None, because this term is in RHS Neumann condition
          yydiff = (-4*alpha[b_idx[0]-1, b_idx[1],b_idx[2]] + 2*alpha[b_idx[0]-2, b_idx[1],b_idx[2]]
                   + 2*alpha[b_idx[0]-1, b_idx[1]-1,b_idx[2]] - alpha[b_idx[0]-2, b_idx[1]-1,b_idx[2]]
                   + 2*alpha[b_idx[0]-1, b_idx[1]+1,b_idx[2]] - alpha[b_idx[0]-2, b_idx[1]+1,b_idx[2]])
          #xydiff taken care of already
        if ("shiftz" in btype) or ("shiftrear" in btype) or ("shiftfront" in btype):
          ydiff = None # Set to None, because this term is in RHS Neumann condition
          if "tol" in btype:
            yydiff = (-4*alpha[b_idx[0]+1, b_idx[1],b_idx[2]] + 2*alpha[b_idx[0]+2, b_idx[1],b_idx[2]]
                   + 2*alpha[b_idx[0]+1, b_idx[1]-1,b_idx[2]] - alpha[b_idx[0]+2, b_idx[1]-1,b_idx[2]]
                   + 2*alpha[b_idx[0]+1, b_idx[1]+1,b_idx[2]] - alpha[b_idx[0]+2, b_idx[1]+1,b_idx[2]])
            # TODO figure out yzdiff in this case
            yzdiff = diff.grady_idx_3d(gradZ, [b_idx[0],b_idx[1],b_idx[2]+1])
          elif "tori" in btype:
            yydiff = (-4*alpha[b_idx[0]-1, b_idx[1],b_idx[2]] + 2*alpha[b_idx[0]-2, b_idx[1],b_idx[2]]
                   + 2*alpha[b_idx[0]-1, b_idx[1]-1,b_idx[2]] - alpha[b_idx[0]-2, b_idx[1]-1,b_idx[2]]
                   + 2*alpha[b_idx[0]-1, b_idx[1]+1,b_idx[2]] - alpha[b_idx[0]-2, b_idx[1]+1,b_idx[2]])
            # TODO figure out yzdiff in this case
            yzdiff = diff.grady_idx_3d(gradZ, [b_idx[0],b_idx[1],b_idx[2]-1])
        elif "tore" in btype or "rear" in btype:
          ydiff = None # Set to None, because this term is in RHS Neumann condition
          yydiff = (-4*alpha[b_idx[0], b_idx[1],b_idx[2]+1] + 2*alpha[b_idx[0], b_idx[1],b_idx[2]+2]
                   + 2*alpha[b_idx[0], b_idx[1]-1,b_idx[2]+1] - alpha[b_idx[0], b_idx[1]-1,b_idx[2]+2]
                   + 2*alpha[b_idx[0], b_idx[1]+1,b_idx[2]+1] - alpha[b_idx[0], b_idx[1]+1,b_idx[2]+2])
          yzdiff = (-1.5*alpha[b_idx[0],b_idx[1],b_idx[2]+1] + 3*alpha[b_idx[0],b_idx[1],b_idx[2]+2] - 1.5*alpha[b_idx[0],b_idx[1],b_idx[2]+3]
                     + 2*alpha[b_idx[0],b_idx[1]-1,b_idx[2]+1] - 3.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]+2] + 1.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]+3]
                   - 0.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]+1] + 0.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]+2])
        elif "tof" in btype or "front" in btype:
          ydiff = None # Set to None, because this term is in RHS Neumann condition
          yydiff = (-4*alpha[b_idx[0], b_idx[1],b_idx[2]-1] + 2*alpha[b_idx[0], b_idx[1],b_idx[2]-2]
                   + 2*alpha[b_idx[0], b_idx[1]-1,b_idx[2]-1] - alpha[b_idx[0], b_idx[1]-1,b_idx[2]-2]
                   + 2*alpha[b_idx[0], b_idx[1]+1,b_idx[2]-1] - alpha[b_idx[0], b_idx[1]+1,b_idx[2]-2])
          yzdiff = (1.5*alpha[b_idx[0],b_idx[1],b_idx[2]-1] - 3*alpha[b_idx[0],b_idx[1],b_idx[2]-2] + 1.5*alpha[b_idx[0],b_idx[1],b_idx[2]-3]
                     - 2*alpha[b_idx[0],b_idx[1]-1,b_idx[2]-1] + 3.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]-2] - 1.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]-3]
                   + 0.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]-1] - 0.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]-2])
        else:
          ydiff = None # Set to None, because this term is in RHS Neumann condition
          # TODO confirm yydiff in this case. yzdiff is good already
          #yydiff = (-4*alpha[b_idx[0], b_idx[1],b_idx[2]-1] + 2*alpha[b_idx[0], b_idx[1],b_idx[2]-2]
          #           + alpha[b_idx[0], b_idx[1]-1,b_idx[2]-1] - alpha[b_idx[0], b_idx[1]-1,b_idx[2]-2]
          #         + 2*alpha[b_idx[0], b_idx[1]+1,b_idx[2]-1] - alpha[b_idx[0], b_idx[1]+1,b_idx[2]-2])
          #yzdiff = (0.25*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]+1] - 0.25*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]-1]
          #        - 0.25*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]+1] + 0.25*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]-1])
          if "tol" in btype or "left" in btype:
            yzdiff = (0.25*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]+1] - 0.25*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]-1]
                    - 0.25*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]+1] + 0.25*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]-1])
          elif "tori" in btype or "right" in btype:
            yzdiff = (0.25*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]+1] - 0.25*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]-1]
                    - 0.25*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]+1] + 0.25*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]-1])
          
      elif "bottom" in btype:
        ydiff = None # Set to None, because this term is in RHS Neumann condition
        yydiff = (4*alpha[b_idx[0], b_idx[1]+1,b_idx[2]] - 0.5 * alpha[b_idx[0], b_idx[1]+2,b_idx[2]] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]])
        # NOTE xydiff should already be set by this point
        if "shiftz" in btype:
          yzdiff = (-1.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]] + 3*alpha[b_idx[0],b_idx[1]+2,b_idx[2]] - 1.5*alpha[b_idx[0],b_idx[1]+3,b_idx[2]]
                     + 2*alpha[b_idx[0],b_idx[1]+1,b_idx[2]-1] - 3.5*alpha[b_idx[0],b_idx[1]+2,b_idx[2]-1] + 1.5*alpha[b_idx[0],b_idx[1]+3,b_idx[2]-1]
                   - 0.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]+1] + 0.5*alpha[b_idx[0],b_idx[1]+2,b_idx[2]+1])
        elif "shiftrear" in btype:
          yzdiff = (3*alpha[b_idx[0],b_idx[1]+1,b_idx[2]] - 4.5*alpha[b_idx[0],b_idx[1]+2,b_idx[2]] + 1.5*alpha[b_idx[0],b_idx[1]+3,b_idx[2]]
                - 3.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]+1] + 5*alpha[b_idx[0],b_idx[1]+2,b_idx[2]+1] - 1.5*alpha[b_idx[0],b_idx[1]+3,b_idx[2]+1]
                + 0.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]+2] - 0.5*alpha[b_idx[0],b_idx[1]+2,b_idx[2]+2])
        elif "shiftfront" in btype:
          yzdiff = (-3*alpha[b_idx[0],b_idx[1]+1,b_idx[2]] + 4.5*alpha[b_idx[0],b_idx[1]+2,b_idx[2]] - 1.5*alpha[b_idx[0],b_idx[1]+3,b_idx[2]]
                 + 3.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]-1] - 5*alpha[b_idx[0],b_idx[1]+2,b_idx[2]-1] + 1.5*alpha[b_idx[0],b_idx[1]+3,b_idx[2]-1]
                 - 0.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]-2] + 0.5*alpha[b_idx[0],b_idx[1]+2,b_idx[2]-2])
        elif "rear" in btype:
          yzdiff = (2*alpha[b_idx[0],b_idx[1]+1,b_idx[2]+1] - 0.25*alpha[b_idx[0],b_idx[1]+2,b_idx[2]+2] + 1.75*alpha[b_idx[0],b_idx[1],b_idx[2]]
                  - 2*alpha[b_idx[0],b_idx[1]+1,b_idx[2]] + 0.25*alpha[b_idx[0],b_idx[1]+2,b_idx[2]]
                  - 2*alpha[b_idx[0],b_idx[1],b_idx[2]+1] + 0.25*alpha[b_idx[0],b_idx[1],b_idx[2]+2])
        elif "front" in btype:
          yzdiff = (-2*alpha[b_idx[0],b_idx[1]+1,b_idx[2]-1] + 0.25*alpha[b_idx[0],b_idx[1]+2,b_idx[2]-2] - 1.75*alpha[b_idx[0],b_idx[1],b_idx[2]]
                  + 2*alpha[b_idx[0],b_idx[1]+1,b_idx[2]] - 0.25*alpha[b_idx[0],b_idx[1]+2,b_idx[2]]
                  + 2*alpha[b_idx[0],b_idx[1],b_idx[2]-1] - 0.25*alpha[b_idx[0],b_idx[1],b_idx[2]-2])
        else:
          yzdiff = (-alpha[b_idx[0],b_idx[1]+1,b_idx[2]-1] + alpha[b_idx[0],b_idx[1]+1,b_idx[2]+1]
              + 0.25*alpha[b_idx[0], b_idx[1]+2,b_idx[2]-1] - 0.25*alpha[b_idx[0], b_idx[1]+2,b_idx[2]+1]
              + 0.75*alpha[b_idx[0], b_idx[1],b_idx[2]-1] - 0.75*alpha[b_idx[0], b_idx[1],b_idx[2]+1])
      elif "top" in btype:
        ydiff = None # Set to None, because this term is in RHS Neumann condition
        yydiff = (4*alpha[b_idx[0], b_idx[1]-1,b_idx[2]] - 0.5 * alpha[b_idx[0], b_idx[1]-2,b_idx[2]] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]])
        # NOTE xydiff should already be set by this point
        if "shiftz" in btype:
          yzdiff = (1.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]] - 3*alpha[b_idx[0],b_idx[1]-2,b_idx[2]] + 1.5*alpha[b_idx[0],b_idx[1]-3,b_idx[2]]
                     - 2*alpha[b_idx[0],b_idx[1]-1,b_idx[2]-1] + 3.5*alpha[b_idx[0],b_idx[1]-2,b_idx[2]-1] - 1.5*alpha[b_idx[0],b_idx[1]-3,b_idx[2]-1]
                   + 0.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]+1] - 0.5*alpha[b_idx[0],b_idx[1]-2,b_idx[2]+1])
        elif "shiftrear" in btype:
          yzdiff = (-3*alpha[b_idx[0],b_idx[1]-1,b_idx[2]] + 4.5*alpha[b_idx[0],b_idx[1]-2,b_idx[2]] - 1.5*alpha[b_idx[0],b_idx[1]-3,b_idx[2]]
                + 3.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]+1] - 5*alpha[b_idx[0],b_idx[1]-2,b_idx[2]+1] + 1.5*alpha[b_idx[0],b_idx[1]-3,b_idx[2]+1]
                - 0.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]+2] + 0.5*alpha[b_idx[0],b_idx[1]-2,b_idx[2]+2])
        elif "shiftfront" in btype:
         yzdiff = (3*alpha[b_idx[0],b_idx[1]-1,b_idx[2]] - 4.5*alpha[b_idx[0],b_idx[1]-2,b_idx[2]] + 1.5*alpha[b_idx[0],b_idx[1]-3,b_idx[2]]
                 - 3.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]-1] + 5*alpha[b_idx[0],b_idx[1]-2,b_idx[2]-1] - 1.5*alpha[b_idx[0],b_idx[1]-3,b_idx[2]-1]
                 + 0.5*alpha[b_idx[0],b_idx[1]-1,b_idx[2]-2] - 0.5*alpha[b_idx[0],b_idx[1]-2,b_idx[2]-2])
        elif "rear" in btype:
          yzdiff = (-2*alpha[b_idx[0],b_idx[1]-1,b_idx[2]+1] + 0.25*alpha[b_idx[0],b_idx[1]-2,b_idx[2]+2] - 1.75*alpha[b_idx[0],b_idx[1],b_idx[2]]
                  + 2*alpha[b_idx[0],b_idx[1]-1,b_idx[2]] - 0.25*alpha[b_idx[0],b_idx[1]-2,b_idx[2]]
                  + 2*alpha[b_idx[0],b_idx[1],b_idx[2]+1] - 0.25*alpha[b_idx[0],b_idx[1],b_idx[2]+2])
        elif "front" in btype:
          yzdiff = (2*alpha[b_idx[0],b_idx[1]-1,b_idx[2]-1] - 0.25*alpha[b_idx[0],b_idx[1]-2,b_idx[2]-2] + 1.75*alpha[b_idx[0],b_idx[1],b_idx[2]]
                  - 2*alpha[b_idx[0],b_idx[1]-1,b_idx[2]] + 0.25*alpha[b_idx[0],b_idx[1]-2,b_idx[2]]
                  - 2*alpha[b_idx[0],b_idx[1],b_idx[2]-1] + 0.25*alpha[b_idx[0],b_idx[1],b_idx[2]-2])
        else:
          yzdiff = (alpha[b_idx[0],b_idx[1]-1,b_idx[2]-1] - alpha[b_idx[0],b_idx[1]-1,b_idx[2]+1]
              - 0.25*alpha[b_idx[0], b_idx[1]-2,b_idx[2]-1] + 0.25*alpha[b_idx[0], b_idx[1]-2,b_idx[2]+1]
              - 0.75*alpha[b_idx[0], b_idx[1],b_idx[2]-1] + 0.75*alpha[b_idx[0], b_idx[1],b_idx[2]+1])
      else:
        ydiff = gradY[b_idx[0], b_idx[1],b_idx[2]]
        yydiff = diff.gradyy_idx_3d(alpha, b_idx)
        #xydiff should already be calculated by this point
        if "shiftz" in btype:
          if "tol" in btype or "left" in btype:
            yzdiff = (0.25*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]+1] - 0.25*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]+1]
                    - 0.25*alpha[b_idx[0]+1,b_idx[1]+1,b_idx[2]-1] + 0.25*alpha[b_idx[0]+1,b_idx[1]-1,b_idx[2]-1])
          elif "tori" in btype or "right" in btype:
            yzdiff = (0.25*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]+1] - 0.25*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]+1]
                    - 0.25*alpha[b_idx[0]-1,b_idx[1]+1,b_idx[2]-1] + 0.25*alpha[b_idx[0]-1,b_idx[1]-1,b_idx[2]-1])
          else:
            yzdiff = (0.25*alpha[b_idx[0],b_idx[1]+1,b_idx[2]+1] - 0.25*alpha[b_idx[0],b_idx[1]-1,b_idx[2]+1]
                    - 0.25*alpha[b_idx[0],b_idx[1]+1,b_idx[2]-1] + 0.25*alpha[b_idx[0],b_idx[1]-1,b_idx[2]-1])
        elif "shiftrear" in btype:
          # TODO figure out this case
          yzdiff = (3*alpha[b_idx[0],b_idx[1]+1,b_idx[2]] - 4.5*alpha[b_idx[0],b_idx[1]+2,b_idx[2]] + 1.5*alpha[b_idx[0],b_idx[1]+3,b_idx[2]]
                - 3.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]+1] + 5*alpha[b_idx[0],b_idx[1]+2,b_idx[2]+1] - 1.5*alpha[b_idx[0],b_idx[1]+3,b_idx[2]+1]
                + 0.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]+2] - 0.5*alpha[b_idx[0],b_idx[1]+2,b_idx[2]+2])
        elif "shiftfront" in btype:
          # TODO figure out this case
          yzdiff = (-3*alpha[b_idx[0],b_idx[1]+1,b_idx[2]] + 4.5*alpha[b_idx[0],b_idx[1]+2,b_idx[2]] - 1.5*alpha[b_idx[0],b_idx[1]+3,b_idx[2]]
                + 3.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]-1] - 5*alpha[b_idx[0],b_idx[1]+2,b_idx[2]-1] + 1.5*alpha[b_idx[0],b_idx[1]+3,b_idx[2]-1]
                - 0.5*alpha[b_idx[0],b_idx[1]+1,b_idx[2]-2] + 0.5*alpha[b_idx[0],b_idx[1]+2,b_idx[2]-2])
        elif "rear" in btype:
          yzdiff = (-alpha[b_idx[0],b_idx[1]-1,b_idx[2]+1] + alpha[b_idx[0],b_idx[1]+1,b_idx[2]+1]
              + 0.25*alpha[b_idx[0],b_idx[1]-1,b_idx[2]+2] - 0.25*alpha[b_idx[0],b_idx[1]+1,b_idx[2]+2]
              + 0.75*alpha[b_idx[0],b_idx[1]-1,b_idx[2]] - 0.75*alpha[b_idx[0],b_idx[1]+1,b_idx[2]])
        elif "front" in btype:
          yzdiff = (alpha[b_idx[0],b_idx[1]-1,b_idx[2]-1] - alpha[b_idx[0],b_idx[1]+1,b_idx[2]-1]
              - 0.25*alpha[b_idx[0],b_idx[1]-1,b_idx[2]-2] + 0.25*alpha[b_idx[0],b_idx[1]+1,b_idx[2]-2]
              - 0.75*alpha[b_idx[0],b_idx[1]-1,b_idx[2]] + 0.75*alpha[b_idx[0],b_idx[1]+1,b_idx[2]])
        else:    
          yzdiff = diff.gradz_idx_3d(gradY,b_idx)
      if "shiftz" in btype:
        if ("shiftx" in btype) or ("shiftleft" in btype) or ("shiftright" in btype):
          zdiff = None # Set to None, because this term is in RHS Neumann condition
          if "tob" in btype:
            zzdiff = (-4*alpha[b_idx[0], b_idx[1]+1,b_idx[2]] + 2*alpha[b_idx[0], b_idx[1]+2,b_idx[2]]
                     + 2*alpha[b_idx[0], b_idx[1]+1,b_idx[2]-1] - alpha[b_idx[0], b_idx[1]+2,b_idx[2]-1]
                     + 2*alpha[b_idx[0], b_idx[1]+1,b_idx[2]+1] - alpha[b_idx[0], b_idx[1]+2,b_idx[2]+1])
          elif "tot" in btype:
            zzdiff = (-4*alpha[b_idx[0], b_idx[1]-1,b_idx[2]] + 2*alpha[b_idx[0], b_idx[1]-2,b_idx[2]]
                     + 2*alpha[b_idx[0], b_idx[1]-1,b_idx[2]-1] - alpha[b_idx[0], b_idx[1]-2,b_idx[2]-1]
                     + 2*alpha[b_idx[0], b_idx[1]-1,b_idx[2]+1] - alpha[b_idx[0], b_idx[1]-2,b_idx[2]+1])
        elif "tol" in btype:
          zdiff = None # Set to None, because this term is in RHS Neumann condition
          zzdiff = (-4*alpha[b_idx[0]+1, b_idx[1],b_idx[2]] + 2*alpha[b_idx[0]+2, b_idx[1],b_idx[2]]
                   + 2*alpha[b_idx[0]+1, b_idx[1],b_idx[2]-1] - alpha[b_idx[0]+2, b_idx[1],b_idx[2]-1]
                   + 2*alpha[b_idx[0]+1, b_idx[1],b_idx[2]+1] - alpha[b_idx[0]+2, b_idx[1],b_idx[2]+1])
        elif "tori" in btype:
          zdiff = None # Set to None, because this term is in RHS Neumann condition
          zzdiff = (-4*alpha[b_idx[0]-1, b_idx[1],b_idx[2]] + 2*alpha[b_idx[0]-2, b_idx[1],b_idx[2]]
                   + 2*alpha[b_idx[0]-1, b_idx[1],b_idx[2]-1] - alpha[b_idx[0]-2, b_idx[1],b_idx[2]-1]
                   + 2*alpha[b_idx[0]-1, b_idx[1],b_idx[2]+1] - alpha[b_idx[0]-2, b_idx[1],b_idx[2]+1])
          
        if ("shifty" in btype) or ("shiftbottom" in btype) or ("shifttop" in btype):
          zdiff = None # Set to None, because this term is in RHS Neumann condition
          if "tol" in btype:
            zzdiff = (-4*alpha[b_idx[0]+1, b_idx[1],b_idx[2]] + 2*alpha[b_idx[0]+2, b_idx[1],b_idx[2]]
                     + 2*alpha[b_idx[0]+1, b_idx[1],b_idx[2]-1] - alpha[b_idx[0]+2, b_idx[1],b_idx[2]-1]
                     + 2*alpha[b_idx[0]+1, b_idx[1],b_idx[2]+1] - alpha[b_idx[0]+2, b_idx[1],b_idx[2]+1])
          elif "tori" in btype:
            zzdiff = (-4*alpha[b_idx[0]-1, b_idx[1],b_idx[2]] + 2*alpha[b_idx[0]-2, b_idx[1],b_idx[2]]
                     + 2*alpha[b_idx[0]-1, b_idx[1],b_idx[2]-1] - alpha[b_idx[0]-2, b_idx[1],b_idx[2]-1]
                     + 2*alpha[b_idx[0]-1, b_idx[1],b_idx[2]+1] - alpha[b_idx[0]-2, b_idx[1],b_idx[2]+1])
        elif "tob" in btype:
          zdiff = None # Set to None, because this term is in RHS Neumann condition
          zzdiff = (-4*alpha[b_idx[0], b_idx[1]+1,b_idx[2]] + 2*alpha[b_idx[0], b_idx[1]+2,b_idx[2]]
                   + 2*alpha[b_idx[0], b_idx[1]+1,b_idx[2]-1] - alpha[b_idx[0], b_idx[1]+2,b_idx[2]-1]
                   + 2*alpha[b_idx[0], b_idx[1]+1,b_idx[2]+1] - alpha[b_idx[0], b_idx[1]+2,b_idx[2]+1])
        elif "tot" in btype:
          zdiff = None # Set to None, because this term is in RHS Neumann condition
          zzdiff = (-4*alpha[b_idx[0], b_idx[1]-1,b_idx[2]] + 2*alpha[b_idx[0], b_idx[1]-2,b_idx[2]]
                   + 2*alpha[b_idx[0], b_idx[1]-1,b_idx[2]-1] - alpha[b_idx[0], b_idx[1]-2,b_idx[2]-1]
                   + 2*alpha[b_idx[0], b_idx[1]-1,b_idx[2]+1] - alpha[b_idx[0], b_idx[1]-2,b_idx[2]+1])
      elif "rear" in btype:
        zdiff = None # Set to None, because this term is in RHS Neumann condition
        zzdiff = (4*alpha[b_idx[0], b_idx[1],b_idx[2]+1] - 0.5 * alpha[b_idx[0], b_idx[1],b_idx[2]+2] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]])
        # xzdiff, yzdiff should already be set by this point
      elif "front" in btype:
        zdiff = None # Set to None, because this term is in RHS Neumann condition
        zzdiff = (4*alpha[b_idx[0], b_idx[1],b_idx[2]-1] - 0.5 * alpha[b_idx[0], b_idx[1],b_idx[2]-2] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]])
        # xzdiff, yzdiff should already be set by this point
      else:
        zdiff = gradZ[b_idx[0], b_idx[1],b_idx[2]]
        zzdiff = diff.gradzz_idx_3d(alpha, b_idx)
        # xzdiff, yzdiff should already be set by this point
      
      # div[b_idx[0], b_idx[1],b_idx[2]] = (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,0,0])* xdiff \
      #                                   + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * xxdiff \
      #                                  + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,1,0])* ydiff \
      #                                   + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1] * xydiff \
      #                                  + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,2,0])* zdiff \
      #                                   + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,2] * xzdiff \
      #                                  + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,0,1])* xdiff \
      #                                   + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] * xydiff \
      #                                  + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,1,1])* ydiff \
      #                                   + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * yydiff \
      #                                  + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,2,1])* zdiff \
      #                                   + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,2] * yzdiff \
      #                                  + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],2,0,2])* xdiff \
      #                                   + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,0] * xzdiff \
      #                                  + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],2,1,2])* ydiff \
      #                                   + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,1] * yzdiff \
      #                                  + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],2,2,2])* zdiff \
      #                                   + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2] * zzdiff
      try:
        if type(xxdiff) == 'int' and xxdiff == 0:
          print('xxdiff == 0, btype=', btype)
        if type(xydiff) == 'int' and xydiff == 0:
          print('xydiff == 0, btype=', btype)
        if type(xzdiff) == 'int' and xzdiff == 0:
          print('xzdiff == 0, btype=', btype)
        if type(yydiff) == 'int' and yydiff == 0:
          print('yydiff == 0, btype=', btype)
        if type(yzdiff) == 'int' and yzdiff == 0:
          print('yzdiff == 0, btype=', btype)
        if type(zzdiff) == 'int' and zzdiff == 0:
          print('zzdiff == 0, btype=', btype)

        
        div[b_idx[0], b_idx[1],b_idx[2]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * xxdiff \
                                         + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1] * xydiff \
                                         + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,2] * xzdiff \
                                         + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] * xydiff \
                                         + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * yydiff \
                                         + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,2] * yzdiff \
                                         + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,0] * xzdiff \
                                         + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,1] * yzdiff \
                                         + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2] * zzdiff
      except:
        print('btype:', btype)
        print('det_g_x_g_inv shape:', args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2]].shape)
        print('xxdiff shape:', xxdiff.shape,'xydiff shape:', xydiff.shape,'xzdiff shape:', xzdiff.shape)
        print('yydiff shape:', yydiff.shape,'yzdiff shape:', yzdiff.shape,'zzdiff shape:', zzdiff.shape)
        raise
        
      if xdiff is not None:
        div[b_idx[0], b_idx[1],b_idx[2]] += (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,0,0] 
                                           + args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,0,1]
                                           + args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],2,0,2]) * xdiff
      if ydiff is not None:
        div[b_idx[0], b_idx[1],b_idx[2]] += (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,1,0]
                                           + args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,1,1]
                                           + args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],2,1,2]) * ydiff
      if zdiff is not None:
        div[b_idx[0], b_idx[1],b_idx[2]] += (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,2,0]
                                           + args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,2,1]
                                           + args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],2,2,2]) * zdiff

                                              


    # elif btype == "left":   
    #   gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_3d(gradY, b_idx)
    #   gradXZ[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_3d(gradZ, b_idx)
    #   gradYY[b_idx[0], b_idx[1],b_idx[2]] = diff.gradyy_idx_3d(alpha, b_idx)
    #   gradYZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradz_idx_3d(gradY, b_idx)
    #   gradZZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradzz_idx_3d(alpha, b_idx)
    #   div[b_idx[0], b_idx[1],b_idx[2]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * (4*alpha[b_idx[0]+1, b_idx[1],b_idx[2]] - 0.5 * alpha[b_idx[0]+2, b_idx[1],b_idx[2]] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]]) \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,1,0])* gradY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,1,1])* gradY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],2,1,2])* gradY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,2,0])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,2,1])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],2,2,2])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1]) * gradXY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,2]) * gradXZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,2] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,1]) * gradYZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * gradYY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2] * gradZZ[b_idx[0], b_idx[1],b_idx[2]]
    #   #if ii == 7 and jj == 51:
    #   #  print(ii,jj,div[ii,jj],(4*alpha[ii+1,jj] - 0.5 * alpha[ii+2,jj] - 3.5 * alpha[ii,jj]), \
    #     #        gradY[ii,jj], gradxy, grady2)

    # elif btype == "bottomleft":
    #   gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_3d(gradY, b_idx)
    #   gradXZ[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_3d(gradZ, b_idx)
    #   gradYZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradz_idx_3d(gradY, b_idx)
    #   gradZZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradzz_idx_3d(alpha, b_idx)

    #   div[b_idx[0], b_idx[1],b_idx[2]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * (4*alpha[b_idx[0]+1, b_idx[1],b_idx[2]] - 0.5 * alpha[b_idx[0]+2, b_idx[1],b_idx[2]] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]]) \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,2,0])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,2,1])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],2,2,2])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1]) * gradXY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,2]) * gradXZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,2] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,1]) * gradYZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * (4*alpha[b_idx[0], b_idx[1]+1,b_idx[2]] - 0.5 * alpha[b_idx[0], b_idx[1]+2,b_idx[2]] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]]) \
    #                             + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2] * gradZZ[b_idx[0], b_idx[1],b_idx[2]]
 
    # elif  btype == "topleft":
    #   gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_3d(gradY, b_idx)
    #   gradXZ[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_3d(gradZ, b_idx)
    #   gradYZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradz_idx_3d(gradY, b_idx)
    #   gradZZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradzz_idx_3d(alpha, b_idx)

    #   div[b_idx[0], b_idx[1],b_idx[2]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * (4*alpha[b_idx[0]+1, b_idx[1],b_idx[2]] - 0.5 * alpha[b_idx[0]+2, b_idx[1],b_idx[2]] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]]) \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,2,0])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,2,1])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],2,2,2])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1]) * gradXY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,2]) * gradXZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,2] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,1]) * gradYZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                            + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * (4*alpha[b_idx[0], b_idx[1]-1,b_idx[2]] - 0.5 * alpha[b_idx[0], b_idx[1]-2,b_idx[2]] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]]) \
    #                             + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2] * gradZZ[b_idx[0], b_idx[1],b_idx[2]]
 
    # elif btype == "notright":
    #   HERE HERE HERE - TRYING TO FIGURE OUT BCS FOR 3D CASE
    #   gradXX[b_idx[0], b_idx[1],b_idx[2]] = (alpha[b_idx[0]+2, b_idx[1],b_idx[2]] - 2 * alpha[b_idx[0]+1, b_idx[1],b_idx[2]] + alpha[b_idx[0], b_idx[1],b_idx[2]])
    #   gradXZ[b_idx[0], b_idx[1],b_idx[2]] = diff.left_diff_idx_3d(gradZ,[b_idx[0]+1, b_idx[1], b_idx[2]])
    #   gradYY[b_idx[0], b_idx[1],b_idx[2]] = (alpha[b_idx[0]+1, b_idx[1]+1,b_idx[2]] - 2 * alpha[b_idx[0]+1, b_idx[1],b_idx[2]] + alpha[b_idx[0]+1, b_idx[1]-1,b_idx[2]])
    #   gradYZ[b_idx[0], b_idx[1],b_idx[2]] = diff.gradz_idx_3d(gradY,[b_idx[0]+1, b_idx[1], b_idx[2]])
    #   gradZZ[b_idx[0], b_idx[1],b_idx[2]] = (alpha[b_idx[0]+1, b_idx[1],b_idx[2]+1] - 2 * alpha[b_idx[0]+1, b_idx[1],b_idx[2]] + alpha[b_idx[0]+1, b_idx[1],b_idx[2]-1])
    #   div[b_idx[0], b_idx[1],b_idx[2]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * gradXX[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,2,0])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,2,1])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2,2] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],2,2,2])* gradZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1]) \
    #                             * (0.5*alpha[b_idx[0]+1, b_idx[1]+1,b_idx[2]]-1.5*alpha[b_idx[0], b_idx[1],b_idx[2]]+2*alpha[b_idx[0]+1, b_idx[1],b_idx[2]] \
    #                                - 0.5*alpha[b_idx[0]+1, b_idx[1]-1,b_idx[2]]-0.5*alpha[b_idx[0]+2, b_idx[1],b_idx[2]]) \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,2]) * gradXZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,2] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,1]) * gradYZ[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * gradYY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],2,2] * gradZZ[b_idx[0], b_idx[1],b_idx[2]]
                     
    # elif  btype == "right":
    #   gradYY[b_idx[0], b_idx[1],b_idx[2]] = (alpha[b_idx[0], b_idx[1]+1,b_idx[2]] - 2 * alpha[b_idx[0], b_idx[1],b_idx[2]] + alpha[b_idx[0], b_idx[1]-1,b_idx[2]])
    #   gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_3d(gradY, b_idx)
    #   div[b_idx[0], b_idx[1],b_idx[2]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * (4*alpha[b_idx[0]-1, b_idx[1],b_idx[2]] - 0.5 * alpha[b_idx[0]-2, b_idx[1],b_idx[2]] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]]) \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,1,0])* gradY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,1,1])* gradY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1]) * gradXY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * gradYY[b_idx[0], b_idx[1],b_idx[2]]
    #   #if ii == 100-7 and jj == 51:
    #   #  print(ii,jj,div[ii,jj],(4*alpha[ii-1,jj] - 0.5 * alpha[ii-2,jj] - 3.5 * alpha[ii,jj]), \
    #     #        gradY[ii,jj], gradxy, grady2)
 
    # elif btype == "bottomright":
    #   gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_3d(gradY, b_idx)
    #   div[b_idx[0], b_idx[1],b_idx[2]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * (4*alpha[b_idx[0]-1, b_idx[1],b_idx[2]] - 0.5 * alpha[b_idx[0]-2, b_idx[1],b_idx[2]] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]]) \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1]) * gradXY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * (4*alpha[b_idx[0], b_idx[1]+1,b_idx[2]] - 0.5 * alpha[b_idx[0], b_idx[1]+2,b_idx[2]] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]])
      
    # elif btype == "topright":
    #   gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.right_diff_idx_3d(gradY, b_idx)
    #   div[b_idx[0], b_idx[1],b_idx[2]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * (4*alpha[b_idx[0]-1, b_idx[1],b_idx[2]] - 0.5 * alpha[b_idx[0]-2, b_idx[1],b_idx[2]] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]]) \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1]) * gradXY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * (4*alpha[b_idx[0], b_idx[1]-1,b_idx[2]] - 0.5 * alpha[b_idx[0], b_idx[1]-2,b_idx[2]] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]])
                     
    # elif btype == "notleft":
    #   gradXX[b_idx[0], b_idx[1],b_idx[2]] = (alpha[b_idx[0]-2, b_idx[1],b_idx[2]] - 2 * alpha[b_idx[0]-1, b_idx[1],b_idx[2]] + alpha[b_idx[0], b_idx[1],b_idx[2]])
    #   gradYY[b_idx[0], b_idx[1],b_idx[2]] = (alpha[b_idx[0]-1, b_idx[1]+1,b_idx[2]] - 2 * alpha[b_idx[0]-1, b_idx[1],b_idx[2]] + alpha[b_idx[0]-1, b_idx[1]-1,b_idx[2]])
    #   div[b_idx[0], b_idx[1],b_idx[2]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * gradXX[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1]) \
    #                             * (0.5*alpha[b_idx[0]-1, b_idx[1]-1,b_idx[2]]-1.5*alpha[b_idx[0], b_idx[1],b_idx[2]]+2*alpha[b_idx[0]-1, b_idx[1],b_idx[2]] \
    #                                - 0.5*alpha[b_idx[0]-1, b_idx[1]+1,b_idx[2]]-0.5*alpha[b_idx[0]-2, b_idx[1],b_idx[2]]) \
    #                             + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * gradYY[b_idx[0], b_idx[1],b_idx[2]]
 
    # elif btype == "bottom":
    #   gradXX[b_idx[0], b_idx[1],b_idx[2]] = (alpha[b_idx[0]+1, b_idx[1],b_idx[2]] - 2 * alpha[b_idx[0], b_idx[1],b_idx[2]] + alpha[b_idx[0]-1, b_idx[1],b_idx[2]])
    #   gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.bottom_diff_idx_3d(gradX, b_idx)
    #   div[b_idx[0], b_idx[1],b_idx[2]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * gradXX[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,0,1])* gradX[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,0,0])* gradX[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1]) * gradXY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * (4*alpha[b_idx[0], b_idx[1]+1,b_idx[2]] - 0.5 * alpha[b_idx[0], b_idx[1]+2,b_idx[2]] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]])
 
    # elif  btype == "nottop":
    #   gradXX[b_idx[0], b_idx[1],b_idx[2]] = (alpha[b_idx[0]+1, b_idx[1]+1,b_idx[2]] - 2 * alpha[b_idx[0], b_idx[1]+1,b_idx[2]] + alpha[b_idx[0]-1, b_idx[1]+1,b_idx[2]])
    #   gradYY[b_idx[0], b_idx[1],b_idx[2]] = (alpha[b_idx[0], b_idx[1]+2,b_idx[2]] - 2 * alpha[b_idx[0], b_idx[1]+1,b_idx[2]] + alpha[b_idx[0], b_idx[1],b_idx[2]])
    #   div[b_idx[0], b_idx[1],b_idx[2]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * gradXX[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1]) \
    #                             * (0.5*alpha[b_idx[0]+1, b_idx[1]+1,b_idx[2]]-1.5*alpha[b_idx[0], b_idx[1],b_idx[2]]+2*alpha[b_idx[0], b_idx[1]+1,b_idx[2]] \
    #                                - 0.5*alpha[b_idx[0]-1, b_idx[1]+1,b_idx[2]]-0.5*alpha[b_idx[0], b_idx[1]+2,b_idx[2]]) \
    #                             + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * gradYY[b_idx[0], b_idx[1],b_idx[2]]
                     
    # elif btype == "top":
    #   gradXX[b_idx[0], b_idx[1],b_idx[2]] = (alpha[b_idx[0]+1, b_idx[1],b_idx[2]] - 2 * alpha[b_idx[0], b_idx[1],b_idx[2]] + alpha[b_idx[0]-1, b_idx[1],b_idx[2]])
    #   gradXY[b_idx[0], b_idx[1],b_idx[2]] = diff.top_diff_idx_3d(gradX, b_idx)
    #   div[b_idx[0], b_idx[1],b_idx[2]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * gradXX[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],1,0,1])* gradX[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],b_idx[2],0,0,0])* gradX[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1]) * gradXY[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * (4*alpha[b_idx[0], b_idx[1]-1,b_idx[2]] - 0.5 * alpha[b_idx[0], b_idx[1]-2,b_idx[2]] - 3.5 * alpha[b_idx[0], b_idx[1],b_idx[2]])
    #       #if (ii == 24 or ii == (100-24)) and jj == 85:
    #       #  print(ii,jj,div[ii,jj], gradx2, \
    #       #        gradX[ii,jj], gradxy, (4*alpha[ii,jj-1] - 0.5 * alpha[ii,jj-2] - 3.5 * alpha[ii,jj]))
 
    # elif btype == "notbottom":
    #   gradXX[b_idx[0], b_idx[1],b_idx[2]] = (alpha[b_idx[0]+1, b_idx[1]-1,b_idx[2]] - 2 * alpha[b_idx[0], b_idx[1]-1,b_idx[2]] + alpha[b_idx[0]-1, b_idx[1]-1,b_idx[2]])
    #   gradYY[b_idx[0], b_idx[1],b_idx[2]] = (alpha[b_idx[0], b_idx[1]-2,b_idx[2]] - 2 * alpha[b_idx[0], b_idx[1]-1,b_idx[2]] + alpha[b_idx[0], b_idx[1],b_idx[2]])
    #   div[b_idx[0], b_idx[1],b_idx[2]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,0] * gradXX[b_idx[0], b_idx[1],b_idx[2]] \
    #                             + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],0,1]) \
    #                             * (0.5*alpha[b_idx[0]-1, b_idx[1]-1,b_idx[2]]-1.5*alpha[b_idx[0], b_idx[1],b_idx[2]]+2*alpha[b_idx[0], b_idx[1]-1,b_idx[2]] \
    #                                - 0.5*alpha[b_idx[0]+1, b_idx[1]-1,b_idx[2]]-0.5*alpha[b_idx[0], b_idx[1]-2,b_idx[2]]) \
    #                                + args['det_g_x_g_inv'][b_idx[0], b_idx[1],b_idx[2],1,1] * gradYY[b_idx[0], b_idx[1],b_idx[2]]

    # elif btype == "outside":
    #   # outside mask, skip
    #   pass

    # else:
    #   # unrecognized type
    #   print(btype, "unrecognized.  Skipping")

     
  lhs = div / args['sqrt_det_g']
  lhs[np.isnan(lhs)] = 0
  
  print('Residual:', np.sqrt(np.sum(lhs - args['rhs']) ** 2))
  
  return(lhs[args['mask']>0])

# end Ax

class gmres_iter_status(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.rks = []
    def __call__(self, rk=None):
        self.niter += 1
        self.rks.append(rk)
        if self._disp:
            print("iter %3i\trk=%s" % (self.niter, str(rk)))

def solve_3d(in_tens, in_mask, max_iters, clipped_range=None, thresh_ratio=None, save_intermediate_results = False, small_eval = 5e-5, sigma=None):
  # This is the main entry point
  # assumes in_tens is the upper-triangular representation
  # clipped_range defaults to [-2,2]
  # thresh_ratio defaults to 1.0
  # any eigenvalues < small_eval are set to small_eval

  if clipped_range is None:
    clipped_range = [-2, 2]
  if thresh_ratio is None:
    thresh_ratio = 1.0

  xsz = in_mask.shape[0]
  ysz = in_mask.shape[1]
  zsz = in_mask.shape[2]
  mask = np.copy(in_mask)
  
  tens = np.zeros((xsz,ysz,zsz,3,3))
  intermed_results = {}
  tot_time = 0

  scale_factor = 1.0
  max_tens = np.max(in_tens)
  while scale_factor * max_tens < 1:
    scale_factor = scale_factor * 10
  print("Scaling tensors by factor of", scale_factor)
  in_tens *= scale_factor
  
  iso_tens = np.zeros((3,3))
  # TODO find a better scale factor here for these tensors outside the mask
  # want them invertible and not interfering w/ display
  #iso_tens[0,0] = 1.0e-4 
  #iso_tens[1,1] = 1.0e-4
  #iso_tens[2,2] = 1.0e-4
  iso_tens[0,0] = 1.0 / scale_factor
  iso_tens[1,1] = 1.0 / scale_factor
  iso_tens[2,2] = 1.0 / scale_factor 

  start_time = time.time()
  #####################
  # preprocess inputs #
  #####################
  print("Preprocessing tensors and mask...")

  if sigma is None:
    filt_tens = in_tens
  else:
    filt_tens = GetNPArrayFromSITK(
                sitk.RecursiveGaussian(sitk.RecursiveGaussian(sitk.RecursiveGaussian(
                  GetSITKImageFromNP(in_tens,True),sigma=sigma,direction=0),sigma=sigma,direction=1), sigma=sigma, direction=2),True)
    
  # convert to full 3x3 representation of tensors
  tens[:,:,:,0,0] = filt_tens[:,:,:,0]
  tens[:,:,:,0,1] = filt_tens[:,:,:,1]
  tens[:,:,:,1,0] = filt_tens[:,:,:,1]
  tens[:,:,:,0,2] = filt_tens[:,:,:,2]
  tens[:,:,:,2,0] = filt_tens[:,:,:,2]
  tens[:,:,:,1,1] = filt_tens[:,:,:,3]
  tens[:,:,:,1,2] = filt_tens[:,:,:,4]
  tens[:,:,:,2,1] = filt_tens[:,:,:,4]
  tens[:,:,:,2,2] = filt_tens[:,:,:,5]

  # precondition and setup boundary info
  precondition_tensors(tens, mask)
  if save_intermediate_results:
    intermed_results['in_tens'] = in_tens
    intermed_results['tens'] = tens
    intermed_results['orig_mask'] = mask

  # remove small components via morphological operations
  mo.open_mask_3d(mask)
    
  bdry_type, bdry_idx, bdry_map = mo.determine_boundary_3d(mask, True)
  if save_intermediate_results:
    intermed_results['bdry_type1'] = bdry_type
    intermed_results['bdry_idx1'] = bdry_idx
    intermed_results['bdry_map1'] = bdry_map
    intermed_results['diff_mask1'] = mask

  print("Calling determine_boundary_3d a second time...")
  bdry_type, bdry_idx, bdry_map = mo.determine_boundary_3d(mask, False)

  if save_intermediate_results:
    intermed_results['bdry_type'] = bdry_type
    intermed_results['bdry_idx'] = bdry_idx
    intermed_results['bdry_map'] = bdry_map
    intermed_results['differentiable_mask'] = mask

  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  #######################################
  # fix negative eigenvalues of tensors # 
  #######################################
  print("Fixing negative eigenvalues of any tensors in tensorfield...")

  # TODO make this a preprocessing step
  tens[mask == 0] = iso_tens
  recon_tens = tensors.make_pos_def(tens, mask, small_eval=small_eval) #5e-2
  #recon_tens = tens

  if save_intermediate_results:
    intermed_results['recon_tens'] = recon_tens
  else:
    del tens
    gc.collect()
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  ##################################
  # setup metric (g) and g inverse #
  ##################################
  print("Setting up metric...")

  g_inv = np.copy(recon_tens)
  g_inv[mask == 0] = iso_tens
  
  g = np.linalg.inv(g_inv)
  g[np.isnan(g)] = 0
  g_inv[np.isnan(g_inv)] = 0

  sqrt_det_g = np.sqrt(np.linalg.det(g)) * mask
  sqrt_det_g[np.isnan(sqrt_det_g)] = 1
  sqrt_det_g[sqrt_det_g == 0] = 1
  # TODO not necessary to threshold sqrt_det_g as small_eval avoids the
  # situation where sqrt_det_g blows up
  print('thresholded',np.sum(sqrt_det_g > 1e4), 'pixels where sqrt_det_g > 1e4') 
  sqrt_det_g[sqrt_det_g > 1e4] = 1e4 # originally 10

  if save_intermediate_results:
    intermed_results['g'] = g
    intermed_results['g_inv'] = np.copy(g_inv)
    intermed_results['sqrt_det_g'] = sqrt_det_g

  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  #################################
  # compute derivatives of metric # 
  #################################
  print("Computing derivatives of metric...")
  
  gradG, grad_g_inv, grad_sqrt_det_g = compute_g_derivs_new(g, g_inv, sqrt_det_g, bdry_idx, bdry_map)

  grad_det_g_x_g_inv = np.zeros((xsz,ysz,zsz,3,3,3))
  grad_g_inv_x_det_g = np.zeros((xsz,ysz,zsz,3,3,3))
  det_g_x_g_inv = np.zeros((xsz,ysz,zsz,3,3))
  # grad_det_g_x_g_inv[i,j, gradient-direction, g-component, g-component] = gradient of sqrt_det_g .* g_inv
  # grad_g_inv_x_det_g[i,j, g-component, g-component, gradient-direction] = gradient of g_inv .* sqrt_det_g
  # det_g_x_g_inv[i,j, g-component, g-component] = sqrt_det_g .* g_inv

  for ii in range(3):
    for jj in range(3):
      det_g_x_g_inv[:,:,:,ii,jj] = sqrt_det_g[:,:,:] * g_inv[:,:,:,ii,jj]
      for kk in range(3):
        grad_det_g_x_g_inv[:,:,:,ii,jj,kk] = grad_sqrt_det_g[:,:,:,ii] * g_inv[:,:,:,jj,kk]
        grad_g_inv_x_det_g[:,:,:,ii,jj,kk] = grad_g_inv[:,:,:,ii,jj,kk] * sqrt_det_g[:,:,:]

  if save_intermediate_results:
    intermed_results['grad_g'] = gradG
    intermed_results['grad_g_inv'] = grad_g_inv
    intermed_results['grad_sqrt_det_g'] = grad_sqrt_det_g
    intermed_results['det_g_x_g_inv'] = det_g_x_g_inv
    intermed_results['grad_det_g_x_g_inv'] = grad_det_g_x_g_inv
    intermed_results['grad_g_inv_x_det_g'] = grad_g_inv_x_det_g
  else:
    del grad_g_inv, grad_sqrt_det_g
    gc.collect()
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  ########################################################
  # compute principal eigenvector (TDir) of tensor field # 
  ########################################################
  print("Computing principal eigenvector of tensor field...")

  # TODO might be better to have as separate preconditioning step
  # Doing here to only compute eigenvalues once
  eigenvecs = tensors.eigv_3d(recon_tens)
  #eigenvals, eigenvecs = np.linalg.eigh(tens)
  #v = eigenvecs[:,:,:,:,2]
  TDir = riem_vec_norm(eigenvecs[:,:,:,:,2], g)

  TDir[np.isnan(TDir)] = 0
  TDir[np.isinf(TDir)] = 0

  TDirx, TDiry, TDirz = tensors.eigv_sign_deambig(TDir)

  if save_intermediate_results:
    intermed_results['TDir'] = TDir
    intermed_results['TDirx'] = TDirx
    intermed_results['TDiry'] = TDiry
    intermed_results['TDirz'] = TDirz
  else:
    del eigenvecs, TDir
    gc.collect()
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  ###################################################
  # compute gradients of principal eigenvector TDir # 
  ###################################################
  print("Computing gradients of principal eigenvector...")
  gradTx_delx, gradTx_dely, gradTx_delz, gradTy_delx, gradTy_dely, gradTy_delz, gradTz_delx, gradTz_dely, gradTz_delz = compute_T_derivs_new(TDirx, TDiry, TDirz, bdry_idx, bdry_map)

  if save_intermediate_results:
    intermed_results['gradTx_delx'] = gradTx_delx
    intermed_results['gradTx_dely'] = gradTx_dely
    intermed_results['gradTx_delz'] = gradTx_delz
    intermed_results['gradTy_delx'] = gradTy_delx
    intermed_results['gradTy_dely'] = gradTy_dely
    intermed_results['gradTy_delz'] = gradTy_delz
    intermed_results['gradTz_delx'] = gradTz_delx
    intermed_results['gradTz_dely'] = gradTz_dely
    intermed_results['gradTz_delz'] = gradTz_delz
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  ########################################
  # compute first component of nabla_T T # 
  ########################################
  print("Computing first component of nabla_T T...")
  
  first_nabla_TT = np.zeros((xsz,ysz,zsz,3))
  first_nabla_TT[:,:,:,0] = np.multiply(TDirx[:,:,:,0], gradTx_delx[:,:,:]) + np.multiply(TDiry[:,:,:,1], gradTx_dely[:,:,:]) + np.multiply(TDirz[:,:,:,2], gradTx_delz[:,:,:])
  first_nabla_TT[:,:,:,1] = np.multiply(TDirx[:,:,:,0], gradTy_delx[:,:,:]) + np.multiply(TDiry[:,:,:,1], gradTy_dely[:,:,:]) + np.multiply(TDirz[:,:,:,2], gradTy_delz[:,:,:])
  first_nabla_TT[:,:,:,2] = np.multiply(TDirx[:,:,:,0], gradTz_delx[:,:,:]) + np.multiply(TDiry[:,:,:,1], gradTz_dely[:,:,:]) + np.multiply(TDirz[:,:,:,2], gradTz_delz[:,:,:])

  if save_intermediate_results:
    intermed_results['first_nabla_TT'] = first_nabla_TT
  else:
    del TDiry, TDirz, gradTx_delx, gradTy_delx, gradTz_delx
    del gradTx_dely, gradTy_dely, gradTz_dely
    del gradTx_delz, gradTy_delz, gradTz_delz
    gc.collect()
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  #########################################
  # compute second component of nabla_T T # 
  #########################################
  print("Computing second component of nabla_T T...")

  christoffel=np.zeros((xsz,ysz,zsz,3,3,3))
  second_nabla_TT = np.zeros((xsz,ysz,zsz,3))
  for k in range(3):
    for p in range(3):
      for q in range(3):
        christoffel[:,:,:,k,p,q] = 0.5 * g_inv[:,:,:,k,0]*(gradG[:,:,:,q,0,p] + gradG[:,:,:,p,0,q]-gradG[:,:,:,p,q,0])
        christoffel[:,:,:,k,p,q] += 0.5 * g_inv[:,:,:,k,1]*(gradG[:,:,:,q,1,p] + gradG[:,:,:,p,1,q]-gradG[:,:,:,p,q,1])
        christoffel[:,:,:,k,p,q] += 0.5 * g_inv[:,:,:,k,2]*(gradG[:,:,:,q,2,p] + gradG[:,:,:,p,2,q]-gradG[:,:,:,p,q,2])
    #christoffel[k,0,0] = 0.5 * g_inv[ii,jj,k,0]*(gradG[ii,jj,0,0,0])
    #christoffel[k,0,0] += 0.5 * g_inv[ii,jj,k,1]*(2*gradG[ii,jj,0,1,0] - gradG[ii,jj,0,0,1])
    #christoffel[k,0,1] = 0.5 * g_inv[ii,jj,k,0]*(gradG[ii,jj,0,0,1])
    #christoffel[k,0,1] += 0.5 * g_inv[ii,jj,k,1]*(gradG[ii,jj,1,1,0])
    #christoffel[k,1,0] = 0.5 * g_inv[ii,jj,k,0]*(gradG[ii,jj,0,0,1])
    #christoffel[k,1,0] += 0.5 * g_inv[ii,jj,k,1]*(gradG[ii,jj,1,1,0])
    #christoffel[k,1,1] = 0.5 * g_inv[ii,jj,k,0]*(2*gradG[ii,jj,1,0,1] - gradG[ii,jj,1,1,0])
    #christoffel[k,1,1] += 0.5 * g_inv[ii,jj,k,1]*(gradG[ii,jj,1,1,1])
    second_nabla_TT[:,:,:,k] = christoffel[:,:,:,k,0,0] * TDirx[:,:,:,0] * TDirx[:,:,:,0] # sign ambiguity won't matter here, so use any TDir
    second_nabla_TT[:,:,:,k] += christoffel[:,:,:,k,1,0] * TDirx[:,:,:,0] * TDirx[:,:,:,1]
    second_nabla_TT[:,:,:,k] += christoffel[:,:,:,k,0,1] * TDirx[:,:,:,1] * TDirx[:,:,:,0]
    second_nabla_TT[:,:,:,k] += christoffel[:,:,:,k,2,0] * TDirx[:,:,:,0] * TDirx[:,:,:,2]
    second_nabla_TT[:,:,:,k] += christoffel[:,:,:,k,0,2] * TDirx[:,:,:,2] * TDirx[:,:,:,0]
    second_nabla_TT[:,:,:,k] += christoffel[:,:,:,k,1,1] * TDirx[:,:,:,1] * TDirx[:,:,:,1]
    second_nabla_TT[:,:,:,k] += christoffel[:,:,:,k,2,1] * TDirx[:,:,:,1] * TDirx[:,:,:,2]
    second_nabla_TT[:,:,:,k] += christoffel[:,:,:,k,1,2] * TDirx[:,:,:,2] * TDirx[:,:,:,1]
    second_nabla_TT[:,:,:,k] += christoffel[:,:,:,k,2,2] * TDirx[:,:,:,2] * TDirx[:,:,:,2]

  if save_intermediate_results:
    intermed_results['second_nabla_TT'] = second_nabla_TT
    intermed_results['christoffel'] = christoffel
  else:
    del gradG, TDirx, christoffel
    gc.collect()
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  #####################
  # compute nabla_T T # 
  #####################
  print("Computing nabla_T T...")
  
  nabla_TT = first_nabla_TT + second_nabla_TT
  sqrt_det_nabla_TT = np.zeros((xsz,ysz,zsz,3))
  sqrt_det_nabla_TT[:,:,:,0] = sqrt_det_g * nabla_TT[:,:,:,0]
  sqrt_det_nabla_TT[:,:,:,1] = sqrt_det_g * nabla_TT[:,:,:,1]
  sqrt_det_nabla_TT[:,:,:,2] = sqrt_det_g * nabla_TT[:,:,:,2]

  if save_intermediate_results:
    intermed_results['nabla_TT'] = nabla_TT
    intermed_results['sqrt_det_nabla_TT'] = sqrt_det_nabla_TT
  else:
    del first_nabla_TT, second_nabla_TT
    gc.collect()
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  #####################################################
  # compute rhs = 2*divergence (nabla_T T)/sqrt_det_g # 
  #####################################################
  print("Computing rhs = 2*divergence (nabla_T T)/sqrt_det_g...")

  grad_nabla_TT, grad_sqrt_det_nabla_TT = compute_nabla_derivs_new(nabla_TT, sqrt_det_nabla_TT, bdry_idx, bdry_map)
  
  # TODO save time by not computing grad_nabla_TT[:,:,0,1], grad_nabla_TT[:,:,1,0], grad_sqrt_det_nabla_TT[:,:,0,1] or grad_sqrt_det_nabla_TT[:,:,1,0]
  # Note, following happened in Jupyter Notebook, but doesn't seem exactly right:
  #divergence_rhs = grad_sqrt_det_nabla_TT[:,:,0,0] + grad_sqrt_det_nabla_TT[:,:,1,1]
  # Do following instead:
  # TODO if this works, don't need to compute grad_sqrt_det_nabla_TT
  #divergence_rhs = (grad_sqrt_det_g[:,:,0] * nabla_TT[:,:,0] + \
  #                  sqrt_det_g[:,:] * grad_nabla_TT[:,:,0,0] + \
  #                  grad_sqrt_det_g[:,:,1] * nabla_TT[:,:,1] + \
  #                  sqrt_det_g[:,:] * grad_nabla_TT[:,:,1,1])
  #divergence_rhs = (sqrt_det_g[:,:] * grad_nabla_TT[:,:,0,0] + \
  #                  sqrt_det_g[:,:] * grad_nabla_TT[:,:,1,1])

  # Reason why we don't need cross terms here, can get them from numeric diff is that on rhs we keep the full divergence
  # When we calculate divergence on the lhs, on the other hand, we need to split terms across lhs and rhs for the Neumann conditions
  divergence_rhs = (grad_sqrt_det_nabla_TT[:,:,:,0,0] + \
                    grad_sqrt_det_nabla_TT[:,:,:,1,1] + \
                    grad_sqrt_det_nabla_TT[:,:,:,2,2])

  
  rhs = 2.0 * divergence_rhs / sqrt_det_g

  if save_intermediate_results:
    intermed_results['grad_nabla_TT'] = grad_nabla_TT
    intermed_results['grad_sqrt_det_nabla_TT'] = grad_sqrt_det_nabla_TT
    intermed_results['divergence_rhs'] = divergence_rhs
    intermed_results['rhs_b4_neumann'] = np.copy(rhs)
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  #######################################
  # compute Neumann Boundary Conditions # 
  #######################################
  print("Adjusting RHS for Neumann Boundary Conditions...")

  # TODO following is no good, because it includes as boundary points, certain pixels that are not on the boundary.  (Or at least not consistently on boundary on lhs too)
  # At the very least, use Ax_orig w/ neumann_conditions_rhs_orig
  # See whether it works to pair Ax w/ neumann_conditions_rhs
  neumann_terms = neumann_conditions_rhs(nabla_TT, g, grad_det_g_x_g_inv, grad_g_inv_x_det_g, det_g_x_g_inv, sqrt_det_g, bdry_idx, bdry_map)
  rhs -= neumann_terms

  if save_intermediate_results:
    intermed_results['neumann_terms'] = neumann_terms
    intermed_results['rhs'] = rhs
  else:
    del nabla_TT, neumann_terms, g
    gc.collect()
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  #############
  # setup lhs # 
  #############
  print("Setting up lhs...")

  args = {}
  args['det_g_x_g_inv'] = det_g_x_g_inv
  args['grad_det_g_x_g_inv'] = grad_det_g_x_g_inv
  args['grad_g_inv_x_det_g'] = grad_g_inv_x_det_g
  args['sqrt_det_g'] = sqrt_det_g
  args['mask'] = mask
  args['xsz'] = xsz
  args['ysz'] = ysz
  args['zsz'] = zsz
  args['rhs'] = rhs
  args['bdry_idx'] = bdry_idx
  args['bdry_map'] = bdry_map

  #Ax_args = lambda x: Ax_orig(x, args)
  Ax_args = lambda x: Ax(x, args)
  mask_shape = mask[mask > 0].shape
  AxLO = LinearOperator((mask_shape[0], mask_shape[0]), Ax_args)

  b = rhs[mask > 0]
  iters = gmres_iter_status()

  if save_intermediate_results:
    intermed_results['AxLO'] = AxLO
  
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  #############
  # run gmres # 
  #############
  print("Running gmres...")

  x, exitCode = gmres(AxLO, b, maxiter = max_iters, callback=iters)#, tol=1e-10, atol = 1e-10)

  alpha = np.zeros((xsz,ysz,zsz))
  alpha[mask > 0] = x
  
  if save_intermediate_results:
    intermed_results['rks'] = iters.rks
    intermed_results['alpha_preclipping'] = alpha
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
####################################
  # clip alpha and threshold tensors # 
  ####################################
  print("Clipping and thresholding...")

  clipped_alpha = np.copy(alpha)
  clipped_alpha[alpha < clipped_range[0]] = clipped_range[0]
  clipped_alpha[alpha > clipped_range[1]] = clipped_range[1]

  # final Ax
  Ax_final_noclip = np.zeros((xsz,ysz,zsz))
  Ax_final_noclip[mask > 0] = AxLO(x)
  Ax_final_clip = np.zeros((xsz,ysz,zsz))
  Ax_final_clip[mask > 0] = AxLO(clipped_alpha[mask > 0])

  res_img_noclip = np.abs(Ax_final_noclip-rhs)
  res_img_clip = np.abs(Ax_final_clip-rhs)
  res_noclip = np.sqrt(np.sum(res_img_noclip**2))
  res_clip = np.sqrt(np.sum(res_img_clip**2))

  scaled_tensors_noclip = tensors.scale_by_alpha(recon_tens,alpha)
  scaled_ginv = tensors.scale_by_alpha(g_inv, clipped_alpha)
  scaled_tensors = tensors.scale_by_alpha(recon_tens, clipped_alpha)

  # Not necessary for metric estimation. Need to decide separately
  # if tensors should be thresholded for metric matching
  #threshold_ginv = tensors.threshold_to_input(scaled_ginv, tens, thresh_ratio)
  #threshold_tensors = tensors.threshold_to_input(scaled_tensors, tens, thresh_ratio)
  
  if save_intermediate_results:
    intermed_results['alpha'] = clipped_alpha
    intermed_results['Ax_final_noclip'] = Ax_final_noclip
    intermed_results['Ax_final_clip'] = Ax_final_clip
    intermed_results['res_img_noclip'] = res_img_noclip
    intermed_results['res_img_clip'] = res_img_clip
    intermed_results['res_noclip'] = res_noclip
    intermed_results['res_clip'] = res_clip
    intermed_results['scaled_tensors_noclip'] = scaled_tensors_noclip
    intermed_results['scaled_ginv'] = scaled_ginv
    intermed_results['scaled_tensors'] = scaled_tensors
    #intermed_results['threshold_ginv'] = threshold_ginv
    #intermed_results['threshold_tensors'] = threshold_tensors    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")

  print("Total solve time:", tot_time, "seconds")
  #return(clipped_alpha, threshold_tensors, mask, iters.rks, intermed_results)
  return(clipped_alpha, scaled_tensors, mask, iters.rks, intermed_results)
# end solve_3d

def compute_analytic_solution(mask, center_line=35):
  # computes -2 ln r + 2 ln center_line
  xcent = mask.shape[0] / 2
  ycent = mask.shape[1] / 2
  zcent = mask.shape[2] / 2
  ln_img = np.zeros_like(mask)
  for x in range(mask.shape[0]):
    for y in range(mask.shape[1]):
      for z in range(mask.shape[2]):
        if x==xcent and y == ycent:
          pass
        else:
          if mask[x,y,z]:
            ln_img[x,y,z] = -2.0*math.log(math.sqrt((x-xcent)**2 + (y-ycent)**2)) + 2.0*math.log(math.sqrt(center_line**2)) # for center line

  return(ln_img)

def solve_analytic_annulus_3d(in_tens, mask, center_line=35, save_intermediate_results = False):
  # TODO Either delete this function or finish fixing it
  xsz = mask.shape[0]
  ysz = mask.shape[1]
  zsz = mask.shape[2]
  
  ln_img = compute_analytic_solution(mask, center_line)
  intermed_results = {}
  if save_intermediate_results:
    intermed_results['ln_img'] = ln_img

  # convert to full 3x3 representation of tensors
  tens[:,:,:,0,0] = in_tens[:,:,:,0]
  tens[:,:,:,0,1] = in_tens[:,:,:,1]
  tens[:,:,:,1,0] = in_tens[:,:,:,1]
  tens[:,:,:,0,2] = in_tens[:,:,:,2]
  tens[:,:,:,2,0] = in_tens[:,:,:,2]
  tens[:,:,:,1,1] = in_tens[:,:,:,3]
  tens[:,:,:,1,2] = in_tens[:,:,:,4]
  tens[:,:,:,2,1] = in_tens[:,:,:,4]
  tens[:,:,:,2,2] = in_tens[:,:,:,5]

  iso_tens = np.zeros((3,3))
  iso_tens[0,0] = 1.0 
  iso_tens[1,1] = 1.0
  iso_tens[2,2] = 1.0 

  # precondition and setup boundary info
  precondition_tensors(tens, mask)
  
  if save_intermediate_results:
    intermed_results['tens'] = tens
    intermed_results['orig_mask'] = mask
  
  bdry_type, bdry_idx, bdry_map = determine_boundary_3d(mask)

  if save_intermediate_results:
    intermed_results['bdry_type'] = bdry_type
    intermed_results['bdry_idx'] = bdry_idx
    intermed_results['differentiable_mask'] = mask

  g_inv = np.copy(tens)
  g_inv_cond = np.linalg.cond(g_inv)
  g_inv[g_inv_cond > condition_num_thresh] = iso_tens

  g = np.linalg.inv(g_inv)
  g[np.isnan(g)] = 0
  g_inv[np.isnan(g_inv)] = 0

  sqrt_det_g = np.sqrt(np.linalg.det(g))
  sqrt_det_g[np.isnan(sqrt_det_g)] = 1
  sqrt_det_g[sqrt_det_g==0] = 1

  if save_intermediate_results:
    intermed_results['g'] = g
    intermed_results['g_inv'] = g_inv
    intermed_results['g_inv_cond'] = g_inv_cond
    intermed_results['sqrt_det_g'] = sqrt_det_g

  #gradG, grad_g_inv, grad_sqrt_det_g = compute_g_derivs_orig(g, g_inv, sqrt_det_g, mask) # This was the version used for 2D
  gradG, grad_g_inv, grad_sqrt_det_g = compute_g_derivs_new(g, g_inv, sqrt_det_g, bdry_idx, bdry_map)
  #gradG, grad_g_inv, grad_sqrt_det_g = compute_g_derivs(g, g_inv, sqrt_det_g, bdry_idx, bdry_map)

  grad_det_g_x_g_inv = np.zeros((xsz,ysz,zsz,3,3,3))
  grad_g_inv_x_det_g = np.zeros((xsz,ysz,zsz,3,3,3))
  det_g_x_g_inv = np.zeros((xsz,ysz,zsz,3,3))
  # grad_det_g_x_g_inv[i,j, gradient-direction, g-component, g-component] = gradient of sqrt_det_g .* g_inv
  # grad_g_inv_x_det_g[i,j, g-component, g-component, gradient-direction] = gradient of g_inv .* sqrt_det_g
  # det_g_x_g_inv[i,j, g-component, g-component] = sqrt_det_g .* g_inv

  for ii in range(3):
    for jj in range(3):
      det_g_x_g_inv[:,:,:,ii,jj] = sqrt_det_g[:,:,:] * g_inv[:,:,:,ii,jj]
      for kk in range(3):
        grad_det_g_x_g_inv[:,:,:,ii,jj,kk] = grad_sqrt_det_g[:,:,:,ii] * g_inv[:,:,:,jj,kk]
        grad_g_inv_x_det_g[:,:,:,ii,jj,kk] = grad_g_inv[:,:,:,ii,jj,kk] * sqrt_det_g[:,:,:]

  if save_intermediate_results:
    intermed_results['grad_g'] = gradG
    intermed_results['grad_g_inv'] = grad_g_inv
    intermed_results['grad_sqrt_det_g'] = grad_sqrt_det_g
    intermed_results['det_g_x_g_inv'] = det_g_x_g_inv
    intermed_results['grad_det_g_x_g_inv'] = grad_det_g_x_g_inv
    intermed_results['grad_g_inv_x_det_g'] = grad_g_inv_x_det_g
  else:
    del gradG, grad_g_inv, grad_sqrt_det_g
    gc.collect()

  #alpha_gradX, alpha_gradY, alpha_gradZ = compute_alpha_derivs_orig(ln_img, mask) # This was the version used for 2D
  alpha_gradX, alpha_gradY, alpha_gradZ = compute_alpha_derivs(ln_img, bdry_idx, bdry_map)
  #alpha_gradX, alpha_gradY = compute_alpha_derivs(ln_img, bdry_idx, bdry_map)

  if save_intermediate_results:
    intermed_results['alpha_gradX'] = alpha_gradX
    intermed_results['alpha_gradY'] = alpha_gradY
    intermed_results['alpha_gradZ'] = alpha_gradZ

  analytic_grad_alpha = np.zeros((xsz, ysz, zsz))
  analytic_grad_alpha[:,:,:,0] = g_inv[:,:,:,0,0] * alpha_gradX + g_inv[:,:,:,0,1] * alpha_gradY + g_inv[:,:,:,0,2] * alpha_gradZ
  analytic_grad_alpha[:,:,:,1] = g_inv[:,:,:,1,0] * alpha_gradX + g_inv[:,:,:,1,1] * alpha_gradY + g_inv[:,:,:,1,2] * alpha_gradZ
  analytic_grad_alpha[:,:,:,2] = g_inv[:,:,:,2,0] * alpha_gradX + g_inv[:,:,:,2,1] * alpha_gradY + g_inv[:,:,:,2,2] * alpha_gradZ

  analytic_div = compute_div_grad_alpha_orig(analytic_grad_alpha, mask) # This was the version used for 2D
  analytic_div = compute_div_grad_alpha(analytic_grad_alpha, bdry_idx, bdry_map)
  #analytic_div = compute_div_grad_alpha(analytic_grad_alpha, bdry_idx, bdry_map)

  if save_intermediate_results:
    intermed_results['analytic_grad_alpha'] = analytic_grad_alpha
    intermed_results['analytic_div'] = analytic_div

  return(analytic_div, intermed_results)
# end solve_analytic_annulus_3d
