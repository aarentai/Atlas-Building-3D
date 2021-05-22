from lazy_imports import np
from lazy_imports import LinearOperator, gmres
import math
import time

from util import diff
from util import tensors
from util import maskops as mo
from util.riemann import riem_vec_norm

# starting function will make a copy of the input data in order to keep the originals clean.

def precondition_tensors(tens, mask):
  # get rid of nans, poor conditioned tensors
  # Only keep elements of mask that are large enough to take accurate derivatives
  tens[np.isnan(tens)] = 0
    
def compute_g_derivs_orig(g, g_inv, sqrt_det_g, mask):
  xsz = g.shape[0]
  ysz = g.shape[1]
  gradG = np.zeros((xsz,ysz, 2, 2, 2))
  gradG_x_left = np.zeros((xsz,ysz,2,2))
  gradG_x_right = np.zeros((xsz,ysz,2,2))
  gradG_y_bottom = np.zeros((xsz,ysz,2,2))
  gradG_y_top = np.zeros((xsz,ysz,2,2))

  grad_g_inv = np.zeros((xsz,ysz, 2, 2, 2))
  grad_g_inv_left = np.zeros((xsz,ysz,2,2))
  grad_g_inv_right = np.zeros((xsz,ysz,2,2))
  grad_g_inv_bottom = np.zeros((xsz,ysz,2,2))
  grad_g_inv_top = np.zeros((xsz,ysz,2,2))

  grad_sqrt_det_g = np.zeros((xsz,ysz,2))
  grad_sqrt_det_g_x_left = np.zeros((xsz,ysz))
  grad_sqrt_det_g_x_right = np.zeros((xsz,ysz))
  grad_sqrt_det_g_y_bottom = np.zeros((xsz,ysz))
  grad_sqrt_det_g_y_top = np.zeros((xsz,ysz))
  
  for xx in range(2):
    for yy in range(2):
      gradG[:,:,xx,yy,0], gradG[:,:,xx,yy,1] = np.gradient(g[:,:,xx,yy])
      grad_g_inv[:,:,xx,yy,0], grad_g_inv[:,:,xx,yy,1] = np.gradient(g_inv[:,:,xx,yy])
      gradG_x_left[:,:,xx,yy] = diff.left_diff_2d(g[:,:,xx,yy])
      gradG_x_right[:,:,xx,yy] = diff.right_diff_2d(g[:,:,xx,yy])
      gradG_y_bottom[:,:,xx,yy] = diff.bottom_diff_2d(g[:,:,xx,yy])
      gradG_y_top[:,:,xx,yy] = diff.top_diff_2d(g[:,:,xx,yy])

  for xx in range(2):
    for yy in range(2):
      grad_g_inv_left[:,:,xx,yy] = diff.left_diff_2d(g_inv[:,:,xx,yy])
      grad_g_inv_right[:,:,xx,yy] = diff.right_diff_2d(g_inv[:,:,xx,yy])
      grad_g_inv_bottom[:,:,xx,yy] = diff.bottom_diff_2d(g_inv[:,:,xx,yy])
      grad_g_inv_top[:,:,xx,yy] = diff.top_diff_2d(g_inv[:,:,xx,yy])

  grad_sqrt_det_g[:,:,0], grad_sqrt_det_g[:,:,1] = np.gradient(sqrt_det_g)
  grad_sqrt_det_g_x_left[:,:] = diff.left_diff_2d(sqrt_det_g)
  grad_sqrt_det_g_x_right[:,:] = diff.right_diff_2d(sqrt_det_g)
  grad_sqrt_det_g_y_bottom[:,:] = diff.bottom_diff_2d(sqrt_det_g)
  grad_sqrt_det_g_y_top[:,:] = diff.top_diff_2d(sqrt_det_g)

  # correct center derivatives (np.gradient) on all 4 boundaries

  for ii in range(xsz):
    for jj in range(ysz):
      if mask[ii,jj]:
        if (not mask[ii-1,jj]) and mask[ii+1,jj] and mask[ii,jj-1] and mask[ii,jj+1]: # left boundary
          gradG[ii,jj,:,:,0] = gradG_x_left[ii,jj]
          grad_g_inv[ii,jj,:,:,0] = grad_g_inv_left[ii,jj]
          grad_sqrt_det_g[ii,jj,0] = grad_sqrt_det_g_x_left[ii,jj]
          
        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottomleft boundary
          gradG[ii,jj,:,:,0] = gradG_x_left[ii,jj]
          gradG[ii,jj,:,:,1] = gradG_y_bottom[ii,jj]
          grad_g_inv[ii,jj,:,:,0] = grad_g_inv_left[ii,jj]
          grad_sqrt_det_g[ii,jj,0] = grad_sqrt_det_g_x_left[ii,jj]
          grad_g_inv[ii,jj,:,:,1] = grad_g_inv_bottom[ii,jj]
          grad_sqrt_det_g[ii,jj,1] = grad_sqrt_det_g_y_bottom[ii,jj]
          
        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and mask[ii,jj-1] and (not mask[ii,jj+1]): # topleft boundary
          gradG[ii,jj,:,:,0] = gradG_x_left[ii,jj]
          gradG[ii,jj,:,:,1] = gradG_y_top[ii,jj]
          grad_g_inv[ii,jj,:,:,0] = grad_g_inv_left[ii,jj]
          grad_sqrt_det_g[ii,jj,0] = grad_sqrt_det_g_x_left[ii,jj]
          grad_g_inv[ii,jj,:,:,1] = grad_g_inv_top[ii,jj]
          grad_sqrt_det_g[ii,jj,1] = grad_sqrt_det_g_y_top[ii,jj]
   
        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and (not mask[ii,jj-1]) and (not mask[ii,jj+1]): # notright boundary
          gradG[ii,jj,:,:,0] = gradG_x_left[ii,jj]
          gradG[ii,jj,:,:,1] = 0
          grad_g_inv[ii,jj,:,:,0] = grad_g_inv_left[ii,jj]
          grad_sqrt_det_g[ii,jj,0] = grad_sqrt_det_g_x_left[ii,jj]
          grad_g_inv[ii,jj,:,:,1] = 0
          grad_sqrt_det_g[ii,jj,1] = 0
  
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and mask[ii,jj-1] and mask[ii,jj+1]: # right boundary
          gradG[ii,jj,:,:,0] = gradG_x_right[ii,jj]
          grad_g_inv[ii,jj,:,:,0] = grad_g_inv_right[ii,jj]
          grad_sqrt_det_g[ii,jj,0] = grad_sqrt_det_g_x_right[ii,jj]
   
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottomright boundary
          gradG[ii,jj,:,:,0] = gradG_x_right[ii,jj]
          gradG[ii,jj,:,:,1] = gradG_y_bottom[ii,jj]
          grad_g_inv[ii,jj,:,:,0] = grad_g_inv_right[ii,jj]
          grad_sqrt_det_g[ii,jj,0] = grad_sqrt_det_g_x_right[ii,jj]
          grad_g_inv[ii,jj,:,:,1] = grad_g_inv_bottom[ii,jj]
          grad_sqrt_det_g[ii,jj,1] = grad_sqrt_det_g_y_bottom[ii,jj]
                      
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and mask[ii,jj-1] and (not mask[ii,jj+1]): # topright boundary
          gradG[ii,jj,:,:,0] = gradG_x_right[ii,jj]
          gradG[ii,jj,:,:,1] = gradG_y_top[ii,jj]
          grad_g_inv[ii,jj,:,:,0] = grad_g_inv_right[ii,jj]
          grad_sqrt_det_g[ii,jj,0] = grad_sqrt_det_g_x_right[ii,jj]
          grad_g_inv[ii,jj,:,:,1] = grad_g_inv_top[ii,jj]
          grad_sqrt_det_g[ii,jj,1] = grad_sqrt_det_g_y_top[ii,jj]
              
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and (not mask[ii,jj+1]): # notleft boundary
          gradG[ii,jj,:,:,0] = gradG_x_right[ii,jj]
          gradG[ii,jj,:,:,1] = 0
          grad_g_inv[ii,jj,:,:,0] = grad_g_inv_right[ii,jj]
          grad_sqrt_det_g[ii,jj,0] = grad_sqrt_det_g_x_right[ii,jj]
          grad_g_inv[ii,jj,:,:,1] = 0
          grad_sqrt_det_g[ii,jj,1] = 0
              
        elif mask[ii-1,jj] and mask[ii+1,jj] and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottom boundary
          gradG[ii,jj,:,:,1] = gradG_y_bottom[ii,jj]
          grad_g_inv[ii,jj,:,:,1] = grad_g_inv_bottom[ii,jj]
          grad_sqrt_det_g[ii,jj,1] = grad_sqrt_det_g_y_bottom[ii,jj]
  
        elif (not mask[ii-1,jj]) and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and mask[ii,jj+1]: # nottop boundary
          gradG[ii,jj,:,:,0] = 0
          gradG[ii,jj,:,:,1] = gradG_y_bottom[ii,jj]
          grad_g_inv[ii,jj,:,:,0] = 0
          grad_sqrt_det_g[ii,jj,0] = 0
          grad_g_inv[ii,jj,:,:,1] = grad_g_inv_bottom[ii,jj]
          grad_sqrt_det_g[ii,jj,1] = grad_sqrt_det_g_y_bottom[ii,jj]
          
        elif mask[ii-1,jj] and mask[ii+1,jj] and mask[ii,jj-1] and (not mask[ii,jj+1]): # top boundary
          gradG[ii,jj,:,:,1] = gradG_y_top[ii,jj]
          grad_g_inv[ii,jj,:,:,1] = grad_g_inv_top[ii,jj]
          grad_sqrt_det_g[ii,jj,1] = grad_sqrt_det_g_y_top[ii,jj]
          
        elif (not mask[ii-1,jj]) and (not mask[ii+1,jj]) and mask[ii,jj-1] and (not mask[ii,jj+1]): # notbottom boundary
          gradG[ii,jj,:,:,0] = 0
          gradG[ii,jj,:,:,1] = gradG_y_top[ii,jj]
          grad_g_inv[ii,jj,:,:,0] = 0
          grad_sqrt_det_g[ii,jj,0] = 0
          grad_g_inv[ii,jj,:,:,1] = grad_g_inv_top[ii,jj]
          grad_sqrt_det_g[ii,jj,1] = grad_sqrt_det_g_y_top[ii,jj]
   
        else:
          # interior point
          pass
  
  for xx in range(2):
    for yy in range(2):
      for zz in range(2):
        gradG[:,:,xx,yy,zz] = mask * gradG[:,:,xx,yy,zz]

  return(gradG, grad_g_inv, grad_sqrt_det_g)
# end compute_g_derivs_orig

def compute_g_derivs(g, g_inv, sqrt_det_g, bdry_idx, bdry_map):
  xsz = g.shape[0]
  ysz = g.shape[1]
  gradG = np.zeros((xsz,ysz, 2, 2, 2))
  grad_g_inv = np.zeros((xsz,ysz, 2, 2, 2))
  grad_sqrt_det_g = np.zeros((xsz,ysz,2))

 # correct center derivatives (np.gradient) on all 4 boundaries

  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue
    
    b_idx = bdry_idx[bnum]
    
    if btype[0:8] == "interior":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],xx,yy,0] = diff.gradx_idx_2d(g[:,:,xx,yy], b_idx)
          gradG[b_idx[0],b_idx[1],xx,yy,1] = diff.grady_idx_2d(g[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],xx,yy,0] = diff.gradx_idx_2d(g_inv[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],xx,yy,1] = diff.grady_idx_2d(g_inv[:,:,xx,yy], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],0] = diff.gradx_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],1] = diff.grady_idx_2d(sqrt_det_g, b_idx)
                                                              
    elif btype == "left":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],xx,yy,0] = diff.left_diff_idx_2d(g[:,:,xx,yy], b_idx)
          gradG[b_idx[0],b_idx[1],xx,yy,1] = diff.grady_idx_2d(g[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],xx,yy,0] = diff.left_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],xx,yy,1] = diff.grady_idx_2d(g_inv[:,:,xx,yy], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],0] = diff.left_diff_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],1] = diff.grady_idx_2d(sqrt_det_g, b_idx)
      
    elif btype == "bottomleft":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],xx,yy,0] = diff.left_diff_idx_2d(g[:,:,xx,yy], b_idx)
          gradG[b_idx[0],b_idx[1],xx,yy,1] = diff.bottom_diff_idx_2d(g[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],xx,yy,0] = diff.left_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],xx,yy,1] = diff.bottom_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],0] = diff.left_diff_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],1] = diff.bottom_diff_idx_2d(sqrt_det_g, b_idx)
  
    elif btype == "topleft":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],xx,yy,0] = diff.left_diff_idx_2d(g[:,:,xx,yy], b_idx)
          gradG[b_idx[0],b_idx[1],xx,yy,1] = diff.top_diff_idx_2d(g[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],xx,yy,0] = diff.left_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],xx,yy,1] = diff.top_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],0] = diff.left_diff_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],1] = diff.top_diff_idx_2d(sqrt_det_g, b_idx)
          
    elif btype == "notright":
      # TODO confirm we want 0 in the y direction in this case
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],xx,yy,0] = diff.left_diff_idx_2d(g[:,:,xx,yy], b_idx)
          gradG[b_idx[0],b_idx[1],xx,yy,1] = diff.grady_idx_2d(g[:,:,xx,yy], [b_idx[0]+1, b_idx[1]])
          grad_g_inv[b_idx[0], b_idx[1],xx,yy,0] = diff.left_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],xx,yy,1] = diff.grady_idx_2d(g_inv[:,:,xx,yy], [b_idx[0]+1, b_idx[1]])

      grad_sqrt_det_g[b_idx[0], b_idx[1],0] = diff.left_diff_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],1] =  diff.grady_idx_2d(sqrt_det_g, [b_idx[0]+1, b_idx[1]])

    elif btype == "right":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],xx,yy,0] = diff.right_diff_idx_2d(g[:,:,xx,yy], b_idx)
          gradG[b_idx[0],b_idx[1],xx,yy,1] = diff.grady_idx_2d(g[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],xx,yy,0] = diff.right_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],xx,yy,1] = diff.grady_idx_2d(g_inv[:,:,xx,yy], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],0] = diff.right_diff_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],1] = diff.grady_idx_2d(sqrt_det_g, b_idx)
      
    elif btype == "bottomright":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],xx,yy,0] = diff.right_diff_idx_2d(g[:,:,xx,yy], b_idx)
          gradG[b_idx[0],b_idx[1],xx,yy,1] = diff.bottom_diff_idx_2d(g[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],xx,yy,0] = diff.right_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],xx,yy,1] = diff.bottom_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],0] = diff.right_diff_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],1] = diff.bottom_diff_idx_2d(sqrt_det_g, b_idx)
  
    elif btype == "topright":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],xx,yy,0] = diff.right_diff_idx_2d(g[:,:,xx,yy], b_idx)
          gradG[b_idx[0],b_idx[1],xx,yy,1] = diff.top_diff_idx_2d(g[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],xx,yy,0] = diff.right_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],xx,yy,1] = diff.top_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],0] = diff.right_diff_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],1] = diff.top_diff_idx_2d(sqrt_det_g, b_idx)
          
    elif btype == "notleft":
      # TODO confirm we want 0 in the y direction in this case
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],xx,yy,0] = diff.right_diff_idx_2d(g[:,:,xx,yy], b_idx)
          gradG[b_idx[0],b_idx[1],xx,yy,1] = diff.grady_idx_2d(g[:,:,xx,yy], [b_idx[0]-1, b_idx[1]])
          grad_g_inv[b_idx[0], b_idx[1],xx,yy,0] = diff.right_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],xx,yy,1] = diff.grady_idx_2d(g_inv[:,:,xx,yy], [b_idx[0]-1, b_idx[1]])

      grad_sqrt_det_g[b_idx[0], b_idx[1],0] = diff.right_diff_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],1] = diff.grady_idx_2d(sqrt_det_g, [b_idx[0]-1, b_idx[1]])
      
    elif btype == "bottom":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],xx,yy,0] = diff.gradx_idx_2d(g[:,:,xx,yy], b_idx)
          gradG[b_idx[0],b_idx[1],xx,yy,1] = diff.bottom_diff_idx_2d(g[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],xx,yy,0] = diff.gradx_idx_2d(g_inv[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],xx,yy,1] = diff.bottom_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],0] = diff.gradx_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],1] = diff.bottom_diff_idx_2d(sqrt_det_g, b_idx)

    elif btype == "nottop":
      # TODO confirm want 0 in x direction
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],xx,yy,0] = diff.gradx_idx_2d(g[:,:,xx,yy], [b_idx[0], b_idx[1]+1])
          gradG[b_idx[0],b_idx[1],xx,yy,1] = diff.bottom_diff_idx_2d(g[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],xx,yy,0] = diff.gradx_idx_2d(g_inv[:,:,xx,yy],[b_idx[0], b_idx[1]+1])
          grad_g_inv[b_idx[0],b_idx[1],xx,yy,1] = diff.bottom_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],0] = diff.gradx_idx_2d(sqrt_det_g, [b_idx[0], b_idx[1]+1])
      grad_sqrt_det_g[b_idx[0], b_idx[1],1] = diff.bottom_diff_idx_2d(sqrt_det_g, b_idx)
  
    elif btype == "top":
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],xx,yy,0] = diff.gradx_idx_2d(g[:,:,xx,yy], b_idx)
          gradG[b_idx[0],b_idx[1],xx,yy,1] = diff.top_diff_idx_2d(g[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],xx,yy,0] = diff.gradx_idx_2d(g_inv[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0],b_idx[1],xx,yy,1] = diff.top_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],0] = diff.gradx_idx_2d(sqrt_det_g, b_idx)
      grad_sqrt_det_g[b_idx[0], b_idx[1],1] = diff.top_diff_idx_2d(sqrt_det_g, b_idx)

    elif btype == "notbottom":
      # TODO confirm want 0 in x direction
      for xx in range(2):
        for yy in range(2):
          gradG[b_idx[0], b_idx[1],xx,yy,0] = diff.gradx_idx_2d(g[:,:,xx,yy], [b_idx[0], b_idx[1]-1])
          gradG[b_idx[0],b_idx[1],xx,yy,1] = diff.top_diff_idx_2d(g[:,:,xx,yy], b_idx)
          grad_g_inv[b_idx[0], b_idx[1],xx,yy,0] = diff.gradx_idx_2d(g_inv[:,:,xx,yy],[b_idx[0], b_idx[1]-1])
          grad_g_inv[b_idx[0],b_idx[1],xx,yy,1] = diff.top_diff_idx_2d(g_inv[:,:,xx,yy], b_idx)

      grad_sqrt_det_g[b_idx[0], b_idx[1],0] =  diff.gradx_idx_2d(sqrt_det_g, [b_idx[0], b_idx[1]-1])
      grad_sqrt_det_g[b_idx[0], b_idx[1],1] = diff.top_diff_idx_2d(sqrt_det_g, b_idx)
      
              
    elif btype == "outside":
      # outside mask, skip
      pass

    else:
      # unrecognized type
      print(btype, "unrecognized.  Skipping")
      
  return(gradG, grad_g_inv, grad_sqrt_det_g)
# end compute_g_derivs

def compute_T_derivs_orig(TDir, mask):
  xsz = TDir.shape[0]
  ysz = TDir.shape[1]
  gradTx_delx = np.zeros((xsz,ysz))
  gradTx_dely = np.zeros((xsz,ysz))
  gradTy_delx = np.zeros((xsz,ysz))
  gradTy_dely = np.zeros((xsz,ysz))
  
  gradTx_delx, gradTx_dely = np.gradient(TDir[:,:,0])
  gradTy_delx, gradTy_dely = np.gradient(TDir[:,:,1])
  
  gradTx_delx_left = diff.left_diff_2d(TDir[:,:,0])
  gradTx_delx_right = diff.right_diff_2d(TDir[:,:,0])
  gradTx_dely_bottom = diff.bottom_diff_2d(TDir[:,:,0])
  gradTx_dely_top = diff.top_diff_2d(TDir[:,:,0])
  gradTy_delx_left = diff.left_diff_2d(TDir[:,:,1])
  gradTy_delx_right = diff.right_diff_2d(TDir[:,:,1])
  gradTy_dely_bottom = diff.bottom_diff_2d(TDir[:,:,1])
  gradTy_dely_top = diff.top_diff_2d(TDir[:,:,1])

  for ii in range(xsz):
    for jj in range(ysz):
      if mask[ii,jj]:
        if (not mask[ii-1,jj]) and mask[ii+1,jj] and mask[ii,jj-1] and mask[ii,jj+1]: # left boundary
          gradTx_delx[ii,jj] = gradTx_delx_left[ii,jj]
          gradTy_delx[ii,jj] = gradTy_delx_left[ii,jj]
    
        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottomleft boundary
          gradTx_delx[ii,jj] = gradTx_delx_left[ii,jj]
          gradTy_delx[ii,jj] = gradTy_delx_left[ii,jj]
          gradTx_dely[ii,jj] = gradTx_dely_bottom[ii,jj]
          gradTy_dely[ii,jj] = gradTy_dely_bottom[ii,jj]
    
        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and mask[ii,jj-1] and (not mask[ii,jj+1]): # topleft boundary
          gradTx_delx[ii,jj] = gradTx_delx_left[ii,jj]
          gradTy_delx[ii,jj] = gradTy_delx_left[ii,jj]
          gradTx_dely[ii,jj] = gradTx_dely_top[ii,jj]
          gradTy_dely[ii,jj] = gradTy_dely_top[ii,jj]
    
        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and (not mask[ii,jj-1]) and (not mask[ii,jj+1]): # notright boundary
          gradTx_delx[ii,jj] = gradTx_delx_left[ii,jj]
          gradTy_delx[ii,jj] = gradTy_delx_left[ii,jj]
          gradTx_dely[ii,jj] = 0
          gradTy_dely[ii,jj] = 0
    
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and mask[ii,jj-1] and mask[ii,jj+1]: # right boundary
          gradTx_delx[ii,jj] = gradTx_delx_right[ii,jj]
          gradTy_delx[ii,jj] = gradTy_delx_right[ii,jj]
    
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottomright boundary
          gradTx_delx[ii,jj] = gradTx_delx_right[ii,jj]
          gradTy_delx[ii,jj] = gradTy_delx_right[ii,jj]
          gradTx_dely[ii,jj] = gradTx_dely_bottom[ii,jj]
          gradTy_dely[ii,jj] = gradTy_dely_bottom[ii,jj]
                    
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and mask[ii,jj-1] and (not mask[ii,jj+1]): # topright boundary
          gradTx_delx[ii,jj] = gradTx_delx_right[ii,jj]
          gradTy_delx[ii,jj] = gradTy_delx_right[ii,jj]
          gradTx_dely[ii,jj] = gradTx_dely_top[ii,jj]
          gradTy_dely[ii,jj] = gradTy_dely_top[ii,jj]
            
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and (not mask[ii,jj+1]): # notleft boundary
          gradTx_delx[ii,jj] = gradTx_delx_right[ii,jj]
          gradTy_delx[ii,jj] = gradTy_delx_right[ii,jj]
          gradTx_dely[ii,jj] = 0
          gradTy_dely[ii,jj] = 0
            
        elif mask[ii-1,jj] and mask[ii+1,jj] and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottom boundary
          gradTx_dely[ii,jj] = gradTx_dely_bottom[ii,jj]
          gradTy_dely[ii,jj] = gradTy_dely_bottom[ii,jj]
        
        elif (not mask[ii-1,jj]) and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and mask[ii,jj+1]: # nottop boundary
          gradTx_delx[ii,jj] = 0
          gradTy_delx[ii,jj] = 0
          gradTx_dely[ii,jj] = gradTx_dely_bottom[ii,jj]
          gradTy_dely[ii,jj] = gradTy_dely_bottom[ii,jj]
        
        elif mask[ii-1,jj] and mask[ii+1,jj] and mask[ii,jj-1] and (not mask[ii,jj+1]): # top boundary
          gradTx_dely[ii,jj] = gradTx_dely_top[ii,jj]
          gradTy_dely[ii,jj] = gradTy_dely_top[ii,jj]
        
        elif (not mask[ii-1,jj]) and (not mask[ii+1,jj]) and mask[ii,jj-1] and (not mask[ii,jj+1]): # notbottom boundary
          gradTx_delx[ii,jj] = 0
          gradTy_delx[ii,jj] = 0
          gradTx_dely[ii,jj] = gradTx_dely_top[ii,jj]
          gradTy_dely[ii,jj] = gradTy_dely_top[ii,jj]
    
        else:
          # interior point
          pass

  return(gradTx_delx, gradTx_dely, gradTy_delx, gradTy_dely)
# end compute_T_derivs_orig

def compute_T_derivs(TDir, bdry_idx, bdry_map):
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
      #gradTx_delx[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(TDir[:,:,0], b_idx)
      #gradTx_dely[b_idx[0], b_idx[1]] = diff.grady_idx_2d(TDir[:,:,0], b_idx)
      #gradTy_delx[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(TDir[:,:,1], b_idx)
      #gradTy_dely[b_idx[0], b_idx[1]] = diff.grady_idx_2d(TDir[:,:,1], b_idx)
      gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]] = diff.eigv_gradx_idx_2d(TDir, b_idx)
      gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_grady_idx_2d(TDir, b_idx)
      
    elif btype == "left":
      #gradTx_delx[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(TDir[:,:,0], b_idx)
      #gradTx_dely[b_idx[0], b_idx[1]] = diff.grady_idx_2d(TDir[:,:,0], b_idx)
      #gradTy_delx[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(TDir[:,:,1], b_idx)
      #gradTy_dely[b_idx[0], b_idx[1]] = diff.grady_idx_2d(TDir[:,:,1], b_idx)
      gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]] = diff.eigv_left_idx_2d(TDir, b_idx)
      gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_grady_idx_2d(TDir, b_idx)

    elif btype == "bottomleft":
      ##gradTx_delx[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(TDir[:,:,0], b_idx)
      ##gradTx_dely[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(TDir[:,:,0], b_idx)
      ##gradTy_delx[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(TDir[:,:,1], b_idx)
      ##gradTy_dely[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]] = diff.eigv_left_idx_2d(TDir, b_idx)
      gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_bottom_idx_2d(TDir, b_idx)
      #gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]], gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_bottomleft_idx_2d(TDir, b_idx)

    elif btype == "topleft":
      ##gradTx_delx[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(TDir[:,:,0], b_idx)
      ##gradTx_dely[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(TDir[:,:,0], b_idx)
      ##gradTy_delx[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(TDir[:,:,1], b_idx)
      ##gradTy_dely[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]] = diff.eigv_left_idx_2d(TDir, b_idx)
      gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_top_idx_2d(TDir, b_idx)
      #gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]], gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_topleft_idx_2d(TDir, b_idx)
    elif btype == "notright":
      #gradTx_delx[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(TDir[:,:,0], b_idx)
      #gradTx_dely[b_idx[0], b_idx[1]] = diff.grady_idx_2d(TDir[:,:,0], [b_idx[0]+1, b_idx[1]])
      #gradTy_delx[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(TDir[:,:,1], b_idx)
      #gradTy_dely[b_idx[0], b_idx[1]] = diff.grady_idx_2d(TDir[:,:,1], [b_idx[0]+1, b_idx[1]])
      gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]] = diff.eigv_left_idx_2d(TDir, b_idx)
      gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_grady_idx_2d(TDir, [b_idx[0]+1, b_idx[1]])

    elif btype == "right":
      #gradTx_delx[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(TDir[:,:,0], b_idx)
      #gradTx_dely[b_idx[0], b_idx[1]] = diff.grady_idx_2d(TDir[:,:,0], b_idx)
      #gradTy_delx[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(TDir[:,:,1], b_idx)
      #gradTy_dely[b_idx[0], b_idx[1]] = diff.grady_idx_2d(TDir[:,:,1], b_idx)
      gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]] = diff.eigv_right_idx_2d(TDir, b_idx)
      gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_grady_idx_2d(TDir, b_idx)

    elif btype == "bottomright":
      ##gradTx_delx[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(TDir[:,:,0], b_idx)
      ##gradTx_dely[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(TDir[:,:,0], b_idx)
      ##gradTy_delx[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(TDir[:,:,1], b_idx)
      ##gradTy_dely[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]] = diff.eigv_right_idx_2d(TDir, b_idx)
      gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_bottom_idx_2d(TDir, b_idx)
      #gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]], gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_bottomright_idx_2d(TDir, b_idx)

    elif btype == "topright":
      ##gradTx_delx[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(TDir[:,:,0], b_idx)
      ##gradTx_dely[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(TDir[:,:,0], b_idx)
      ##gradTy_delx[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(TDir[:,:,1], b_idx)
      ##gradTy_dely[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]] = diff.eigv_right_idx_2d(TDir, b_idx)
      gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_top_idx_2d(TDir, b_idx)
      #gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]], gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_topright_idx_2d(TDir, b_idx)

    elif btype == "notleft":
      #gradTx_delx[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(TDir[:,:,0], b_idx)
      #gradTx_dely[b_idx[0], b_idx[1]] = diff.grady_idx_2d(TDir[:,:,0], [b_idx[0]-1, b_idx[1]])
      #gradTy_delx[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(TDir[:,:,1], b_idx)
      #gradTy_dely[b_idx[0], b_idx[1]] = diff.grady_idx_2d(TDir[:,:,1], [b_idx[0]-1, b_idx[1]])      
      gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]] = diff.eigv_right_idx_2d(TDir, b_idx)
      gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_grady_idx_2d(TDir,  [b_idx[0]-1, b_idx[1]])

    elif btype == "bottom":
      #gradTx_delx[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(TDir[:,:,0], b_idx)
      #gradTx_dely[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(TDir[:,:,0], b_idx)
      #gradTy_delx[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(TDir[:,:,1], b_idx)
      #gradTy_dely[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]] = diff.eigv_gradx_idx_2d(TDir, b_idx)
      gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_bottom_idx_2d(TDir, b_idx)

    elif btype == "nottop":
      #gradTx_delx[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(TDir[:,:,0],  [b_idx[0], b_idx[1]+1])
      #gradTx_dely[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(TDir[:,:,0], b_idx)
      #gradTy_delx[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(TDir[:,:,1], [b_idx[0], b_idx[1]+1])
      #gradTy_dely[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]] = diff.eigv_gradx_idx_2d(TDir, [b_idx[0], b_idx[1]+1])
      gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_bottom_idx_2d(TDir, b_idx)
      
    elif btype == "top":
      #gradTx_delx[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(TDir[:,:,0], b_idx)
      #gradTx_dely[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(TDir[:,:,0], b_idx)
      #gradTy_delx[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(TDir[:,:,1], b_idx)
      #gradTy_dely[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(TDir[:,:,1], b_idx)
      gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]] = diff.eigv_gradx_idx_2d(TDir, b_idx)
      gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_top_idx_2d(TDir, b_idx)
      
    elif btype == "notbottom":
      #gradTx_delx[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(TDir[:,:,0],  [b_idx[0], b_idx[1]-1])
      #gradTx_dely[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(TDir[:,:,0], b_idx)
      #gradTy_delx[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(TDir[:,:,1], [b_idx[0], b_idx[1]-1])
      #gradTy_dely[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(TDir[:,:,1], b_idx)      
      gradTx_delx[b_idx[0], b_idx[1]], gradTy_delx[b_idx[0], b_idx[1]] = diff.eigv_gradx_idx_2d(TDir, [b_idx[0], b_idx[1]-1])
      gradTx_dely[b_idx[0], b_idx[1]], gradTy_dely[b_idx[0], b_idx[1]] = diff.eigv_top_idx_2d(TDir, b_idx)

    elif btype == "outside":
      # outside mask, skip
      pass

    else:
      # unrecognized type
      print(btype, "unrecognized.  Skipping")
      
  return(gradTx_delx, gradTx_dely, gradTy_delx, gradTy_dely)
# end compute_T_derivs

def compute_nabla_derivs_orig(nabla_TT, sqrt_det_nabla_TT):
  xsz = nabla_TT.shape[0]
  ysz = nabla_TT.shape[1]
  grad_sqrt_det_nabla_TT = np.zeros((xsz,ysz, 2, 2))
  grad_sqrt_det_nabla_TT[:,:,0,0], grad_sqrt_det_nabla_TT[:,:,0,1] = np.gradient(sqrt_det_nabla_TT[:,:,0])
  grad_sqrt_det_nabla_TT[:,:,1,0], grad_sqrt_det_nabla_TT[:,:,1,1] = np.gradient(sqrt_det_nabla_TT[:,:,1])
  
  gradx_nabla_TT_x_left = diff.left_diff_2d(nabla_TT[:,:,0])
  gradx_nabla_TT_x_right = diff.right_diff_2d(nabla_TT[:,:,0])
  grady_nabla_TT_y_bottom = diff.bottom_diff_2d(nabla_TT[:,:,1])
  grady_nabla_TT_y_top = diff.top_diff_2d(nabla_TT[:,:,1])
  grad_nabla_TT = np.zeros((xsz,ysz,2,2))
  grad_nabla_TT[:,:,0,0], grad_nabla_TT[:,:,0,1] = np.gradient(nabla_TT[:,:,0])
  grad_nabla_TT[:,:,1,0], grad_nabla_TT[:,:,1,1] = np.gradient(nabla_TT[:,:,1])

  return(grad_nabla_TT, grad_sqrt_det_nabla_TT)
# end compute_nabla_derivs_orig

def compute_nabla_derivs(nabla_TT, sqrt_det_nabla_TT, bdry_idx, bdry_map):
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
      grad_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.gradx_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.grady_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.gradx_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.grady_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      
    elif btype == "left":
      grad_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.left_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.grady_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.left_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.grady_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "bottomleft":
      grad_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.left_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.left_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "topleft":
      grad_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.left_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.top_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.left_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.top_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "notright":
      grad_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.left_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.grady_idx_2d(nabla_TT[:,:,0], [b_idx[0]+1, b_idx[1]])
      grad_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.left_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.grady_idx_2d(nabla_TT[:,:,1], [b_idx[0]+1, b_idx[1]])

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,0], [b_idx[0]+1, b_idx[1]])
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.left_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,1], [b_idx[0]+1, b_idx[1]])

    elif btype == "right":
      grad_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.right_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.grady_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.right_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.grady_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "bottomright":
      grad_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.right_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.right_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "topright":
      grad_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.right_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.top_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.right_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.top_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "notleft":
      grad_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.right_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.grady_idx_2d(nabla_TT[:,:,0], [b_idx[0]-1, b_idx[1]])
      grad_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.right_diff_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.grady_idx_2d(nabla_TT[:,:,1], [b_idx[0]-1, b_idx[1]])

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,0], [b_idx[0]-1, b_idx[1]])
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.right_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.grady_idx_2d(sqrt_det_nabla_TT[:,:,1], [b_idx[0]-1, b_idx[1]])

    elif btype == "bottom":
      grad_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.gradx_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.gradx_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      
    elif btype == "nottop":
      grad_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.gradx_idx_2d(nabla_TT[:,:,0], [b_idx[0], b_idx[1]+1])
      grad_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.gradx_idx_2d(nabla_TT[:,:,1], [b_idx[0], b_idx[1]+1])
      grad_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.bottom_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,0], [b_idx[0], b_idx[1]+1])
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,1], [b_idx[0], b_idx[1]+1])
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.bottom_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      
    elif btype == "top":
      grad_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.gradx_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.top_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.gradx_idx_2d(nabla_TT[:,:,1], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.top_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

    elif btype == "notbottom":
      grad_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.gradx_idx_2d(nabla_TT[:,:,0], [b_idx[0], b_idx[1]-1])
      grad_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.top_diff_idx_2d(nabla_TT[:,:,0], b_idx)
      grad_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.gradx_idx_2d(nabla_TT[:,:,1], [b_idx[0], b_idx[1]-1])
      grad_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.top_diff_idx_2d(nabla_TT[:,:,1], b_idx)

      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,0], [b_idx[0], b_idx[1]-1])
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],0,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,0], b_idx)
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,0] = diff.gradx_idx_2d(sqrt_det_nabla_TT[:,:,1], [b_idx[0], b_idx[1]-1])
      grad_sqrt_det_nabla_TT[b_idx[0], b_idx[1],1,1] = diff.top_diff_idx_2d(sqrt_det_nabla_TT[:,:,1], b_idx)

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
  alpha_gradX, alpha_gradY = np.gradient(alpha)
  alpha_gradX_left = diff.left_diff_2d(alpha)
  alpha_gradX_right = diff.right_diff_2d(alpha)
  alpha_gradY_bottom = diff.bottom_diff_2d(alpha)
  alpha_gradY_top = diff.top_diff_2d(alpha)

  for ii in range(xsz):
    for jj in range(ysz):
      if mask[ii,jj]:
        if (not mask[ii-1,jj]) and mask[ii+1,jj] and mask[ii,jj-1] and mask[ii,jj+1]: # left boundary
          alpha_gradX[ii,jj] = alpha_gradX_left[ii,jj]

        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottomleft boundary
          alpha_gradX[ii,jj] = alpha_gradX_left[ii,jj]
          alpha_gradY[ii,jj] = alpha_gradY_bottom[ii,jj]

        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and mask[ii,jj-1] and (not mask[ii,jj+1]): # topleft boundary
          alpha_gradX[ii,jj] = alpha_gradX_left[ii,jj]
          alpha_gradY[ii,jj] = alpha_gradY_top[ii,jj]

        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and (not mask[ii,jj-1]) and (not mask[ii,jj+1]): # notright boundary
          alpha_gradX[ii,jj] = alpha_gradX_left[ii,jj]
          alpha_gradY[ii,jj] = 0

        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and mask[ii,jj-1] and mask[ii,jj+1]: # right boundary
          alpha_gradX[ii,jj] = alpha_gradX_right[ii,jj]

        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottomright boundary
          alpha_gradX[ii,jj] = alpha_gradX_right[ii,jj]
          alpha_gradY[ii,jj] = alpha_gradY_bottom[ii,jj]
                  
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and mask[ii,jj-1] and (not mask[ii,jj+1]): # topright boundary
          alpha_gradX[ii,jj] = alpha_gradX_right[ii,jj]
          alpha_gradY[ii,jj] = alpha_gradY_top[ii,jj]
          
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and (not mask[ii,jj+1]): # notleft boundary
          alpha_gradX[ii,jj] = alpha_gradX_right[ii,jj]
          alpha_gradY[ii,jj] = 0
          
        elif mask[ii-1,jj] and mask[ii+1,jj] and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottom boundary
          alpha_gradY[ii,jj] = alpha_gradY_bottom[ii,jj]
          
        elif (not mask[ii-1,jj]) and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and mask[ii,jj+1]: # nottop boundary
          alpha_gradX[ii,jj] = 0
          alpha_gradY[ii,jj] = alpha_gradY_bottom[ii,jj]
      
        elif mask[ii-1,jj] and mask[ii+1,jj] and mask[ii,jj-1] and (not mask[ii,jj+1]): # top boundary
          alpha_gradY[ii,jj] = alpha_gradY_top[ii,jj]
          alpha_gradY[ii,jj] = alpha_gradY_top[ii,jj]
      
        elif (not mask[ii-1,jj]) and (not mask[ii+1,jj]) and mask[ii,jj-1] and (not mask[ii,jj+1]): # notbottom boundary
          alpha_gradX[ii,jj] = 0
          alpha_gradY[ii,jj] = alpha_gradY_top[ii,jj]

        else:
          # interior point
          pass

  return(alpha_gradX, alpha_gradY)
# end compute_alpha_derivs_orig

def compute_alpha_derivs(alpha, bdry_idx, bdry_map):
  xsz = alpha.shape[0]
  ysz = alpha.shape[1]
  alpha_gradX = np.zeros((xsz,ysz))
  alpha_gradY = np.zeros((xsz,ysz))

  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue

    b_idx = bdry_idx[bnum]
    
    if btype[0:8] == "interior":
      alpha_gradX[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(alpha, b_idx)
      alpha_gradY[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha, b_idx)
      
    elif btype == "left":
      alpha_gradX[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(alpha, b_idx)
      alpha_gradY[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha, b_idx)

    elif btype == "bottomleft":
      alpha_gradX[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(alpha, b_idx)
      alpha_gradY[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(alpha, b_idx)

    elif btype == "topleft":
      alpha_gradX[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(alpha, b_idx)
      alpha_gradY[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(alpha, b_idx)

    elif btype == "notright":
      alpha_gradX[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(alpha, b_idx)
      alpha_gradY[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha, [b_idx[0]+1, b_idx[1]])

    elif btype == "right":
      alpha_gradX[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(alpha, b_idx)
      alpha_gradY[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha, b_idx)

    elif btype == "bottomright":
      alpha_gradX[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(alpha, b_idx)
      alpha_gradY[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(alpha, b_idx)

    elif btype == "topright":
      alpha_gradX[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(alpha, b_idx)
      alpha_gradY[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(alpha, b_idx)

    elif btype == "notleft":
      alpha_gradX[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(alpha, b_idx)
      alpha_gradY[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha, [b_idx[0]-1, b_idx[1]])

    elif btype == "bottom":
      alpha_gradX[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(alpha, b_idx)
      alpha_gradY[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(alpha, b_idx)

    elif btype == "nottop":
      alpha_gradX[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(alpha, [b_idx[0], b_idx[1]+1])
      alpha_gradY[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(alpha, b_idx)
      
    elif btype == "top":
      alpha_gradX[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(alpha, b_idx)
      alpha_gradY[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(alpha, b_idx)

    elif btype == "notbottom":
      alpha_gradX[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(alpha, [b_idx[0], b_idx[1]-1])
      alpha_gradY[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(alpha, b_idx)

    elif btype == "outside":
      # outside mask, skip
      pass

    else:
      # unrecognized type
      print(btype, "unrecognized.  Skipping")
  return(alpha_gradX, alpha_gradY)
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
  alpha_div = np.zeros((xsz,ysz))
  alpha_gradX_X = np.zeros((xsz,ysz))
  alpha_gradX_Y = np.zeros((xsz,ysz))
  #alpha_gradY_X = np.zeros((xsz,ysz))
  alpha_gradY_Y = np.zeros((xsz,ysz))
  
  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue

    b_idx = bdry_idx[bnum]
    
    if btype[0:8] == "interior":
      alpha_gradX_X[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha_gradY, b_idx)
      
    elif btype == "left":
      alpha_gradX_X[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha_gradY, b_idx)

    elif btype == "bottomleft":
      alpha_gradX_X[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(alpha_gradY, b_idx)

    elif btype == "topleft":
      alpha_gradX_X[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(alpha_gradY, b_idx)
    elif btype == "notright":
      alpha_gradX_X[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha_gradX, [b_idx[0]+1, b_idx[1]])
      #alpha_gradY_X[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha_gradY, [b_idx[0]+1, b_idx[1]])

    elif btype == "right":
      alpha_gradX_X[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha_gradY, b_idx)

    elif btype == "bottomright":
      alpha_gradX_X[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(alpha_gradY, b_idx)

    elif btype == "topright":
      alpha_gradX_X[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(alpha_gradY, b_idx)

    elif btype == "notleft":
      alpha_gradX_X[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha_gradX, [b_idx[0]-1, b_idx[1]])
      #alpha_gradY_X[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1]] = diff.grady_idx_2d(alpha_gradY, [b_idx[0]-1, b_idx[1]])

    elif btype == "bottom":
      alpha_gradX_X[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(alpha_gradY, b_idx)

    elif btype == "nottop":
      alpha_gradX_X[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(alpha_gradX, [b_idx[0], b_idx[1]+1])
      alpha_gradX_Y[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1]] = 0
      alpha_gradY_Y[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(alpha_gradY, b_idx)
      
    elif btype == "top":
      alpha_gradX_X[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(alpha_gradX, b_idx)
      alpha_gradX_Y[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(alpha_gradY, b_idx)
      alpha_gradY_Y[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(alpha_gradY, b_idx)

    elif btype == "notbottom":
      alpha_gradX_X[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(alpha_gradX, [b_idx[0], b_idx[1]-1])
      alpha_gradX_Y[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(alpha_gradX, b_idx)
      #alpha_gradY_X[b_idx[0], b_idx[1]] = 0
      alpha_gradY_Y[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(alpha_gradY, b_idx)

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

def neumann_conditions_rhs_orig(nabla_TT, g, grad_det_g_x_g_inv, grad_g_inv_x_det_g, det_g_x_g_inv, sqrt_det_g, mask):
  xsz = nabla_TT.shape[0]
  ysz = nabla_TT.shape[1]

  bdry_term = np.zeros((xsz,ysz))

  for ii in range(xsz):
    for jj in range(ysz):
      if mask[ii,jj]:
        
        if sqrt_det_g[ii,jj]:
          denom = sqrt_det_g[ii,jj]
        else:
          denom = 1
  
        if (not mask[ii-1,jj]) and mask[ii+1,jj] and mask[ii,jj-1] and mask[ii,jj+1]: # left boundary
          Bx = 2 * nabla_TT[ii,jj,0] * g[ii,jj,0,0] + 2 * nabla_TT[ii,jj,1] * g[ii,jj,1,0]
          # By = 0
        
          bdry_term[ii,jj] = (grad_det_g_x_g_inv[ii,jj,0,0,0] + grad_g_inv_x_det_g[ii,jj,0,0,0] \
                              + grad_det_g_x_g_inv[ii,jj,1,1,0] + grad_g_inv_x_det_g[ii,jj,1,0,1] \
                              - 3 * det_g_x_g_inv[ii,jj,0,0]) * Bx / denom
          #rhs[ii,jj] = rhs[ii,jj] - bdry_term[ii,jj]
        
        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottomleft boundary
          Bx = nabla_TT[ii,jj,0] * g[ii,jj,0,0] + nabla_TT[ii,jj,0] * g[ii,jj,0,1] \
               + nabla_TT[ii,jj,1] * g[ii,jj,1,0] + nabla_TT[ii,jj,1] * g[ii,jj,1,1]
          By = Bx
          bdry_term[ii,jj] = ( (grad_det_g_x_g_inv[ii,jj,0,0,0] + grad_g_inv_x_det_g[ii,jj,0,0,0] \
                                + grad_det_g_x_g_inv[ii,jj,1,1,0] + grad_g_inv_x_det_g[ii,jj,1,0,1] \
                                - 3 * det_g_x_g_inv[ii,jj,0,0]) * Bx \
                               +(grad_det_g_x_g_inv[ii,jj,1,1,1] + grad_g_inv_x_det_g[ii,jj,1,1,1] \
                                 + grad_det_g_x_g_inv[ii,jj,0,0,1] + grad_g_inv_x_det_g[ii,jj,0,1,0] \
                                 - 3 * det_g_x_g_inv[ii,jj,1,1]) * By ) / denom
          #rhs[ii,jj] = rhs[ii,jj] - bdry_term[ii,jj]
          
        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and mask[ii,jj-1] and (not mask[ii,jj+1]): # topleft boundary
          Bx = nabla_TT[ii,jj,0] * g[ii,jj,0,0] - nabla_TT[ii,jj,0] * g[ii,jj,0,1] \
               + nabla_TT[ii,jj,1] * g[ii,jj,1,0] - nabla_TT[ii,jj,1] * g[ii,jj,1,1]
          By = -Bx
          bdry_term[ii,jj] = ( (grad_det_g_x_g_inv[ii,jj,0,0,0] + grad_g_inv_x_det_g[ii,jj,0,0,0] \
                                + grad_det_g_x_g_inv[ii,jj,1,1,0] + grad_g_inv_x_det_g[ii,jj,1,0,1] \
                                - 3 * det_g_x_g_inv[ii,jj,0,0]) * Bx \
                               +(grad_det_g_x_g_inv[ii,jj,1,1,1] + grad_g_inv_x_det_g[ii,jj,1,1,1] \
                                 + grad_det_g_x_g_inv[ii,jj,0,0,1] + grad_g_inv_x_det_g[ii,jj,0,1,0] \
                                 + 3 * det_g_x_g_inv[ii,jj,1,1]) * By ) / denom
          #rhs[ii,jj] = rhs[ii,jj] - bdry_term[ii,jj]
          
        elif (not mask[ii-1,jj]) and mask[ii+1,jj] and (not mask[ii,jj-1]) and (not mask[ii,jj+1]): # notright boundary
          Bx = 2 * nabla_TT[ii,jj,0] * g[ii,jj,0,0] + 2 * nabla_TT[ii,jj,1] * g[ii,jj,1,0]
          By = Bx
          # By = 0
          bdry_term[ii,jj] = ( (grad_det_g_x_g_inv[ii,jj,0,0,0] + grad_g_inv_x_det_g[ii,jj,0,0,0] \
                                + grad_det_g_x_g_inv[ii,jj,1,1,0] + grad_g_inv_x_det_g[ii,jj,1,0,1] \
                                - det_g_x_g_inv[ii,jj,0,1] - det_g_x_g_inv[ii,jj,1,0]) * Bx \
                               +(grad_det_g_x_g_inv[ii,jj,1,1,1] + grad_g_inv_x_det_g[ii,jj,1,1,1] \
                                 + grad_det_g_x_g_inv[ii,jj,0,0,1] + grad_g_inv_x_det_g[ii,jj,0,1,0] \
                                 - det_g_x_g_inv[ii,jj,0,1] - det_g_x_g_inv[ii,jj,1,0]) * By ) / denom
          #rhs[ii,jj] = rhs[ii,jj] - bdry_term[ii,jj]
          
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and mask[ii,jj-1] and mask[ii,jj+1]: # right boundary
          Bx = 2 * nabla_TT[ii,jj,0] * g[ii,jj,0,0] + 2 * nabla_TT[ii,jj,1] * g[ii,jj,1,0]
          # By = 0
          bdry_term[ii,jj] = (grad_det_g_x_g_inv[ii,jj,0,0,0] + grad_g_inv_x_det_g[ii,jj,0,0,0] \
                              + grad_det_g_x_g_inv[ii,jj,1,1,0] + grad_g_inv_x_det_g[ii,jj,1,0,1] \
                              + 3 * det_g_x_g_inv[ii,jj,0,0]) * Bx / denom
          #rhs[ii,jj] = rhs[ii,jj] - bdry_term[ii,jj]
          
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottomright boundary
          Bx = nabla_TT[ii,jj,0] * g[ii,jj,0,0] - nabla_TT[ii,jj,0] * g[ii,jj,0,1] \
               + nabla_TT[ii,jj,1] * g[ii,jj,1,0] - nabla_TT[ii,jj,1] * g[ii,jj,1,1]
          By = -Bx
          bdry_term[ii,jj] = ( (grad_det_g_x_g_inv[ii,jj,0,0,0] + grad_g_inv_x_det_g[ii,jj,0,0,0] \
                                + grad_det_g_x_g_inv[ii,jj,1,1,0] + grad_g_inv_x_det_g[ii,jj,1,0,1] \
                                + 3 * det_g_x_g_inv[ii,jj,0,0]) * Bx \
                               +(grad_det_g_x_g_inv[ii,jj,1,1,1] + grad_g_inv_x_det_g[ii,jj,1,1,1] \
                                 + grad_det_g_x_g_inv[ii,jj,0,0,1] + grad_g_inv_x_det_g[ii,jj,0,1,0] \
                                 - 3 * det_g_x_g_inv[ii,jj,1,1]) * By ) / denom
          #rhs[ii,jj] = rhs[ii,jj] - bdry_term[ii,jj]
          
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and mask[ii,jj-1] and (not mask[ii,jj+1]): # topright boundary
          Bx = nabla_TT[ii,jj,0] * g[ii,jj,0,0] + nabla_TT[ii,jj,0] * g[ii,jj,0,1] \
               + nabla_TT[ii,jj,1] * g[ii,jj,1,0] + nabla_TT[ii,jj,1] * g[ii,jj,1,1]
          By = Bx
          bdry_term[ii,jj] = ( (grad_det_g_x_g_inv[ii,jj,0,0,0] + grad_g_inv_x_det_g[ii,jj,0,0,0] \
                                + grad_det_g_x_g_inv[ii,jj,1,1,0] + grad_g_inv_x_det_g[ii,jj,1,0,1] \
                                + 3 * det_g_x_g_inv[ii,jj,0,0]) * Bx \
                               +(grad_det_g_x_g_inv[ii,jj,1,1,1] + grad_g_inv_x_det_g[ii,jj,1,1,1] \
                                 + grad_det_g_x_g_inv[ii,jj,0,0,1] + grad_g_inv_x_det_g[ii,jj,0,1,0] \
                                 + 3 * det_g_x_g_inv[ii,jj,1,1]) * By ) / denom
          #rhs[ii,jj] = rhs[ii,jj] - bdry_term[ii,jj]
          
        elif mask[ii-1,jj] and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and (not mask[ii,jj+1]): # notleft boundary
          Bx = 2 * nabla_TT[ii,jj,0] * g[ii,jj,0,0] + 2 * nabla_TT[ii,jj,1] * g[ii,jj,1,0]
          By = Bx
          # By = 0
          bdry_term[ii,jj] = ( (grad_det_g_x_g_inv[ii,jj,0,0,0] + grad_g_inv_x_det_g[ii,jj,0,0,0] \
                                + grad_det_g_x_g_inv[ii,jj,1,1,0] + grad_g_inv_x_det_g[ii,jj,1,0,1] \
                                + det_g_x_g_inv[ii,jj,0,1] + det_g_x_g_inv[ii,jj,1,0]) * Bx \
                               +(grad_det_g_x_g_inv[ii,jj,1,1,1] + grad_g_inv_x_det_g[ii,jj,1,1,1] \
                                 + grad_det_g_x_g_inv[ii,jj,0,0,1] + grad_g_inv_x_det_g[ii,jj,0,1,0] \
                                 + det_g_x_g_inv[ii,jj,0,1] + det_g_x_g_inv[ii,jj,1,0]) * By ) / denom
          #rhs[ii,jj] = rhs[ii,jj] - bdry_term[ii,jj]
          
        elif mask[ii-1,jj] and mask[ii+1,jj] and (not mask[ii,jj-1]) and mask[ii,jj+1]: # bottom boundary
          # Bx = 0
          By = 2 * nabla_TT[ii,jj,0] * g[ii,jj,0,1] + 2 * nabla_TT[ii,jj,1] * g[ii,jj,1,1]
          bdry_term[ii,jj] = (grad_det_g_x_g_inv[ii,jj,1,1,1] + grad_g_inv_x_det_g[ii,jj,1,1,1] \
                              + grad_det_g_x_g_inv[ii,jj,0,0,1] + grad_g_inv_x_det_g[ii,jj,0,1,0] \
                              - 3 * det_g_x_g_inv[ii,jj,1,1]) * By / denom
          #rhs[ii,jj] = rhs[ii,jj] - bdry_term[ii,jj]
          
        elif (not mask[ii-1,jj]) and (not mask[ii+1,jj]) and (not mask[ii,jj-1]) and mask[ii,jj+1]: # nottop boundary
          # Bx = 0
          By = 2 * nabla_TT[ii,jj,0] * g[ii,jj,0,1] + 2 * nabla_TT[ii,jj,1] * g[ii,jj,1,1]
          Bx = By
          bdry_term[ii,jj] = ( (grad_det_g_x_g_inv[ii,jj,0,0,0] + grad_g_inv_x_det_g[ii,jj,0,0,0] \
                                + grad_det_g_x_g_inv[ii,jj,1,1,0] + grad_g_inv_x_det_g[ii,jj,1,0,1] \
                                - det_g_x_g_inv[ii,jj,0,1] - det_g_x_g_inv[ii,jj,1,0]) * Bx \
                               +(grad_det_g_x_g_inv[ii,jj,1,1,1] + grad_g_inv_x_det_g[ii,jj,1,1,1] \
                                 + grad_det_g_x_g_inv[ii,jj,0,0,1] + grad_g_inv_x_det_g[ii,jj,0,1,0] \
                                 - det_g_x_g_inv[ii,jj,0,1] - det_g_x_g_inv[ii,jj,1,0]) * By ) / denom
          #bdry_term[ii,jj] = ( (grad_det_g_x_g_inv[ii,jj,1,1,1] + grad_g_inv_x_det_g[ii,jj,1,1,1] \
          #                      + grad_det_g_x_g_inv[ii,jj,0,0,1] + grad_g_inv_x_det_g[ii,jj,0,1,0] \
          #                      - det_g_x_g_inv[ii,jj,0,1] - det_g_x_g_inv[ii,jj,1,0]) * By ) / denom
          #rhs[ii,jj] = rhs[ii,jj] - bdry_term[ii,jj]
          
        elif mask[ii-1,jj] and mask[ii+1,jj] and mask[ii,jj-1] and (not mask[ii,jj+1]): # top boundary
          # Bx = 0
          By = 2 * nabla_TT[ii,jj,0] * g[ii,jj,0,1] + 2 * nabla_TT[ii,jj,1] * g[ii,jj,1,1]
          bdry_term[ii,jj] = (grad_det_g_x_g_inv[ii,jj,1,1,1] + grad_g_inv_x_det_g[ii,jj,1,1,1] \
                              + grad_det_g_x_g_inv[ii,jj,0,0,1] + grad_g_inv_x_det_g[ii,jj,0,1,0] \
                              + 3 * det_g_x_g_inv[ii,jj,1,1]) * By / denom
          #rhs[ii,jj] = rhs[ii,jj] - bdry_term[ii,jj]
          
        elif (not mask[ii-1,jj]) and (not mask[ii+1,jj]) and mask[ii,jj-1] and (not mask[ii,jj+1]): # notbottom boundary
          # Bx = 0
          By = 2 * nabla_TT[ii,jj,0] * g[ii,jj,0,1] + 2 * nabla_TT[ii,jj,1] * g[ii,jj,1,1]
          Bx = By
          bdry_term[ii,jj] = ( (grad_det_g_x_g_inv[ii,jj,0,0,0] + grad_g_inv_x_det_g[ii,jj,0,0,0] \
                                + grad_det_g_x_g_inv[ii,jj,1,1,0] + grad_g_inv_x_det_g[ii,jj,1,0,1] \
                                + det_g_x_g_inv[ii,jj,0,1] + det_g_x_g_inv[ii,jj,1,0]) * Bx \
                               +(grad_det_g_x_g_inv[ii,jj,1,1,1] + grad_g_inv_x_det_g[ii,jj,1,1,1] \
                                 + grad_det_g_x_g_inv[ii,jj,0,0,1] + grad_g_inv_x_det_g[ii,jj,0,1,0] \
                                 + det_g_x_g_inv[ii,jj,0,1] + det_g_x_g_inv[ii,jj,1,0]) * By ) / denom
          #bdry_term[ii,jj] = ( (grad_det_g_x_g_inv[ii,jj,1,1,1] + grad_g_inv_x_det_g[ii,jj,1,1,1] \
          #                      + grad_det_g_x_g_inv[ii,jj,0,0,1] + grad_g_inv_x_det_g[ii,jj,0,1,0] \
          #                      + det_g_x_g_inv[ii,jj,0,1] + det_g_x_g_inv[ii,jj,1,0]) * By ) / denom
          #rhs[ii,jj] = rhs[ii,jj] - bdry_term[ii,jj]
 
        else:
          # interior point
          pass
            
  return(bdry_term)
# end neumann_conditions_rhs_orig

def neumann_conditions_rhs(nabla_TT, g, grad_det_g_x_g_inv, grad_g_inv_x_det_g, det_g_x_g_inv, sqrt_det_g, bdry_idx, bdry_map):
  xsz = nabla_TT.shape[0]
  ysz = nabla_TT.shape[1]
  bdry_term = np.zeros((xsz,ysz))

  for btype, bnum in bdry_map.items():
    if bnum == 0:
      # skip, since outside
      continue

    b_idx = bdry_idx[bnum]
    
    if btype[0:8] == "interior":
      # No boundary adjustments for interior points
      pass
      
    elif btype == "left":
      Bx = 2 * nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,0] + 2 * nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,0]
      
      bdry_term[b_idx[0], b_idx[1]] = (grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,0,0] \
                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,0,1] \
                                       - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],0,0]) * Bx / sqrt_det_g[b_idx[0], b_idx[1]]
      
    elif btype == "bottomleft":
      Bx = nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,0] + nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,1] \
           + nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,0] + nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,1]
      By = Bx # TODO skip copy as it's wasteful
      
      bdry_term[b_idx[0], b_idx[1]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,0,0] \
                                         + grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,0,1] \
                                         - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],0,0]) * Bx \
                                        +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,1,1] \
                                          + grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,1,0] \
                                          - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],1,1]) * By ) / sqrt_det_g[b_idx[0], b_idx[1]]

    elif btype == "topleft":
     Bx = nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,0] - nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,1] \
          + nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,0] - nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,1]
     By = -Bx
     bdry_term[b_idx[0], b_idx[1]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,0,0] \
                                        + grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,0,1] \
                                        - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],0,0]) * Bx \
                                       +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,1,1] \
                                         + grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,1,0] \
                                         + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],1,1]) * By ) / sqrt_det_g[b_idx[0], b_idx[1]]
     
    elif btype == "notright":
      Bx = 2 * nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,0] + 2 * nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,0]
      By = 0
      #bdry_term[b_idx[0], b_idx[1]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,0,0] \
      #                                   + grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,0,1] \
      #                                   - det_g_x_g_inv[b_idx[0], b_idx[1],0,1] - det_g_x_g_inv[b_idx[0], b_idx[1],1,0]) * Bx ) / sqrt_det_g[b_idx[0], b_idx[1]]
      #                                  #+(grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,1,1] \
      #                                  #  + grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,1,0] \
      #                                  #  - det_g_x_g_inv[b_idx[0], b_idx[1],0,1] - det_g_x_g_inv[b_idx[0], b_idx[1],1,0]) * By ) / sqrt_det_g[b_idx[0], b_idx[1]]
      # Fix to match other cases, TODO fix derivations
      bdry_term[b_idx[0], b_idx[1]] = (grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,0,0] \
                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,0,1]  \
                                       - det_g_x_g_inv[b_idx[0], b_idx[1],0,1] - det_g_x_g_inv[b_idx[0], b_idx[1],1,0] \
                                       - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],0,0]) * Bx / sqrt_det_g[b_idx[0], b_idx[1]]

    elif btype == "right":
      Bx = 2 * nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,0] + 2 * nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,0]
      # By = 0
      bdry_term[b_idx[0], b_idx[1]] = (grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,0,0] \
                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,0,1] \
                                       + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],0,0]) * Bx /  sqrt_det_g[b_idx[0], b_idx[1]]

    elif btype == "bottomright":
     Bx = nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,0] - nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,1] \
          + nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,0] - nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,1]
     By = -Bx
     bdry_term[b_idx[0], b_idx[1]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,0,0] \
                                        + grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,0,1] \
                                        + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],0,0]) * Bx \
                                       +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,1,1] \
                                         + grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,1,0] \
                                         - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],1,1]) * By ) / sqrt_det_g[b_idx[0], b_idx[1]]

    elif btype == "topright":
      Bx = nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,0] + nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,1] \
           + nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,0] + nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,1]
      By = Bx # TODO skip copy as it's wasteful
      bdry_term[b_idx[0], b_idx[1]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,0,0] \
                                         + grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,0,1] \
                                         + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],0,0]) * Bx \
                                        +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,1,1] \
                                          + grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,1,0] \
                                          + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],1,1]) * By ) / sqrt_det_g[b_idx[0], b_idx[1]]

    elif btype == "notleft":
      Bx = 2 * nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,0] + 2 * nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,0]
      By = 0
      #bdry_term[b_idx[0], b_idx[1]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,0,0] \
      #                                   + grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,0,1] \
      #                                   + det_g_x_g_inv[b_idx[0], b_idx[1],0,1] + det_g_x_g_inv[b_idx[0], b_idx[1],1,0]) * Bx ) / sqrt_det_g[b_idx[0], b_idx[1]]
      #                                  #+(grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,1,1] \
      #                                  #  + grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,1,0] \
      #                                  #  + det_g_x_g_inv[b_idx[0], b_idx[1],0,1] + det_g_x_g_inv[b_idx[0], b_idx[1],1,0]) * By ) / sqrt_det_g[b_idx[0], b_idx[1]]
      # Fix to match other cases, TODO fix derivations                                   
      bdry_term[b_idx[0], b_idx[1]] = (grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,0,0] \
                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,0,1] \
                                       + det_g_x_g_inv[b_idx[0], b_idx[1],0,1] + det_g_x_g_inv[b_idx[0], b_idx[1],1,0] \
                                       + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],0,0]) * Bx /  sqrt_det_g[b_idx[0], b_idx[1]]

    elif btype == "bottom":
      # Bx = 0
      By = 2 * nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,1] + 2 * nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,1]
      bdry_term[b_idx[0], b_idx[1]] = (grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,1,1] \
                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,1,0] \
                                       - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],1,1]) * By / sqrt_det_g[b_idx[0], b_idx[1]]

    elif btype == "nottop":
      Bx = 0
      By = 2 * nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,1] + 2 * nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,1]
      ##bdry_term[b_idx[0], b_idx[1]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,0,0] \
      ##                                   + grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,0,1] \
      ##                                   - det_g_x_g_inv[b_idx[0], b_idx[1],0,1] - det_g_x_g_inv[b_idx[0], b_idx[1],1,0]) * Bx \
      ##                                  +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,1,1] \
      ##                                    + grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,1,0] \
      ##                                    - det_g_x_g_inv[b_idx[0], b_idx[1],0,1] - det_g_x_g_inv[b_idx[0], b_idx[1],1,0]) * By ) / sqrt_det_g[b_idx[0], b_idx[1]]
      #bdry_term[b_idx[0], b_idx[1]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,1,1] \
      #                                   + grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,1,0] \
      #                                   - det_g_x_g_inv[b_idx[0], b_idx[1],0,1] - det_g_x_g_inv[b_idx[0], b_idx[1],1,0]) * By ) / sqrt_det_g[b_idx[0], b_idx[1]]
      # Fix to match other cases, TODO fix derivations 
      bdry_term[b_idx[0], b_idx[1]] = (grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,1,1] \
                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,1,0] \
                                       - det_g_x_g_inv[b_idx[0], b_idx[1],0,1] - det_g_x_g_inv[b_idx[0], b_idx[1],1,0] \
                                       - 3 * det_g_x_g_inv[b_idx[0], b_idx[1],1,1]) * By / sqrt_det_g[b_idx[0], b_idx[1]]
    elif btype == "top":
       # Bx = 0
      By = 2 * nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,1] + 2 * nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,1]
      bdry_term[b_idx[0], b_idx[1]] = (grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,1,1] \
                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,1,0] \
                                       + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],1,1]) * By / sqrt_det_g[b_idx[0], b_idx[1]]

    elif btype == "notbottom":
      Bx = 0
      By = 2 * nabla_TT[b_idx[0], b_idx[1],0] * g[b_idx[0], b_idx[1],0,1] + 2 * nabla_TT[b_idx[0], b_idx[1],1] * g[b_idx[0], b_idx[1],1,1]
      ##bdry_term[b_idx[0], b_idx[1]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,0,0] \
      ##                                   + grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,0] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,0,1] \
      ##                                   + det_g_x_g_inv[b_idx[0], b_idx[1],0,1] + det_g_x_g_inv[b_idx[0], b_idx[1],1,0]) * Bx \
      ##                                  +(grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,1,1] \
      ##                                    + grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,1,0] \
      ##                                    + det_g_x_g_inv[b_idx[0], b_idx[1],0,1] + det_g_x_g_inv[b_idx[0], b_idx[1],1,0]) * By ) / sqrt_det_g[b_idx[0], b_idx[1]]
      #bdry_term[b_idx[0], b_idx[1]] = ( (grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,1,1] \
      #                                   + grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,1,0] \
      #                                   + det_g_x_g_inv[b_idx[0], b_idx[1],0,1] + det_g_x_g_inv[b_idx[0], b_idx[1],1,0]) * By ) / sqrt_det_g[b_idx[0], b_idx[1]]
      # Fix to match other cases, TODO fix derivations 
      bdry_term[b_idx[0], b_idx[1]] = (grad_det_g_x_g_inv[b_idx[0], b_idx[1],1,1,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],1,1,1] \
                                       + grad_det_g_x_g_inv[b_idx[0], b_idx[1],0,0,1] + grad_g_inv_x_det_g[b_idx[0], b_idx[1],0,1,0] \
                                       + det_g_x_g_inv[b_idx[0], b_idx[1],0,1] + det_g_x_g_inv[b_idx[0], b_idx[1],1,0] \
                                       + 3 * det_g_x_g_inv[b_idx[0], b_idx[1],1,1]) * By / sqrt_det_g[b_idx[0], b_idx[1]]
    elif btype == "outside":
      # outside mask, skip
      pass

    else:
      # unrecognized type
      print(btype, "unrecognized.  Skipping")

  return(bdry_term)
# end neumann_conditions_rhs

def Ax_orig(x, args):
  # Note, need to fix with args prior to making a LinearOperator
  # args['sqrt_det_g'], args['mask']

  alpha = np.zeros((args['xsz'],args['ysz']))
  alpha[args['mask']>0] = x

  gradX, gradY = compute_alpha_derivs_orig(alpha, args['mask'])
  
  div = np.zeros((args['xsz'],args['ysz']))

  # use np.gradient to calculate cross-derivatives for central point
  gradX_Y, gradY_Y = np.gradient(gradY)
  # use accurate one-sided diffs of central diff for cross-derivatives on the boundary 
  gradY_left = diff.left_diff_2d(gradY)
  gradY_right = diff.right_diff_2d(gradY)
  gradX_bottom = diff.bottom_diff_2d(gradX)
  gradX_top = diff.top_diff_2d(gradX)
   
  # grad_det_g_x_g_inv[i,j, gradient-direction, g-component, g-component] = gradient of sqrt_det_g .* g_inv
  # grad_g_inv_x_det_g[i,j, g-component, g-component, gradient-direction] = gradient of g_inv .* sqrt_det_g
  # det_g_x_g_inv[i,j, g-component, g-component] = sqrt_det_g .* g_inv
    
  for ii in range(args['xsz']):
    for jj in range(args['ysz']):
      if args['mask'][ii,jj]:
        if (not args['mask'][ii-1,jj]) and args['mask'][ii+1,jj] and args['mask'][ii,jj-1] and args['mask'][ii,jj+1]: # left boundary
          #gradxy = gradY_y_shift[ii+1,jj] - gradY_y_shift[ii,jj]    
          #gradxy = ( 8 * alpha[ii+1,jj+1] - alpha[ii+2,jj+2] \
          #         - 8 * alpha[ii,jj+1] + alpha[ii,jj+2] - 8 * alpha[ii+1,jj] + alpha[ii+2,jj] \
          #         + 7 * alpha[ii,jj] ) / 4.0
          #gradxy = (alpha[ii+1,jj+1] - alpha[ii+1,jj] - alpha[ii,jj+1] + alpha[ii,jj])
          gradxy = gradY_left[ii,jj]
          grady2 = (alpha[ii,jj+1] - 2 * alpha[ii,jj] + alpha[ii,jj-1])
          div[ii,jj] = args['det_g_x_g_inv'][ii,jj,0,0] * (4*alpha[ii+1,jj] - 0.5 * alpha[ii+2,jj] - 3.5 * alpha[ii,jj]) \
                     + (args['grad_det_g_x_g_inv'][ii,jj,0,0,1] + args['grad_g_inv_x_det_g'][ii,jj,0,1,0])* gradY[ii,jj] \
                     + (args['grad_det_g_x_g_inv'][ii,jj,1,1,1] + args['grad_g_inv_x_det_g'][ii,jj,1,1,1])* gradY[ii,jj] \
                     + (args['det_g_x_g_inv'][ii,jj,1,0] + args['det_g_x_g_inv'][ii,jj,0,1]) * gradxy \
                     + args['det_g_x_g_inv'][ii,jj,1,1] * grady2
          #if ii == 7 and jj == 51:
          #  print(ii,jj,div[ii,jj],(4*alpha[ii+1,jj] - 0.5 * alpha[ii+2,jj] - 3.5 * alpha[ii,jj]), \
          #        gradY[ii,jj], gradxy, grady2)

        elif (not args['mask'][ii-1,jj]) and args['mask'][ii+1,jj] and (not args['mask'][ii,jj-1]) and args['mask'][ii,jj+1]: # bottomleft boundary
          #gradxy = gradY_y_shift[ii+1,jj] - gradY_y_shift[ii,jj]    
          #gradxy = ( 8 * alpha[ii+1,jj+1] - alpha[ii+2,jj+2] \
          #         - 8 * alpha[ii,jj+1] + alpha[ii,jj+2] - 8 * alpha[ii+1,jj] + alpha[ii+2,jj] \
          #         + 7 * alpha[ii,jj] ) / 4
          #gradxy = (alpha[ii+1,jj+1] - alpha[ii+1,jj] - alpha[ii,jj+1] + alpha[ii,jj])
          gradxy = gradY_left[ii,jj]
          div[ii,jj] = args['det_g_x_g_inv'][ii,jj,0,0] * (4*alpha[ii+1,jj] - 0.5 * alpha[ii+2,jj] - 3.5 * alpha[ii,jj]) \
                     + (args['det_g_x_g_inv'][ii,jj,1,0] + args['det_g_x_g_inv'][ii,jj,0,1]) * gradxy \
                     + args['det_g_x_g_inv'][ii,jj,1,1] * (4*alpha[ii,jj+1] - 0.5 * alpha[ii,jj+2] - 3.5 * alpha[ii,jj])
 
        elif (not args['mask'][ii-1,jj]) and args['mask'][ii+1,jj] and args['mask'][ii,jj-1] and (not args['mask'][ii,jj+1]): # topleft boundary
          #gradxy = gradY_y_shift[ii+1,jj-1] - gradY_y_shift[ii,jj-1]
          #gradxy = (-8 * alpha[ii+1,jj-1] + alpha[ii+2,jj-2] \
          #         + 8 * alpha[ii+1,jj] - alpha[ii+2,jj] + 8 * alpha[ii,jj-1] - alpha[ii,jj-2] \
          #         - 7 * alpha[ii,jj] ) / 4.0
          #gradxy = (-alpha[ii+1,jj-1] + alpha[ii+1,jj] + alpha[ii,jj-1] - alpha[ii,jj])
          gradxy = gradY_left[ii,jj-1]
          div[ii,jj] = args['det_g_x_g_inv'][ii,jj,0,0] * (4*alpha[ii+1,jj] - 0.5 * alpha[ii+2,jj] - 3.5 * alpha[ii,jj]) \
                     + (args['det_g_x_g_inv'][ii,jj,1,0] + args['det_g_x_g_inv'][ii,jj,0,1]) * gradxy \
                     + args['det_g_x_g_inv'][ii,jj,1,1] * (4*alpha[ii,jj-1] - 0.5 * alpha[ii,jj-2] - 3.5 * alpha[ii,jj])
 
        elif (not args['mask'][ii-1,jj]) and args['mask'][ii+1,jj] and (not args['mask'][ii,jj-1]) and (not args['mask'][ii,jj+1]): # notright boundary
          gradx2 = (alpha[ii+2,jj] - 2 * alpha[ii+1,jj] + alpha[ii,jj])
          grady2 = (alpha[ii+1,jj+1] - 2 * alpha[ii+1,jj] + alpha[ii+1,jj-1])
          div[ii,jj] = args['det_g_x_g_inv'][ii,jj,0,0] * gradx2 \
                     + (args['det_g_x_g_inv'][ii,jj,1,0] + args['det_g_x_g_inv'][ii,jj,0,1]) \
                       * (0.5*alpha[ii+1,jj+1]-1.5*alpha[ii,jj]+2*alpha[ii+1,jj]-0.5*alpha[ii+1,jj-1]-0.5*alpha[ii+2,jj]) \
                     + args['det_g_x_g_inv'][ii,jj,1,1] * grady2
                     
        elif args['mask'][ii-1,jj] and (not args['mask'][ii+1,jj]) and args['mask'][ii,jj-1] and args['mask'][ii,jj+1]: # right boundary
          #gradxy = gradY_y_shift[ii,jj] - gradY_y_shift[ii-1,jj]
          #gradxy = ( 8 * alpha[ii-1,jj-1] - alpha[ii-2,jj-2] \
          #         - 8 * alpha[ii,jj-1] + alpha[ii,jj-2] - 8 * alpha[ii-1,jj] + alpha[ii-2,jj] \
          #         + 7 * alpha[ii,jj] ) / 4.0 # don't use this one for right, it doesn't match left correctly
          #gradxy = ( -8 * alpha[ii-1,jj+1] + alpha[ii-2,jj+2] \
          #          + 8 * alpha[ii-1,jj] - alpha[ii-2,jj] + 8 * alpha[ii,jj+1] - alpha[ii,jj+2] \
          #          - 7 * alpha[ii,jj] ) / 4.0
          #gradxy = (alpha[ii-1,jj-1] - alpha[ii-1,jj] - alpha[ii,jj-1] + alpha[ii,jj])
          grady2 = (alpha[ii,jj+1] - 2 * alpha[ii,jj] + alpha[ii,jj-1])
          gradxy = gradY_right[ii,jj]
          div[ii,jj] = args['det_g_x_g_inv'][ii,jj,0,0] * (4*alpha[ii-1,jj] - 0.5 * alpha[ii-2,jj] - 3.5 * alpha[ii,jj]) \
                     + (args['grad_det_g_x_g_inv'][ii,jj,0,0,1] + args['grad_g_inv_x_det_g'][ii,jj,0,1,0])* gradY[ii,jj] \
                     + (args['grad_det_g_x_g_inv'][ii,jj,1,1,1] + args['grad_g_inv_x_det_g'][ii,jj,1,1,1])* gradY[ii,jj] \
                     + (args['det_g_x_g_inv'][ii,jj,1,0] + args['det_g_x_g_inv'][ii,jj,0,1]) * gradxy \
                     + args['det_g_x_g_inv'][ii,jj,1,1] * grady2
          #if ii == 100-7 and jj == 51:
          #  print(ii,jj,div[ii,jj],(4*alpha[ii-1,jj] - 0.5 * alpha[ii-2,jj] - 3.5 * alpha[ii,jj]), \
          #        gradY[ii,jj], gradxy, grady2)
 
        elif args['mask'][ii-1,jj] and (not args['mask'][ii+1,jj]) and (not args['mask'][ii,jj-1]) and args['mask'][ii,jj+1]: # bottomright boundary
          #gradxy = gradY_y_shift[ii,jj] - gradY_y_shift[ii-1,jj]
          #gradxy = ( -8 * alpha[ii-1,jj+1] + alpha[ii-2,jj+2] \
          #          + 8 * alpha[ii-1,jj] - alpha[ii-2,jj] + 8 * alpha[ii,jj+1] - alpha[ii,jj+2] \
          #          - 7 * alpha[ii,jj] ) / 4.0
          #gradxy = (-alpha[ii-1,jj+1] + alpha[ii-1,jj] + alpha[ii,jj+1] - alpha[ii,jj])
          gradxy = gradY_right[ii,jj]
          div[ii,jj] = args['det_g_x_g_inv'][ii,jj,0,0] * (4*alpha[ii-1,jj] - 0.5 * alpha[ii-2,jj] - 3.5 * alpha[ii,jj]) \
                     + (args['det_g_x_g_inv'][ii,jj,1,0] + args['det_g_x_g_inv'][ii,jj,0,1]) * gradxy \
                     + args['det_g_x_g_inv'][ii,jj,1,1] * (4*alpha[ii,jj+1] - 0.5 * alpha[ii,jj+2] - 3.5 * alpha[ii,jj])
                     
        elif args['mask'][ii-1,jj] and (not args['mask'][ii+1,jj]) and args['mask'][ii,jj-1] and (not args['mask'][ii,jj+1]): # topright boundary
          #gradxy = gradY_y_shift[ii,jj-1] - gradY_y_shift[ii-1,jj-1]
          #gradxy = ( 8 * alpha[ii-1,jj-1] - alpha[ii-2,jj-2] \
          #         - 8 * alpha[ii,jj-1] + alpha[ii,jj-2] - 8 * alpha[ii-1,jj] + alpha[ii-2,jj] \
          #         + 7 * alpha[ii,jj] ) / 4.0
          #gradxy = (alpha[ii-1,jj-1] - alpha[ii-1,jj] - alpha[ii,jj-1] + alpha[ii,jj])
          gradxy = gradY_right[ii,jj-1]
          div[ii,jj] = args['det_g_x_g_inv'][ii,jj,0,0] * (4*alpha[ii-1,jj] - 0.5 * alpha[ii-2,jj] - 3.5 * alpha[ii,jj]) \
                     + (args['det_g_x_g_inv'][ii,jj,1,0] + args['det_g_x_g_inv'][ii,jj,0,1]) * gradxy \
                     + args['det_g_x_g_inv'][ii,jj,1,1] * (4*alpha[ii,jj-1] - 0.5 * alpha[ii,jj-2] - 3.5 * alpha[ii,jj])
                     
        elif args['mask'][ii-1,jj] and (not args['mask'][ii+1,jj]) and (not args['mask'][ii,jj-1]) and (not args['mask'][ii,jj+1]): # notleft boundary
          gradx2 = (alpha[ii-2,jj] - 2 * alpha[ii-1,jj] + alpha[ii,jj])
          grady2 = (alpha[ii-1,jj+1] - 2 * alpha[ii-1,jj] + alpha[ii-1,jj-1])
          div[ii,jj] = args['det_g_x_g_inv'][ii,jj,0,0] * gradx2 \
                     + (args['det_g_x_g_inv'][ii,jj,1,0] + args['det_g_x_g_inv'][ii,jj,0,1]) \
                       * (0.5*alpha[ii-1,jj-1]-1.5*alpha[ii,jj]+2*alpha[ii-1,jj]-0.5*alpha[ii-1,jj+1]-0.5*alpha[ii-2,jj]) \
                     + args['det_g_x_g_inv'][ii,jj,1,1] * grady2
 
        elif args['mask'][ii-1,jj] and args['mask'][ii+1,jj] and (not args['mask'][ii,jj-1]) and args['mask'][ii,jj+1]: # bottom boundary
          #gradxy = gradY_y_shift[ii+1,jj] - gradY_y_shift[ii,jj]    
          #gradxy = ( -8 * alpha[ii-1,jj+1] + alpha[ii-2,jj+2] \
          #         + 8 * alpha[ii-1,jj] - alpha[ii-2,jj] + 8 * alpha[ii,jj+1] - alpha[ii,jj+2] \
          #         - 7 * alpha[ii,jj] ) / 4.0
          #gradxy = ( 8 * alpha[ii+1,jj+1] - alpha[ii+2,jj+2] \
          #         - 8 * alpha[ii,jj+1] + alpha[ii,jj+2] - 8 * alpha[ii+1,jj] + alpha[ii+2,jj] \
          #         + 7 * alpha[ii,jj] ) / 4.0
          #gradxy = (alpha[ii-1,jj+1] - 3 * alpha[ii,jj] + 4 * alpha[ii,jj+1] \
          #          - alpha[ii+1,jj+1] - alpha[ii,jj+2] ) / 2.0 \
          #       + gradX[ii,jj] # not working still, guessing we need more accuracy and symmetry about x
          #gradxy = (-alpha[ii-1,jj+1] + alpha[ii-1,jj] + alpha[ii,jj+1] - alpha[ii,jj])
          gradxy = gradX_bottom[ii,jj]
          gradx2 = (alpha[ii+1,jj] - 2 * alpha[ii,jj] + alpha[ii-1,jj])
          div[ii,jj] = args['det_g_x_g_inv'][ii,jj,0,0] * gradx2 \
                     + (args['grad_det_g_x_g_inv'][ii,jj,1,1,0] + args['grad_g_inv_x_det_g'][ii,jj,1,0,1])* gradX[ii,jj] \
                     + (args['grad_det_g_x_g_inv'][ii,jj,0,0,0] + args['grad_g_inv_x_det_g'][ii,jj,0,0,0])* gradX[ii,jj] \
                     + (args['det_g_x_g_inv'][ii,jj,1,0] + args['det_g_x_g_inv'][ii,jj,0,1]) * gradxy \
                     + args['det_g_x_g_inv'][ii,jj,1,1] * (4*alpha[ii,jj+1] - 0.5 * alpha[ii,jj+2] - 3.5 * alpha[ii,jj])
 
        elif (not args['mask'][ii-1,jj]) and (not args['mask'][ii+1,jj]) and (not args['mask'][ii,jj-1]) and args['mask'][ii,jj+1]: # nottop boundary
          gradx2 = (alpha[ii+1,jj+1] - 2 * alpha[ii,jj+1] + alpha[ii-1,jj+1])
          grady2 = (alpha[ii,jj+2] - 2 * alpha[ii,jj+1] + alpha[ii,jj])
          div[ii,jj] = args['det_g_x_g_inv'][ii,jj,0,0] * gradx2 \
                     + (args['det_g_x_g_inv'][ii,jj,1,0] + args['det_g_x_g_inv'][ii,jj,0,1]) \
                       * (0.5*alpha[ii+1,jj+1]-1.5*alpha[ii,jj]+2*alpha[ii,jj+1]-0.5*alpha[ii-1,jj+1]-0.5*alpha[ii,jj+2]) \
                     + args['det_g_x_g_inv'][ii,jj,1,1] * grady2
                     
        elif args['mask'][ii-1,jj] and args['mask'][ii+1,jj] and args['mask'][ii,jj-1] and (not args['mask'][ii,jj+1]): # top boundary
          #gradxy = gradY_y_shift[ii,jj-1] - gradY_y_shift[ii-1,jj-1]
          #gradxy = ( -8 * alpha[ii+1,jj-1] + alpha[ii+2,jj-2] \
          #          + 8 * alpha[ii+1,jj] - alpha[ii+2,jj] + 8 * alpha[ii,jj-1] - alpha[ii,jj-2] \
          #          - 7 * alpha[ii,jj] ) / 4.0
          #gradxy = ( 8 * alpha[ii-1,jj-1] - alpha[ii-2,jj-2] \
          #         - 8 * alpha[ii,jj-1] + alpha[ii,jj-2] - 8 * alpha[ii-1,jj] + alpha[ii-2,jj] \
          #         + 7 * alpha[ii,jj] ) / 4.0
          #gradxy = (alpha[ii-1,jj-1] - 3 * alpha[ii,jj] + 4 * alpha[ii,jj-1] \
          #          - alpha[ii+1,jj-1] - alpha[ii,jj-2] ) / 2.0 \
          #       + gradX[ii,jj] # not working still, guessing we need more accuracy and symmetry about 
          #gradxy = (-alpha[ii+1,jj-1] + alpha[ii+1,jj] + alpha[ii,jj-1] - alpha[ii,jj])
          gradxy = gradX_top[ii,jj]
          gradx2 = (alpha[ii+1,jj] - 2 * alpha[ii,jj] + alpha[ii-1,jj])
          div[ii,jj] = args['det_g_x_g_inv'][ii,jj,0,0] * gradx2 \
                     + (args['grad_det_g_x_g_inv'][ii,jj,1,1,0] + args['grad_g_inv_x_det_g'][ii,jj,1,0,1])* gradX[ii,jj] \
                     + (args['grad_det_g_x_g_inv'][ii,jj,0,0,0] + args['grad_g_inv_x_det_g'][ii,jj,0,0,0])* gradX[ii,jj] \
                     + (args['det_g_x_g_inv'][ii,jj,1,0] + args['det_g_x_g_inv'][ii,jj,0,1]) * gradxy \
                     + args['det_g_x_g_inv'][ii,jj,1,1] * (4*alpha[ii,jj-1] - 0.5 * alpha[ii,jj-2] - 3.5 * alpha[ii,jj])
          #if (ii == 24 or ii == (100-24)) and jj == 85:
          #  print(ii,jj,div[ii,jj], gradx2, \
          #        gradX[ii,jj], gradxy, (4*alpha[ii,jj-1] - 0.5 * alpha[ii,jj-2] - 3.5 * alpha[ii,jj]))
 
        elif (not args['mask'][ii-1,jj]) and (not args['mask'][ii+1,jj]) and args['mask'][ii,jj-1] and (not args['mask'][ii,jj+1]): # notbottom boundary
          gradx2 = (alpha[ii+1,jj-1] - 2 * alpha[ii,jj-1] + alpha[ii-1,jj-1])
          grady2 = (alpha[ii,jj-2] - 2 * alpha[ii,jj-1] + alpha[ii,jj])
          div[ii,jj] = args['det_g_x_g_inv'][ii,jj,0,0] * gradx2 \
                     + (args['det_g_x_g_inv'][ii,jj,1,0] + args['det_g_x_g_inv'][ii,jj,0,1]) \
                       * (0.5*alpha[ii-1,jj-1]-1.5*alpha[ii,jj]+2*alpha[ii,jj-1]-0.5*alpha[ii+1,jj-1]-0.5*alpha[ii,jj-2]) \
                     + args['det_g_x_g_inv'][ii,jj,1,1] * grady2
 
        else:
          # interior point
          gradxy = 0
          gradx2 = (alpha[ii+1,jj] - 2 * alpha[ii,jj] + alpha[ii-1,jj])
          grady2 = (alpha[ii,jj+1] - 2 * alpha[ii,jj] + alpha[ii,jj-1])
          if args['mask'][ii-1,jj-1] and args['mask'][ii+1,jj-1] and args['mask'][ii-1,jj+1] and args['mask'][ii+1,jj+1]:
            # all 4 diagonal neighbors in bounds, so use them all to compute the cross derivative
            #gradxy = (gradY[ii+1,jj] - gradY[ii-1,jj]) / 2
            gradxy = gradX_Y[ii,jj]
          elif args['mask'][ii-1,jj-1] and args['mask'][ii-1,jj+1]:
            # same as for right
            gradxy = gradY_right[ii,jj]
          elif args['mask'][ii+1,jj-1] and args['mask'][ii+1,jj+1]:
            # same as for left
            gradxy = gradY_left[ii,jj]
          elif args['mask'][ii-1,jj+1] and args['mask'][ii+1,jj+1]:
            # same as for bottom
            gradxy = gradX_bottom[ii,jj]
          elif args['mask'][ii-1,jj-1] and args['mask'][ii+1,jj-1]:
            # same as for top
            gradxy = gradX_top[ii,jj]
          else:
            # not sure yet whether we need to worry about these other cases, note if we get one
            #print(ii,jj,args['mask'][ii-1,jj-1],args['mask'][ii-1,jj+1],
            #      args['mask'][ii+1,jj-1],args['mask'][ii+1,jj+1],'unexpected interior point case')
            pass
          #elif args['mask'][ii-1,jj-1]:
            # same as for topright
            #gradxy = gradY_y_shift[ii,jj-1] - gradY_y_shift[ii-1,jj-1]
            #gradxy = (alpha[ii-1,jj-1] - alpha[ii-1,jj] - alpha[ii,jj-1] + alpha[ii,jj])
          #  gradxy = gradY_right[ii,jj-1]
          #elif args['mask'][ii+1,jj-1]:
            # same as for topleft
            #gradxy = gradY_y_shift[ii+1,jj-1] - gradY_y_shift[ii,jj-1]
            #gradxy = (-alpha[ii+1,jj-1] + alpha[ii+1,jj] + alpha[ii,jj-1] - alpha[ii,jj])
          #  gradxy = gradY_left[ii,jj-1]
          #elif args['mask'][ii-1,jj+1]:
            # same as for bottomright
            #gradxy = gradY_y_shift[ii,jj] - gradY_y_shift[ii-1,jj]
            #gradxy = (-alpha[ii-1,jj+1] + alpha[ii-1,jj] + alpha[ii,jj+1] - alpha[ii,jj])
          #  gradxy = gradY_right[ii,jj]
          #elif args['mask'][ii+1,jj+1]:
            # same as for bottomleft
            #gradxy = gradY_y_shift[ii+1,jj] - gradY_y_shift[ii,jj]
            #gradxy = (alpha[ii+1,jj+1] - alpha[ii+1,jj] - alpha[ii,jj+1] + alpha[ii,jj])
          #  gradxy = gradY_left[ii,jj]
          div[ii,jj] = (args['grad_det_g_x_g_inv'][ii,jj,0,0,0] + args['grad_g_inv_x_det_g'][ii,jj,0,0,0])* gradX[ii,jj] \
                     + args['det_g_x_g_inv'][ii,jj,0,0] * gradx2 \
                     + (args['grad_det_g_x_g_inv'][ii,jj,0,0,1] + args['grad_g_inv_x_det_g'][ii,jj,0,1,0])* gradY[ii,jj] \
                     + args['det_g_x_g_inv'][ii,jj,0,1] * gradxy \
                     + (args['grad_det_g_x_g_inv'][ii,jj,1,1,0] + args['grad_g_inv_x_det_g'][ii,jj,1,0,1])* gradX[ii,jj] \
                     + args['det_g_x_g_inv'][ii,jj,1,0] * gradxy \
                     + (args['grad_det_g_x_g_inv'][ii,jj,1,1,1] + args['grad_g_inv_x_det_g'][ii,jj,1,1,1])* gradY[ii,jj] \
                     + args['det_g_x_g_inv'][ii,jj,1,1] * grady2
     
  lhs = div / args['sqrt_det_g']
  lhs[np.isnan(lhs)] = 0
  
  print('Residual:', np.sqrt(np.sum(lhs - args['rhs']) ** 2))
  
  return(lhs[args['mask']>0])
# end Ax_orig
  
def Ax(x, args):
# Note, need to fix with args prior to making a LinearOperator
  # args['sqrt_det_g'], args['mask']

  alpha = np.zeros((args['xsz'],args['ysz']))
  alpha[args['mask']>0] = x

  gradX, gradY = compute_alpha_derivs(alpha, args['bdry_idx'], args['bdry_map'])
  
  div = np.zeros((args['xsz'],args['ysz']))

  gradXX = np.zeros((args['xsz'],args['ysz']))
  gradXY = np.zeros((args['xsz'],args['ysz']))
  #gradYX = np.zeros((args['xsz'],ysz))
  gradYY = np.zeros((args['xsz'],args['ysz']))
  
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
      gradXX[b_idx[0], b_idx[1]] = diff.gradxx_idx_2d(alpha, b_idx)
      gradYY[b_idx[0], b_idx[1]] = diff.gradyy_idx_2d(alpha, b_idx)

      if btype == "interior":
        gradXY[b_idx[0], b_idx[1]] = diff.grady_idx_2d(gradX, b_idx)
        
      elif btype == "interiorleft":
        gradXY[b_idx[0], b_idx[1]] = diff.grady_idx_2d(gradX, b_idx)

      elif btype == "interiorright":
        gradXY[b_idx[0], b_idx[1]] = diff.grady_idx_2d(gradX, b_idx)

      elif btype == "interiorbottom":
        gradXY[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(gradY, b_idx)

      elif btype == "interiortop":
        gradXY[b_idx[0], b_idx[1]] = diff.gradx_idx_2d(gradY, b_idx)

      else:
        # TODO How valid/necessary is this case?
        gradXY[b_idx[0], b_idx[1]] = 0
        

      div[b_idx[0], b_idx[1]] = (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],0,0,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],0,0,0])* gradX[b_idx[0], b_idx[1]] \
                     + args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,0] * gradXX[b_idx[0], b_idx[1]] \
                     + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],0,0,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],0,1,0])* gradY[b_idx[0], b_idx[1]] \
                     + args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,1] * gradXY[b_idx[0], b_idx[1]] \
                     + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],1,1,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],1,0,1])* gradX[b_idx[0], b_idx[1]] \
                     + args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,0] * gradXY[b_idx[0], b_idx[1]] \
                     + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],1,1,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],1,1,1])* gradY[b_idx[0], b_idx[1]] \
                     + args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,1] * gradYY[b_idx[0], b_idx[1]]    
  
    elif btype == "left":   
      gradXY[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(gradY, b_idx)
      gradYY[b_idx[0], b_idx[1]] = diff.gradyy_idx_2d(alpha, b_idx)
      div[b_idx[0], b_idx[1]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,0] * (4*alpha[b_idx[0]+1, b_idx[1]] - 0.5 * alpha[b_idx[0]+2, b_idx[1]] - 3.5 * alpha[b_idx[0], b_idx[1]]) \
                                + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],0,0,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],0,1,0])* gradY[b_idx[0], b_idx[1]] \
                                + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],1,1,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],1,1,1])* gradY[b_idx[0], b_idx[1]] \
                                + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,1]) * gradXY[b_idx[0], b_idx[1]] \
                                + args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,1] * gradYY[b_idx[0], b_idx[1]]
      #if ii == 7 and jj == 51:
      #  print(ii,jj,div[ii,jj],(4*alpha[ii+1,jj] - 0.5 * alpha[ii+2,jj] - 3.5 * alpha[ii,jj]), \
        #        gradY[ii,jj], gradxy, grady2)

    elif btype == "bottomleft":
      gradXY[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(gradY, b_idx)
      div[b_idx[0], b_idx[1]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,0] * (4*alpha[b_idx[0]+1, b_idx[1]] - 0.5 * alpha[b_idx[0]+2, b_idx[1]] - 3.5 * alpha[b_idx[0], b_idx[1]]) \
                                + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,1]) * gradXY[b_idx[0], b_idx[1]] \
                                + args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,1] * (4*alpha[b_idx[0], b_idx[1]+1] - 0.5 * alpha[b_idx[0], b_idx[1]+2] - 3.5 * alpha[b_idx[0], b_idx[1]])
 
    elif  btype == "topleft":
      gradXY[b_idx[0], b_idx[1]] = diff.left_diff_idx_2d(gradY, b_idx)
      div[b_idx[0], b_idx[1]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,0] * (4*alpha[b_idx[0]+1, b_idx[1]] - 0.5 * alpha[b_idx[0]+2, b_idx[1]] - 3.5 * alpha[b_idx[0], b_idx[1]]) \
                               + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,1]) * gradXY[b_idx[0], b_idx[1]] \
                               + args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,1] * (4*alpha[b_idx[0], b_idx[1]-1] - 0.5 * alpha[b_idx[0], b_idx[1]-2] - 3.5 * alpha[b_idx[0], b_idx[1]])
 
    elif btype == "notright":
      #gradXX[b_idx[0], b_idx[1]] = (alpha[b_idx[0]+2, b_idx[1]] - 2 * alpha[b_idx[0]+1, b_idx[1]] + alpha[b_idx[0], b_idx[1]])
      #gradYY[b_idx[0], b_idx[1]] = (alpha[b_idx[0]+1, b_idx[1]+1] - 2 * alpha[b_idx[0]+1, b_idx[1]] + alpha[b_idx[0]+1, b_idx[1]-1])
      gradXX[b_idx[0], b_idx[1]] = (4*alpha[b_idx[0]+1, b_idx[1]] - 0.5 * alpha[b_idx[0]+2, b_idx[1]] - 3.5 * alpha[b_idx[0], b_idx[1]])
      gradYY[b_idx[0], b_idx[1]] = (4*alpha[b_idx[0]+1, b_idx[1]+1] - 0.5 * alpha[b_idx[0]+1, b_idx[1]+2] - 3.5 * alpha[b_idx[0]+1, b_idx[1]])
      div[b_idx[0], b_idx[1]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,0] * gradXX[b_idx[0], b_idx[1]] \
                                + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,1]) \
                                * (0.5*alpha[b_idx[0]+1, b_idx[1]+1]-1.5*alpha[b_idx[0], b_idx[1]]+2*alpha[b_idx[0]+1, b_idx[1]] \
                                   - 0.5*alpha[b_idx[0]+1, b_idx[1]-1]-0.5*alpha[b_idx[0]+2, b_idx[1]]) \
                                + args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,1] * gradYY[b_idx[0], b_idx[1]]
                     
    elif  btype == "right":
      gradYY[b_idx[0], b_idx[1]] = (alpha[b_idx[0], b_idx[1]+1] - 2 * alpha[b_idx[0], b_idx[1]] + alpha[b_idx[0], b_idx[1]-1])
      gradXY[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(gradY, b_idx)
      div[b_idx[0], b_idx[1]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,0] * (4*alpha[b_idx[0]-1, b_idx[1]] - 0.5 * alpha[b_idx[0]-2, b_idx[1]] - 3.5 * alpha[b_idx[0], b_idx[1]]) \
                                + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],0,0,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],0,1,0])* gradY[b_idx[0], b_idx[1]] \
                                + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],1,1,1] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],1,1,1])* gradY[b_idx[0], b_idx[1]] \
                                + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,1]) * gradXY[b_idx[0], b_idx[1]] \
                                + args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,1] * gradYY[b_idx[0], b_idx[1]]
      #if ii == 100-7 and jj == 51:
      #  print(ii,jj,div[ii,jj],(4*alpha[ii-1,jj] - 0.5 * alpha[ii-2,jj] - 3.5 * alpha[ii,jj]), \
        #        gradY[ii,jj], gradxy, grady2)
 
    elif btype == "bottomright":
      gradXY[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(gradY, b_idx)
      div[b_idx[0], b_idx[1]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,0] * (4*alpha[b_idx[0]-1, b_idx[1]] - 0.5 * alpha[b_idx[0]-2, b_idx[1]] - 3.5 * alpha[b_idx[0], b_idx[1]]) \
                                + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,1]) * gradXY[b_idx[0], b_idx[1]] \
                                + args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,1] * (4*alpha[b_idx[0], b_idx[1]+1] - 0.5 * alpha[b_idx[0], b_idx[1]+2] - 3.5 * alpha[b_idx[0], b_idx[1]])
      
    elif btype == "topright":
      gradXY[b_idx[0], b_idx[1]] = diff.right_diff_idx_2d(gradY, b_idx)
      div[b_idx[0], b_idx[1]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,0] * (4*alpha[b_idx[0]-1, b_idx[1]] - 0.5 * alpha[b_idx[0]-2, b_idx[1]] - 3.5 * alpha[b_idx[0], b_idx[1]]) \
                                + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,1]) * gradXY[b_idx[0], b_idx[1]] \
                                + args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,1] * (4*alpha[b_idx[0], b_idx[1]-1] - 0.5 * alpha[b_idx[0], b_idx[1]-2] - 3.5 * alpha[b_idx[0], b_idx[1]])
                     
    elif btype == "notleft":
      #gradXX[b_idx[0], b_idx[1]] = (alpha[b_idx[0]-2, b_idx[1]] - 2 * alpha[b_idx[0]-1, b_idx[1]] + alpha[b_idx[0], b_idx[1]])
      #gradYY[b_idx[0], b_idx[1]] = (alpha[b_idx[0]-1, b_idx[1]+1] - 2 * alpha[b_idx[0]-1, b_idx[1]] + alpha[b_idx[0]-1, b_idx[1]-1])
      gradXX[b_idx[0], b_idx[1]] = (4*alpha[b_idx[0]-1, b_idx[1]] - 0.5 * alpha[b_idx[0]-2, b_idx[1]] - 3.5 * alpha[b_idx[0], b_idx[1]])
      gradYY[b_idx[0], b_idx[1]] = (4*alpha[b_idx[0]-1, b_idx[1]-1] - 0.5 * alpha[b_idx[0]-1, b_idx[1]-2] - 3.5 * alpha[b_idx[0]-1, b_idx[1]])
      div[b_idx[0], b_idx[1]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,0] * gradXX[b_idx[0], b_idx[1]] \
                                + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,1]) \
                                * (0.5*alpha[b_idx[0]-1, b_idx[1]-1]-1.5*alpha[b_idx[0], b_idx[1]]+2*alpha[b_idx[0]-1, b_idx[1]] \
                                   - 0.5*alpha[b_idx[0]-1, b_idx[1]+1]-0.5*alpha[b_idx[0]-2, b_idx[1]]) \
                                + args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,1] * gradYY[b_idx[0], b_idx[1]]
 
    elif btype == "bottom":
      gradXX[b_idx[0], b_idx[1]] = (alpha[b_idx[0]+1, b_idx[1]] - 2 * alpha[b_idx[0], b_idx[1]] + alpha[b_idx[0]-1, b_idx[1]])
      gradXY[b_idx[0], b_idx[1]] = diff.bottom_diff_idx_2d(gradX, b_idx)
      div[b_idx[0], b_idx[1]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,0] * gradXX[b_idx[0], b_idx[1]] \
                                + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],1,1,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],1,0,1])* gradX[b_idx[0], b_idx[1]] \
                                + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],0,0,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],0,0,0])* gradX[b_idx[0], b_idx[1]] \
                                + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,1]) * gradXY[b_idx[0], b_idx[1]] \
                                + args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,1] * (4*alpha[b_idx[0], b_idx[1]+1] - 0.5 * alpha[b_idx[0], b_idx[1]+2] - 3.5 * alpha[b_idx[0], b_idx[1]])
 
    elif  btype == "nottop":
      #gradXX[b_idx[0], b_idx[1]] = (alpha[b_idx[0]+1, b_idx[1]+1] - 2 * alpha[b_idx[0], b_idx[1]+1] + alpha[b_idx[0]-1, b_idx[1]+1])
      #gradYY[b_idx[0], b_idx[1]] = (alpha[b_idx[0], b_idx[1]+2] - 2 * alpha[b_idx[0], b_idx[1]+1] + alpha[b_idx[0], b_idx[1]])
      gradXX[b_idx[0], b_idx[1]] = (4*alpha[b_idx[0]+1, b_idx[1]+1] - 0.5 * alpha[b_idx[0]+2, b_idx[1]+1] - 3.5 * alpha[b_idx[0], b_idx[1]+1])
      gradYY[b_idx[0], b_idx[1]] = (4*alpha[b_idx[0], b_idx[1]+1] - 0.5 * alpha[b_idx[0], b_idx[1]+2] - 3.5 * alpha[b_idx[0], b_idx[1]])
      div[b_idx[0], b_idx[1]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,0] * gradXX[b_idx[0], b_idx[1]] \
                                + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,1]) \
                                * (0.5*alpha[b_idx[0]+1, b_idx[1]+1]-1.5*alpha[b_idx[0], b_idx[1]]+2*alpha[b_idx[0], b_idx[1]+1] \
                                   - 0.5*alpha[b_idx[0]-1, b_idx[1]+1]-0.5*alpha[b_idx[0], b_idx[1]+2]) \
                                + args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,1] * gradYY[b_idx[0], b_idx[1]]
                     
    elif btype == "top":
      gradXX[b_idx[0], b_idx[1]] = (alpha[b_idx[0]+1, b_idx[1]] - 2 * alpha[b_idx[0], b_idx[1]] + alpha[b_idx[0]-1, b_idx[1]])
      gradXY[b_idx[0], b_idx[1]] = diff.top_diff_idx_2d(gradX, b_idx)
      div[b_idx[0], b_idx[1]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,0] * gradXX[b_idx[0], b_idx[1]] \
                                + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],1,1,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],1,0,1])* gradX[b_idx[0], b_idx[1]] \
                                + (args['grad_det_g_x_g_inv'][b_idx[0], b_idx[1],0,0,0] + args['grad_g_inv_x_det_g'][b_idx[0], b_idx[1],0,0,0])* gradX[b_idx[0], b_idx[1]] \
                                + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,1]) * gradXY[b_idx[0], b_idx[1]] \
                                + args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,1] * (4*alpha[b_idx[0], b_idx[1]-1] - 0.5 * alpha[b_idx[0], b_idx[1]-2] - 3.5 * alpha[b_idx[0], b_idx[1]])
          #if (ii == 24 or ii == (100-24)) and jj == 85:
          #  print(ii,jj,div[ii,jj], gradx2, \
          #        gradX[ii,jj], gradxy, (4*alpha[ii,jj-1] - 0.5 * alpha[ii,jj-2] - 3.5 * alpha[ii,jj]))
 
    elif btype == "notbottom":
      #gradXX[b_idx[0], b_idx[1]] = (alpha[b_idx[0]+1, b_idx[1]-1] - 2 * alpha[b_idx[0], b_idx[1]-1] + alpha[b_idx[0]-1, b_idx[1]-1])
      #gradXX[b_idx[0], b_idx[1]] = 0
      #gradYY[b_idx[0], b_idx[1]] = (alpha[b_idx[0], b_idx[1]-2] - 2 * alpha[b_idx[0], b_idx[1]-1] + alpha[b_idx[0], b_idx[1]])
      gradXX[b_idx[0], b_idx[1]] = (4*alpha[b_idx[0]-1, b_idx[1]-1] - 0.5 * alpha[b_idx[0]-2, b_idx[1]-1] - 3.5 * alpha[b_idx[0], b_idx[1]-1])
      gradYY[b_idx[0], b_idx[1]] = (4*alpha[b_idx[0], b_idx[1]-1] - 0.5 * alpha[b_idx[0], b_idx[1]-2] - 3.5 * alpha[b_idx[0], b_idx[1]])
      div[b_idx[0], b_idx[1]] = args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,0] * gradXX[b_idx[0], b_idx[1]] \
                                + (args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,0] + args['det_g_x_g_inv'][b_idx[0], b_idx[1],0,1]) \
                                * (0.5*alpha[b_idx[0]-1, b_idx[1]-1]-1.5*alpha[b_idx[0], b_idx[1]]+2*alpha[b_idx[0], b_idx[1]-1] \
                                   - 0.5*alpha[b_idx[0]+1, b_idx[1]-1]-0.5*alpha[b_idx[0], b_idx[1]-2]) \
                                   + args['det_g_x_g_inv'][b_idx[0], b_idx[1],1,1] * gradYY[b_idx[0], b_idx[1]]

    elif btype == "outside":
      # outside mask, skip
      pass

    else:
      # unrecognized type
      print(btype, "unrecognized.  Skipping")

     
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

def solve_2d(in_tens, in_mask, max_iters, clipped_range=None, thresh_ratio=None, save_intermediate_results = False):
  # This is the main entry point
  # assumes in_tens is the upper-triangular representation
  # clipped_range defaults to [-2,2]
  # thresh_ratio defaults to 1.0

  if clipped_range is None:
    clipped_range = [-2, 2]
  if thresh_ratio is None:
    thresh_ratio = 1.0
  
  xsz = in_mask.shape[0]
  ysz = in_mask.shape[1]

  mask = np.copy(in_mask)
  tens = np.zeros((xsz,ysz,2,2))
  intermed_results = {}
  tot_time = 0

  iso_tens = np.zeros((2,2))
  # TODO find a better scale factor here for these tensors outside the mask
  # want them invertible and not interfering w/ display
  iso_tens[0,0] = 1.0e-4 
  iso_tens[1,1] = 1.0e-4 

  start_time = time.time()
  #####################
  # preprocess inputs #
  #####################
  print("Preprocessing tensors and mask...")
  
  # convert to full 2x2 representation of tensors
  tens[:,:,0,0] = in_tens[:,:,0]
  tens[:,:,0,1] = in_tens[:,:,1]
  tens[:,:,1,0] = in_tens[:,:,1]
  tens[:,:,1,1] = in_tens[:,:,2]

  # precondition and setup boundary info
  precondition_tensors(tens, mask)
  if save_intermediate_results:
    intermed_results['tens'] = tens
    intermed_results['orig_mask'] = mask

  # remove small components via morphological operations
  mo.open_mask(mask)
  
  # Yes, we are calling determine_boundary_2d twice on purpose.
  # This is a hack to handle the case where we remove nondifferentiable boundary elements
  # Then we call a second time to classify the new boundary points
  # TODO figure out a better way...
  bdry_type, bdry_idx, bdry_map = mo.determine_boundary_2d(mask)
  if save_intermediate_results:
    intermed_results['bdry_type1'] = bdry_type
    intermed_results['bdry_idx1'] = bdry_idx
    intermed_results['bdry_map1'] = bdry_map
    intermed_results['diff_mask1'] = mask

  print("Calling determine_boundary_2d a second time...")
  bdry_type, bdry_idx, bdry_map = mo.determine_boundary_2d(mask)

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
  ##################################
  # setup metric (g) and g inverse #
  ##################################
  print("Setting up metric...")

  g_inv = np.copy(tens)
  g_inv[mask == 0] = iso_tens
  
  g = np.linalg.inv(g_inv)
  g[np.isnan(g)] = 0
  g_inv[np.isnan(g_inv)] = 0

  sqrt_det_g = np.sqrt(np.linalg.det(g))
  sqrt_det_g[np.isnan(sqrt_det_g)] = 1
  sqrt_det_g[sqrt_det_g==0] = 1

  if save_intermediate_results:
    intermed_results['g'] = g
    intermed_results['g_inv'] = g_inv
    #intermed_results['g_inv_cond'] = g_inv_cond
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
  
  #gradG, grad_g_inv, grad_sqrt_det_g = compute_g_derivs_orig(g, g_inv, sqrt_det_g, mask)
  gradG, grad_g_inv, grad_sqrt_det_g = compute_g_derivs(g, g_inv, sqrt_det_g, bdry_idx, bdry_map)

  grad_det_g_x_g_inv = np.zeros((xsz,ysz,2,2,2))
  grad_g_inv_x_det_g = np.zeros((xsz,ysz,2,2,2))
  det_g_x_g_inv = np.zeros((xsz,ysz,2,2))
  # grad_det_g_x_g_inv[i,j, gradient-direction, g-component, g-component] = gradient of sqrt_det_g .* g_inv
  # grad_g_inv_x_det_g[i,j, g-component, g-component, gradient-direction] = gradient of g_inv .* sqrt_det_g
  # det_g_x_g_inv[i,j, g-component, g-component] = sqrt_det_g .* g_inv

  for xx in range(2):
    for yy in range(2):
      det_g_x_g_inv[:,:,xx,yy] = sqrt_det_g[:,:] * g_inv[:,:,xx,yy]
      for zz in range(2):
        grad_det_g_x_g_inv[:,:,xx,yy,zz] = grad_sqrt_det_g[:,:,xx] * g_inv[:,:,yy,zz]
        grad_g_inv_x_det_g[:,:,xx,yy,zz] = grad_g_inv[:,:,xx,yy,zz] * sqrt_det_g[:,:]

  if save_intermediate_results:
    intermed_results['grad_g'] = gradG
    intermed_results['grad_g_inv'] = grad_g_inv
    intermed_results['grad_sqrt_det_g'] = grad_sqrt_det_g
    intermed_results['det_g_x_g_inv'] = det_g_x_g_inv
    intermed_results['grad_det_g_x_g_inv'] = grad_det_g_x_g_inv
    intermed_results['grad_g_inv_x_det_g'] = grad_g_inv_x_det_g
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  ########################################################
  # compute principal eigenvector (TDir) of tensor field # 
  ########################################################
  print("Computing principal eigenvector of tensor field...")

  eigenvecs = tensors.eigv(tens)
  v=eigenvecs[:,:,1]
  TDir = riem_vec_norm(eigenvecs[:,:,1], g)

  TDir[np.isnan(TDir)] = 0
  TDir[np.isinf(TDir)] = 0

  if save_intermediate_results:
    intermed_results['TDir'] = TDir
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  ###################################################
  # compute gradients of principal eigenvector TDir # 
  ###################################################
  print("Computing gradients of principal eigenvector...")
  #gradTx_delx, gradTx_dely, gradTy_delx, gradTy_dely = compute_T_derivs_orig(TDir, mask)
  gradTx_delx, gradTx_dely, gradTy_delx, gradTy_dely = compute_T_derivs(TDir, bdry_idx, bdry_map)

  if save_intermediate_results:
    intermed_results['gradTx_delx'] = gradTx_delx
    intermed_results['gradTx_dely'] = gradTx_dely
    intermed_results['gradTy_delx'] = gradTy_delx
    intermed_results['gradTy_dely'] = gradTy_dely
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  ########################################
  # compute first component of nabla_T T # 
  ########################################
  print("Computing first component of nabla_T T...")
  
  first_nabla_TT = np.zeros((xsz,ysz,2))
  first_nabla_TT[:,:,0] = np.multiply(TDir[:,:,0], gradTx_delx[:,:]) + np.multiply(TDir[:,:,1], gradTx_dely[:,:])
  first_nabla_TT[:,:,1] = np.multiply(TDir[:,:,0], gradTy_delx[:,:]) + np.multiply(TDir[:,:,1], gradTy_dely[:,:])

  if save_intermediate_results:
    intermed_results['first_nabla_TT'] = first_nabla_TT
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  #########################################
  # compute second component of nabla_T T # 
  #########################################
  print("Computing second component of nabla_T T...")

  christoffel=np.zeros((xsz,ysz,2,2,2))
  second_nabla_TT = np.zeros((xsz,ysz, 2))
  for k in range(2):
    for p in range(2):
      for q in range(2):
        christoffel[:,:,k,p,q] = 0.5 * g_inv[:,:,k,0]*(gradG[:,:,q,0,p] + gradG[:,:,p,0,q]-gradG[:,:,p,q,0])
        christoffel[:,:,k,p,q] += 0.5 * g_inv[:,:,k,1]*(gradG[:,:,q,1,p] + gradG[:,:,p,1,q]-gradG[:,:,p,q,1])
    #christoffel[k,0,0] = 0.5 * g_inv[ii,jj,k,0]*(gradG[ii,jj,0,0,0])
    #christoffel[k,0,0] += 0.5 * g_inv[ii,jj,k,1]*(2*gradG[ii,jj,0,1,0] - gradG[ii,jj,0,0,1])
    #christoffel[k,0,1] = 0.5 * g_inv[ii,jj,k,0]*(gradG[ii,jj,0,0,1])
    #christoffel[k,0,1] += 0.5 * g_inv[ii,jj,k,1]*(gradG[ii,jj,1,1,0])
    #christoffel[k,1,0] = 0.5 * g_inv[ii,jj,k,0]*(gradG[ii,jj,0,0,1])
    #christoffel[k,1,0] += 0.5 * g_inv[ii,jj,k,1]*(gradG[ii,jj,1,1,0])
    #christoffel[k,1,1] = 0.5 * g_inv[ii,jj,k,0]*(2*gradG[ii,jj,1,0,1] - gradG[ii,jj,1,1,0])
    #christoffel[k,1,1] += 0.5 * g_inv[ii,jj,k,1]*(gradG[ii,jj,1,1,1])
    second_nabla_TT[:,:,k] = christoffel[:,:,k,0,0] * TDir[:,:,0] * TDir[:,:,0]  
    second_nabla_TT[:,:,k] += christoffel[:,:,k,1,0] * TDir[:,:,0] * TDir[:,:,1]
    second_nabla_TT[:,:,k] += christoffel[:,:,k,0,1] * TDir[:,:,1] * TDir[:,:,0]
    second_nabla_TT[:,:,k] += christoffel[:,:,k,1,1] * TDir[:,:,1] * TDir[:,:,1]

  if save_intermediate_results:
    intermed_results['second_nabla_TT'] = second_nabla_TT
    intermed_results['christoffel'] = christoffel
    
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
  sqrt_det_nabla_TT = np.zeros((xsz,ysz,2))
  sqrt_det_nabla_TT[:,:,0] = sqrt_det_g * nabla_TT[:,:,0]
  sqrt_det_nabla_TT[:,:,1] = sqrt_det_g * nabla_TT[:,:,1]

  if save_intermediate_results:
    intermed_results['nabla_TT'] = nabla_TT
    intermed_results['sqrt_det_nabla_TT'] = sqrt_det_nabla_TT
    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")
  start_time = time.time()
  #####################################################
  # compute rhs = 2*divergence (nabla_T T)/sqrt_det_g # 
  #####################################################
  print("Computing rhs = 2*divergence (nabla_T T)/sqrt_det_g...")

  # WARNING! compute_nabla_derivs_orig does NOT adjust for derivatives on boundary!
  #grad_nabla_TT, grad_sqrt_det_nabla_TT = compute_nabla_derivs_orig(nabla_TT, sqrt_det_nabla_TT)
  grad_nabla_TT, grad_sqrt_det_nabla_TT = compute_nabla_derivs(nabla_TT, sqrt_det_nabla_TT, bdry_idx, bdry_map)
  
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
  divergence_rhs = (grad_sqrt_det_nabla_TT[:,:,0,0] + \
                    grad_sqrt_det_nabla_TT[:,:,1,1])

  
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

  #neumann_terms = neumann_conditions_rhs_orig(nabla_TT, g, grad_det_g_x_g_inv, grad_g_inv_x_det_g, det_g_x_g_inv, sqrt_det_g, mask)
  # TODO following is no good, because it includes as boundary points, certain pixels that are not on the boundary.  (Or at least not consistently on boundary on lhs too)
  # At the very least, use Ax_orig w/ neumann_conditions_rhs_orig
  # See whether it works to pair Ax w/ neumann_conditions_rhs
  neumann_terms = neumann_conditions_rhs(nabla_TT, g, grad_det_g_x_g_inv, grad_g_inv_x_det_g, det_g_x_g_inv, sqrt_det_g, bdry_idx, bdry_map)
  rhs -= neumann_terms

  if save_intermediate_results:
    intermed_results['neumann_terms'] = neumann_terms
    intermed_results['rhs'] = rhs
    
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

  alpha = np.zeros((xsz,ysz))
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
  Ax_final_noclip = np.zeros((xsz,ysz))
  Ax_final_noclip[mask > 0] = AxLO(x)
  Ax_final_clip = np.zeros((xsz,ysz))
  Ax_final_clip[mask > 0] = AxLO(clipped_alpha[mask > 0])

  res_img_noclip = np.abs(Ax_final_noclip-rhs)
  res_img_clip = np.abs(Ax_final_clip-rhs)
  res_noclip = np.sqrt(np.sum(res_img_noclip**2))
  res_clip = np.sqrt(np.sum(res_img_clip**2))

  scaled_tensors_noclip = tensors.scale_by_alpha(tens,alpha)
  scaled_ginv = tensors.scale_by_alpha(g_inv, clipped_alpha)
  scaled_tensors = tensors.scale_by_alpha(tens, clipped_alpha)

  threshold_ginv = tensors.threshold_to_input(scaled_ginv, tens, thresh_ratio)
  threshold_tensors = tensors.threshold_to_input(scaled_tensors, tens, thresh_ratio)
  
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
    intermed_results['threshold_ginv'] = threshold_ginv
    intermed_results['threshold_tensors'] = threshold_tensors    
  end_time = time.time()
  seg_time = end_time - start_time
  tot_time += seg_time
  print("... took", seg_time, "seconds")

  print("Total solve time:", tot_time, "seconds")
  return(clipped_alpha, threshold_tensors, mask, intermed_results)
# end solve_2d

def compute_analytic_solution(mask, center_line=35):
  # computes -2 ln r + 2 ln center_line
  xcent = mask.shape[0] / 2
  ycent = mask.shape[1] / 2
  ln_img = np.zeros_like(mask)
  for x in range(mask.shape[0]):
    for y in range(mask.shape[1]):
      if x==xcent and y == ycent:
        pass
      else:
        if mask[x,y]:
          ln_img[x,y] = -2.0*math.log(math.sqrt((x-xcent)**2 + (y-ycent)**2)) + 2.0*math.log(math.sqrt(center_line**2)) # for center line

  return(ln_img)

def solve_analytic_annulus_2d(in_tens, mask, center_line=35, save_intermediate_results = False):
  xsz = mask.shape[0]
  ysz = mask.shape[1]
  
  ln_img = compute_analytic_solution(mask, center_line)
  intermed_results = {}
  if save_intermediate_results:
    intermed_results['ln_img'] = ln_img

  # convert to full 2x2 representation of tensors
  tens[:,:,0,0] = in_tens[:,:,0]
  tens[:,:,0,1] = in_tens[:,:,1]
  tens[:,:,1,0] = in_tens[:,:,1]
  tens[:,:,1,1] = in_tens[:,:,2]

  # precondition and setup boundary info
  precondition_tensors(tens, mask)
  
  if save_intermediate_results:
    intermed_results['tens'] = tens
    intermed_results['orig_mask'] = mask
  
  bdry_type, bdry_idx, bdry_map = mo.determine_boundary_2d(mask)

  if save_intermediate_results:
    intermed_results['bdry_type'] = bdry_type
    intermed_results['bdry_idx'] = bdry_idx
    intermed_results['differentiable_mask'] = mask

  # TODO find a better scale factor here for these tensors outside the mask
  # want them invertible and not interfering w/ display
  iso_tens[0,0] = 1.0 
  iso_tens[1,1] = 1.0 

  g_inv = np.copy(tens)
  g_inv[mask == 0] = iso_tens

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

  gradG, grad_g_inv, grad_sqrt_det_g = compute_g_derivs_orig(g, g_inv, sqrt_det_g, mask)
  #gradG, grad_g_inv, grad_sqrt_det_g = compute_g_derivs(g, g_inv, sqrt_det_g, bdry_idx, bdry_map)

  grad_det_g_x_g_inv = np.zeros((xsz,ysz,2,2,2))
  grad_g_inv_x_det_g = np.zeros((xsz,ysz,2,2,2))
  det_g_x_g_inv = np.zeros((xsz,ysz,2,2))
  # grad_det_g_x_g_inv[i,j, gradient-direction, g-component, g-component] = gradient of sqrt_det_g .* g_inv
  # grad_g_inv_x_det_g[i,j, g-component, g-component, gradient-direction] = gradient of g_inv .* sqrt_det_g
  # det_g_x_g_inv[i,j, g-component, g-component] = sqrt_det_g .* g_inv

  for xx in range(2):
    for yy in range(2):
      det_g_x_g_inv[:,:,xx,yy] = sqrt_det_g[:,:] * g_inv[:,:,xx,yy]
      for zz in range(2):
        grad_det_g_x_g_inv[:,:,xx,yy,zz] = grad_sqrt_det_g[:,:,xx] * g_inv[:,:,yy,zz]
        grad_g_inv_x_det_g[:,:,xx,yy,zz] = grad_g_inv[:,:,xx,yy,zz] * sqrt_det_g[:,:]

  if save_intermediate_results:
    intermed_results['grad_g'] = gradG
    intermed_results['grad_g_inv'] = grad_g_inv
    intermed_results['grad_sqrt_det_g'] = grad_sqrt_det_g
    intermed_results['det_g_x_g_inv'] = det_g_x_g_inv
    intermed_results['grad_det_g_x_g_inv'] = grad_det_g_x_g_inv
    intermed_results['grad_g_inv_x_det_g'] = grad_g_inv_x_det_g

  alpha_gradX, alpha_gradY = compute_alpha_derivs_orig(ln_img, mask)
  #alpha_gradX, alpha_gradY = compute_alpha_derivs(ln_img, bdry_idx, bdry_map)

  if save_intermediate_results:
    intermed_results['alpha_gradX'] = alpha_gradX
    intermed_results['alpha_gradY'] = alpha_gradY

  analytic_grad_alpha = np.zeros((xsz, ysz))
  analytic_grad_alpha[:,:,0] = g_inv[:,:,0,0] * alpha_gradX + g_inv[:,:,0,1] * alpha_gradY
  analytic_grad_alpha[:,:,1] = g_inv[:,:,1,0] * alpha_gradX + g_inv[:,:,1,1] * alpha_gradY

  analytic_div = compute_div_grad_alpha_orig(analytic_grad_alpha, mask)
  #analytic_div = compute_div_grad_alpha(analytic_grad_alpha, bdry_idx, bdry_map)

  if save_intermediate_results:
    intermed_results['analytic_grad_alpha'] = analytic_grad_alpha
    intermed_results['analytic_div'] = analytic_div

  return(analytic_div, intermed_results)
# end solve_analytic_annulus_2d
