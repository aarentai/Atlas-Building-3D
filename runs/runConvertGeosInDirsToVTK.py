import gzip
import _pickle as pickle
import vtk
import os
import platform
import traceback
import whitematteranalysis as wma
import numpy as np


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


def do_conversion(geodir, postfix, spacing, origin):
  try:
    minFiberLength = 1 #40
    maxFiberLength = None
    retainData = False # matches default from wm_preprocess_all.py
    #numberOfFibers = 1000000 # convert all points
    
    fs = os.listdir(geodir)
    ffs = [f for f in fs if (f[-len(postfix):] == postfix)]
    print('Converting geos in directory:', geodir, 'from files:', ffs)
    
    # correct to world spacing since processing was all done in voxel space [1,1,1]
    # Note, do this for comparison to UKF tractography
    # Make sure to adjust tensor images etc back to this spacing when displaying
    # with these paths
    # Also correct origin to line up with UKF tractography (0,0,0) should be center of brain
    
    for fname in ffs:
      try:  
        with gzip.open(geodir+fname,'rb') as f:
          geos = pickle.load(f)
      except Exception as e:
        print("Error,", e, "while reading file", geodir+fname, ". Moving on to next file")
        continue
      vtkp = vtk.vtkPoints()
      vtkc = vtk.vtkCellArray()
      vtkc.InitTraversal()
      

      estim_num_cells = len(geos) * len(geos[0][0])
      estim_max_pts_per_cell = len(geos[0][0][0])
      estim_num_pts = estim_num_cells * estim_max_pts_per_cell
      vtkp.Allocate(estim_num_pts)
      print("Preallocating", estim_num_pts, "points.")

      for b in range(len(geos)):
        for p in range(len(geos[b][0])):
          ids = vtk.vtkIdList()
          # TODO, leaving off all last points in path to avoid false connection to 0,0,0
          # Better option would be to detect that case and only toss sometimes
          #prev_num_pts = vtkp.GetNumberOfPoints()
          #vid = prev_num_pts
          #if len(geos[b][0][p]) > 1:
          #  cur_num_pts = prev_num_pts + len(geos[b][0][p])-1
          #  vtkp.SetNumberOfPoints(cur_num_pts)
          for idx in range(len(geos[b][0][p])-1):
            if np.abs(geos[b][0][p][idx] - geos[b][0][p][idx+1]) > 1:
              print('Gap found for', fname, 'batch', b, 'geodesic', p, 'pt', idx)
            vid=vtkp.InsertNextPoint(geos[b][0][p][idx]*spacing[0]+origin[0], geos[b][1][p][idx]*spacing[1]+origin[1], geos[b][2][p][idx]*spacing[2]+origin[2])
            ids.InsertNextId(vid)
          # end for each point

          if ids.GetNumberOfIds() > 1:
            vtkc.InsertNextCell(ids)
          #vtkc.InsertNextCell(ids)
        # end for each path in batch
      # end for each batch
    
      print('Created', vtkp.GetNumberOfPoints(), 'points and',
            vtkc.GetNumberOfCells(), 'cells')
    
      pd = vtk.vtkPolyData()
      pd.SetPoints(vtkp)
      pd.SetLines(vtkc)
    
      print('Done creating polydata')

      outfname = geodir + fname[:-len(postfix)] + '.vtp'
      print("Writing data to file", outfname, "...")
      try:
        wma.io.write_polydata(pd, outfname)
        print("Wrote output", outfname)
      except Exception as err:
        print("Caught error", err, "while writing", outfname)
        print(traceback.format_exc())

    
  except Exception as err:
    print("Caught error", err, "while converting geos in ", geodir)
    print(traceback.format_exc())
      
# end do_conversion
    
def collect_result(result):
  # Right now, empty list expected.
  print('collected result')


if __name__ == "__main__":
  print('DO NOT RUN this code using VTK 9.0 if want to read fibers in with Slicer 4.11, use VTK 8.* instead')
  kc_atlasname = 'Ball_met_img_rigid_6subj'
  out_tract_dir = f'/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/MELBAResults/{kc_atlasname}_and_subj_tracts/'

  geodirs = []
  geodirs.append(f'{out_tract_dir}atlas_tracts/')
  geodirs.append(f'{out_tract_dir}dtitk_atlas_tracts/')
  geodirs.append(f'{out_tract_dir}subj_tracts/')
  geodirs.append(f'{out_tract_dir}subj_tracts_deformed_to_atlas_space/')
  geodirs.append(f'{out_tract_dir}subj_tracts_computed_in_atlas_space/')
  geodirs.append(f'{out_tract_dir}dtitk_subj_tracts/')
  geodirs.append(f'{out_tract_dir}subj_tracts_deformed_to_dtitk_atlas_space/')
  geodirs.append(f'{out_tract_dir}subj_tracts_computed_in_dtitk_atlas_space/')

  postfix = '.pkl.gz'
  # correct to world spacing since processing was all done in voxel space [1,1,1]
  # Note, do this for comparison to UKF tractography
  # Make sure to adjust tensor images etc back to this spacing when displaying
  # with these paths
  # Also correct origin to line up with UKF tractography (0,0,0) should be center of brain
  # TODO Confirm that this spacing and origin is appropriate for all subjects or read in from header appropriately
  spacing = [1.25,1.25,1.25]
  origin = [-90,-90.25,-72]

  host = platform.node()
  if ('ardashir' in host) or ('lakota' in host) or ('kourosh' in host):
    pool = multiprocessing.Pool(6)
  elif 'beast' in host:
    pool = multiprocessing.Pool(1) # split into 4 batches to avoid hitting swap space
  else:
    print('Unknown host,', host, ' defaulting to 7 processes.  Increase if on capable host')
    pool = multiprocessing.Pool(1) # split into 4 batches to avoid hitting swap space

  ars = []
  for geodir in geodirs:
    #do_conversion(geodir, prefix, postfix, spacing, origin)
    ar = pool.apply_async(do_conversion, args=(geodir, postfix, spacing, origin), callback=collect_result)
    ars.append(ar)

    print("All tasks launched, waiting for completion")
    
  for ar in ars:
    ar.wait()

  print("All waits returned, closing and joining")
  pool.close()
  pool.join()
  

  
