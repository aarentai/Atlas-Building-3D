from lazy_imports import np, ndimage

def is_interior_pt_2d(ii, jj, mask):
  return(( mask[ii-1,jj-1] + mask[ii-1,jj] + mask[ii-1,jj+1] \
         + mask[ii,jj-1] + mask[ii,jj] + mask[ii,jj+1] \
         + mask[ii+1,jj-1] + mask[ii+1,jj] + mask[ii+1,jj+1]) == 9)


def is_interior_pt_3d(ii, jj, kk, mask):
  return(( mask[ii-1,jj-1,kk-1] + mask[ii-1,jj,kk-1] + mask[ii-1,jj+1,kk-1] \
           + mask[ii,jj-1,kk-1] + mask[ii,jj,kk-1] + mask[ii,jj+1,kk-1] \
           + mask[ii+1,jj-1,kk-1] + mask[ii+1,jj,kk-1] + mask[ii+1,jj+1,kk-1] \
           + mask[ii-1,jj-1,kk] + mask[ii-1,jj,kk] + mask[ii-1,jj+1,kk] \
           + mask[ii,jj-1,kk] + mask[ii,jj,kk] + mask[ii,jj+1,kk] \
           + mask[ii+1,jj-1,kk] + mask[ii+1,jj,kk] + mask[ii+1,jj+1,kk] \
           + mask[ii-1,jj-1,kk+1] + mask[ii-1,jj,kk+1] + mask[ii-1,jj+1,kk+1] \
           + mask[ii,jj-1,kk+1] + mask[ii,jj,kk+1] + mask[ii,jj+1,kk+1] \
           + mask[ii+1,jj-1,kk+1] + mask[ii+1,jj,kk+1] + mask[ii+1,jj+1,kk+1]) == 27)

def is_3_in_a_row_3d(ii, jj, kk, mask, axis=0):
  if axis == 0:
    return(mask[ii-1,jj,kk] + mask[ii,jj,kk] + mask[ii+1,jj,kk] == 3)
  elif axis == 1:
    return(mask[ii,jj-1,kk] + mask[ii,jj,kk] + mask[ii,jj+1,kk] == 3)
  else: #axis == 2
    return(mask[ii,jj,kk-1] + mask[ii,jj,kk] + mask[ii,jj,kk+1] == 3)

def determine_boundary_2d(mask):
  # Classify boundary points of mask
  # WARNING!  This method modifies the mask as well, by setting values to 0 for any boundary type
  #  that cannot be differentiated accurately (currently).
  # Returns
  #   bdry_type - a label image labelled with boundary type
  #   bdry_idx - a dictionary whose values contain the indices for pixels with the key's boundary type
  #   bdry_map - a dictionary mapping the boundary type name to its associated value
  #
  #   mask - values are set to 0 for any boundary type we cannot currently differentiate
  xsz = mask.shape[0]
  ysz = mask.shape[1]
  bdry_type = np.zeros_like(mask)
  bdry_idx = {}
  bdry_map = {}
  bdry_map["outside"] = 0
  bdry_map["interior"] = 1
  bdry_map["interiorleft"] = 2
  bdry_map["interiorright"] = 3
  bdry_map["interiorbottom"] = 4
  bdry_map["interiortop"] = 5
  bdry_map["left"] = 6
  bdry_map["right"] = 7
  bdry_map["bottom"] = 8
  bdry_map["top"] = 9
  bdry_map["bottomleft"] = 10
  bdry_map["topleft"] = 11
  bdry_map["bottomright"] = 12
  bdry_map["topright"] = 13
  bdry_map["notleft"] = 14
  bdry_map["notright"] = 15
  bdry_map["notbottom"] = 16
  bdry_map["nottop"] = 17

  for ii in range(xsz):
    for jj in range(ysz):
      if mask[ii,jj]:
        if is_interior_pt_2d(ii, jj, mask):
          # interior point
          bdry_type[ii,jj] = bdry_map["interior"]

        elif (mask[ii-1, jj] and mask[ii+1, jj] and mask[ii, jj-1] and mask[ii, jj+1]):
          # mostly interior, especially for first deriv purposes.
          # counts only as boundary for second derivative cross term calculations d2/dxy etc.
          if mask[ii+1,jj-1] and mask[ii+1,jj+1]:
            # interior left
            bdry_type[ii,jj] = bdry_map["interiorleft"]

          elif mask[ii-1,jj-1] and mask[ii-1,jj+1]:
            # interior right
            bdry_type[ii,jj] = bdry_map["interiorright"]

          elif mask[ii-1,jj+1] and mask[ii+1,jj+1]:
            # interior bottom
            bdry_type[ii,jj] = bdry_map["interiorbottom"]

          elif mask[ii-1,jj-1] and mask[ii+1,jj-1]:
            # interior top
            bdry_type[ii,jj] = bdry_map["interiortop"]

        elif is_interior_pt_2d(ii+1, jj, mask):
          # left
          bdry_type[ii,jj] = bdry_map["left"]

        elif is_interior_pt_2d(ii-1, jj, mask):
          # right
          bdry_type[ii,jj] = bdry_map["right"]

        elif is_interior_pt_2d(ii, jj+1, mask):
          # bottom
          bdry_type[ii,jj] = bdry_map["bottom"]

        elif is_interior_pt_2d(ii, jj-1, mask):
          # top
          bdry_type[ii,jj] = bdry_map["top"]
          
        elif is_interior_pt_2d(ii+1, jj+1, mask):
          # left bottom
          bdry_type[ii,jj] = bdry_map["bottomleft"]

        elif is_interior_pt_2d(ii+1, jj-1, mask):
          # left top
          bdry_type[ii,jj] = bdry_map["topleft"]

        elif is_interior_pt_2d(ii-1, jj+1, mask):
          # right bottom
          bdry_type[ii,jj] = bdry_map["bottomright"]

        elif is_interior_pt_2d(ii-1, jj-1, mask):
          # right top
          bdry_type[ii,jj] = bdry_map["topright"]
          
        elif (mask[ii-1, jj-1] and mask[ii-1, jj+1] and mask[ii-1, jj] and mask[ii-2, jj]):
          # not left
          # TODO decide whether better to mask or not mask these pixels
          # If we set mask to 0 here, then we need to set new boundary pixels accordingly!
          #print("notleft masking pixel", ii, jj)
          bdry_type[ii,jj] = bdry_map["notleft"]
          #mask[ii,jj] = 0

        elif (mask[ii+1, jj-1] and mask[ii+1, jj+1] and mask[ii+1, jj] and mask[ii+2, jj]):
          # not right
          #print("notright masking pixel", ii, jj)
          bdry_type[ii,jj] = bdry_map["notright"]
          #mask[ii,jj] = 0

        elif (mask[ii-1, jj-1] and mask[ii+1, jj-1] and mask[ii, jj-1] and mask[ii, jj-2]):
          # not bottom
          #print("notbottom masking pixel", ii, jj)
          bdry_type[ii,jj] = bdry_map["notbottom"]
          #mask[ii,jj] = 0

        elif (mask[ii-1, jj+1] and mask[ii+1, jj+1] and mask[ii, jj+1] and mask[ii, jj+2]):
          # not top
          #print("nottop masking pixel", ii, jj)
          bdry_type[ii,jj] = bdry_map["nottop"]
          #mask[ii,jj] = 0

        else:
          # can't take derivatives currently
          # Note that with care there are some cases here where we still can take derivatives
          # just not as currently implemented.  For now, mask these regions
          print("masking pixel", ii, jj)
          mask[ii,jj] = 0
          bdry_type[ii,jj] = 0
        
      else:
        bdry_type[ii,jj] = 0

  for ii in range(1,18):
    bdry_idx[ii] = np.where(bdry_type == ii)

  return(bdry_type, bdry_idx, bdry_map)
# end determine_boundary_2d

def determine_boundary_3d(mask):
  # Classify boundary points of mask
  # WARNING!  This method modifies the mask as well, by setting values to 0 for any boundary type
  #  that cannot be differentiated accurately (currently).
  # Returns
  #   bdry_type - a label image labelled with boundary type
  #   bdry_idx - a dictionary whose values contain the indices for pixels with the key's boundary type
  #   bdry_map - a dictionary mapping the boundary type name to its associated value
  #
  #   mask - values are set to 0 for any boundary type we cannot currently differentiate
  xsz = mask.shape[0]
  ysz = mask.shape[1]
  zsz = mask.shape[2]
  bdry_type = np.zeros_like(mask)
  bdry_idx = {}
  bdry_map = {}
  bdry_map["outside"] = 0
  bdry_map["interior"] = 1
  bdry_map["interiorleft"] = 2
  bdry_map["interiorright"] = 3
  bdry_map["interiorbottom"] = 4
  bdry_map["interiortop"] = 5
  # bdry_map["left"] = 6
  # bdry_map["right"] = 7
  # bdry_map["bottom"] = 8
  # bdry_map["top"] = 9
  # bdry_map["bottomleft"] = 10
  # bdry_map["topleft"] = 11
  # bdry_map["bottomright"] = 12
  # bdry_map["topright"] = 13
  # bdry_map["shiftyleft"] = 14
  # bdry_map["shiftyright"] = 15
  # bdry_map["bottomshiftx"] = 16
  # bdry_map["topshiftx"] = 17
  # bdry_map["interiorrear"] = 18
  # bdry_map["interiorfront"] = 19
  # bdry_map["rear"] = 20
  # bdry_map["front"] = 21
  # bdry_map["rearleft"] = 22
  # bdry_map["frontleft"] = 23
  # bdry_map["rearright"] = 24
  # bdry_map["frontright"] = 25
  # bdry_map["rearbottom"] = 26
  # bdry_map["frontbottom"] = 27
  # bdry_map["reartop"] = 28
  # bdry_map["fronttop"] = 29
  # bdry_map["rearshiftx"] = 30
  # bdry_map["frontshiftx"] = 31
  # bdry_map["rearshifty"] = 32
  # bdry_map["frontshifty"] = 33
  # bdry_map["shiftzleft"] = 34
  # bdry_map["shiftzright"] = 35
  # bdry_map["shiftzbottom"] = 36
  # bdry_map["shiftztop"] = 37
  # bdry_map["rearbottomleft"] = 38
  # bdry_map["frontbottomleft"] = 39
  # bdry_map["rearbottomright"] = 40
  # bdry_map["frontbottomright"] = 41
  # bdry_map["reartopleft"] = 42
  # bdry_map["fronttopleft"] = 43
  # bdry_map["reartopright"] = 44
  # bdry_map["fronttopright"] = 45
  # bdry_map["rearshiftyleft"] = 46
  # bdry_map["rearshiftyright"] = 47
  # bdry_map["frontshiftyleft"] = 48
  # bdry_map["frontshiftyright"] = 49
  # bdry_map["shiftzshiftyleft"] = 50
  # bdry_map["shiftzshiftyright"] = 51
  # bdry_map["rearbottomshiftx"] = 52
  # bdry_map["reartopshiftx"] = 53
  # bdry_map["frontbottomshiftx"] = 54
  # bdry_map["fronttopshiftx"] = 55
  # bdry_map["shiftzbottomshiftx"] = 56
  # bdry_map["shiftztopshiftx"] = 57
  # bdry_map["shiftzbottomleft"] = 58
  # bdry_map["shiftzbottomright"] = 59
  # bdry_map["shiftztopleft"] = 60
  # bdry_map["shiftztopright"] = 61
  num_known_bdrys = 6
  
  for ii in range(xsz):
    for jj in range(ysz):
      for kk in range(zsz):
        if mask[ii,jj,kk]:
          if is_interior_pt_3d(ii, jj, kk, mask):
            # interior point
            bdry_type[ii,jj,kk] = bdry_map["interior"]
        
          elif (mask[ii-1, jj, kk] and mask[ii+1, jj, kk]
                and mask[ii, jj-1, kk] and mask[ii, jj+1, kk]
                and mask[ii, jj, kk-1] and mask[ii, jj, kk+1]):
            # mostly interior, especially for first deriv purposes.
            # counts only as boundary for second derivative cross term calculations d2/dxy etc.
            if (mask[ii+1,jj-1,kk] and mask[ii+1,jj+1,kk]
                and mask[ii+1,jj,kk-1] and mask[ii+1,jj,kk+1]):
              # interior left
              bdry_type[ii,jj,kk] = bdry_map["interiorleft"]
        
            elif (mask[ii-1,jj-1,kk] and mask[ii-1,jj+1,kk]
                  and mask[ii-1,jj,kk-1] and mask[ii-1,jj,kk+1]):
              # interior right
              bdry_type[ii,jj,kk] = bdry_map["interiorright"]
        
            elif (mask[ii-1,jj+1,kk] and mask[ii+1,jj+1,kk]
                  and mask[ii,jj+1,kk-1] and mask[ii,jj+1,kk+1]):
              # interior bottom
              bdry_type[ii,jj,kk] = bdry_map["interiorbottom"]
        
            elif (mask[ii-1,jj-1,kk] and mask[ii+1,jj-1,kk]
                  and mask[ii,jj-1,kk-1] and mask[ii,jj-1,kk+1]):
              # interior top
              bdry_type[ii,jj,kk] = bdry_map["interiortop"]

            elif (mask[ii-1,jj,kk+1] and mask[ii+1,jj,kk+1]
                  and mask[ii,jj-1,kk+1] and mask[ii,jj+1,kk+1]):
              # interior rear
              bdry_type[ii,jj,kk] = bdry_map["interiorrear"]
        
            elif (mask[ii-1,jj,kk-1] and mask[ii+1,jj,kk-1]
                  and mask[ii,jj-1,kk-1] and mask[ii,jj+1,kk-1]):
              # interior front
              bdry_type[ii,jj,kk] = bdry_map["interiorfront"]
        
          else:
            # build things up, z to x direction
            bdry_name = ''
            if is_3_in_a_row_3d(ii, jj, kk, mask, 2):
              # interior pt z dir
              bdry_name += ''
            elif is_3_in_a_row_3d(ii, jj, kk+1, mask, 2):
              bdry_name += 'rear'
            elif is_3_in_a_row_3d(ii, jj, kk-1, mask, 2):
              bdry_name += 'front'
            elif (is_3_in_a_row_3d(ii+1, jj, kk, mask, 2) or is_3_in_a_row_3d(ii-1, jj, kk, mask, 2) or
                  is_3_in_a_row_3d(ii, jj+1, kk, mask, 2) or is_3_in_a_row_3d(ii, jj-1, kk, mask, 2)):
              bdry_name += 'shiftz'
            elif (is_3_in_a_row_3d(ii+1, jj, kk+1, mask, 2) or is_3_in_a_row_3d(ii-1, jj, kk+1, mask, 2) or
                  is_3_in_a_row_3d(ii, jj+1, kk+1, mask, 2) or is_3_in_a_row_3d(ii, jj-1, kk+1, mask, 2)):
              bdry_name += 'shiftrear'
            elif (is_3_in_a_row_3d(ii+1, jj, kk-1, mask, 2) or is_3_in_a_row_3d(ii-1, jj, kk-1, mask, 2) or
                  is_3_in_a_row_3d(ii, jj+1, kk-1, mask, 2) or is_3_in_a_row_3d(ii, jj-1, kk-1, mask, 2)):
              bdry_name += 'shiftfront'
            else:
              bdry_name += 'noz'
            if is_3_in_a_row_3d(ii, jj, kk, mask, 1):
              # interior pt y dir
              bdry_name += ''
            elif is_3_in_a_row_3d(ii, jj+1, kk, mask, 1):
              bdry_name += 'bottom'
            elif is_3_in_a_row_3d(ii, jj-1, kk, mask, 1):
              bdry_name += 'top'
            elif (is_3_in_a_row_3d(ii+1, jj, kk, mask, 1) or is_3_in_a_row_3d(ii-1, jj, kk, mask, 1) or
                  is_3_in_a_row_3d(ii, jj, kk+1, mask, 1) or is_3_in_a_row_3d(ii, jj, kk-1, mask, 1)):
              bdry_name += 'shifty'
            elif (is_3_in_a_row_3d(ii+1, jj+1, kk, mask, 1) or is_3_in_a_row_3d(ii-1, jj+1, kk, mask, 1) or
                  is_3_in_a_row_3d(ii, jj+1, kk+1, mask, 1) or is_3_in_a_row_3d(ii, jj+1, kk-1, mask, 1)):
              bdry_name += 'shiftbottom'
            elif (is_3_in_a_row_3d(ii+1, jj-1, kk, mask, 1) or is_3_in_a_row_3d(ii-1, jj-1, kk, mask, 1) or
                  is_3_in_a_row_3d(ii, jj-1, kk+1, mask, 1) or is_3_in_a_row_3d(ii, jj-1, kk-1, mask, 1)):
              bdry_name += 'shifttop'
            else:
              bdry_name += 'noy'
            if is_3_in_a_row_3d(ii, jj, kk, mask, 0):
              # interior pt x dir
              bdry_name += ''
            elif is_3_in_a_row_3d(ii+1, jj, kk, mask, 0):
              bdry_name += 'left'
            elif is_3_in_a_row_3d(ii-1, jj, kk, mask, 0):
              bdry_name += 'right'
            elif (is_3_in_a_row_3d(ii, jj+1, kk, mask, 0) or is_3_in_a_row_3d(ii, jj-1, kk, mask, 0) or
                  is_3_in_a_row_3d(ii, jj, kk+1, mask, 0) or is_3_in_a_row_3d(ii, jj, kk-1, mask, 0)):
              bdry_name += 'shiftx'
            elif (is_3_in_a_row_3d(ii+1, jj+1, kk, mask, 0) or is_3_in_a_row_3d(ii+1, jj-1, kk, mask, 0) or
                  is_3_in_a_row_3d(ii+1, jj, kk+1, mask, 0) or is_3_in_a_row_3d(ii+1, jj, kk-1, mask, 0)):
              bdry_name += 'shiftleft'
            elif (is_3_in_a_row_3d(ii-1, jj+1, kk, mask, 0) or is_3_in_a_row_3d(ii-1, jj-1, kk, mask, 0) or
                  is_3_in_a_row_3d(ii-1, jj, kk+1, mask, 0) or is_3_in_a_row_3d(ii-1, jj, kk-1, mask, 0)):
              bdry_name += 'shiftright'
            else:
              bdry_name += 'nox'

            if 'no' in bdry_name or 'shift' in bdry_name:
              # can't take derivatives currently
              # Note that with care there are some cases here where we still can take derivatives
              # just not as currently implemented.  For now, mask these regions
              print("masking pixel", ii, jj, kk,"for boundary type",bdry_name)
              mask[ii,jj,kk] = 0
              bdry_type[ii,jj,kk] = 0
            else:
              if bdry_name not in bdry_map:
                bdry_map[bdry_name] = num_known_bdrys
                num_known_bdrys += 1
              bdry_type[ii,jj,kk] = bdry_map[bdry_name]  
        
        else:
          bdry_type[ii,jj,kk] = 0


  # for ii in range(xsz):
  #   for jj in range(ysz):
  #     for kk in range(zsz):
  #       if mask[ii,jj,kk]:
  #         if is_interior_pt_3d(ii, jj, kk, mask):
  #           # interior point
  #           bdry_type[ii,jj,kk] = bdry_map["interior"]
        
  #         elif (mask[ii-1, jj, kk] and mask[ii+1, jj, kk]
  #               and mask[ii, jj-1, kk] and mask[ii, jj+1, kk]
  #               and mask[ii, jj, kk-1] and mask[ii, jj, kk+1]):
  #           # mostly interior, especially for first deriv purposes.
  #           # counts only as boundary for second derivative cross term calculations d2/dxy etc.
  #           if (mask[ii+1,jj-1,kk] and mask[ii+1,jj+1,kk]
  #               and mask[ii+1,jj,kk-1] and mask[ii+1,jj,kk+1]):
  #             # interior left
  #             bdry_type[ii,jj,kk] = bdry_map["interiorleft"]
        
  #           elif (mask[ii-1,jj-1,kk] and mask[ii-1,jj+1,kk]
  #                 and mask[ii-1,jj,kk-1] and mask[ii-1,jj,kk+1]):
  #             # interior right
  #             bdry_type[ii,jj,kk] = bdry_map["interiorright"]
        
  #           elif (mask[ii-1,jj+1,kk] and mask[ii+1,jj+1,kk]
  #                 and mask[ii,jj+1,kk-1] and mask[ii,jj+1,kk+1]):
  #             # interior bottom
  #             bdry_type[ii,jj,kk] = bdry_map["interiorbottom"]
        
  #           elif (mask[ii-1,jj-1,kk] and mask[ii+1,jj-1,kk]
  #                 and mask[ii,jj-1,kk-1] and mask[ii,jj-1,kk+1]):
  #             # interior top
  #             bdry_type[ii,jj,kk] = bdry_map["interiortop"]

  #           elif (mask[ii-1,jj,kk+1] and mask[ii+1,jj,kk+1]
  #                 and mask[ii,jj-1,kk+1] and mask[ii,jj+1,kk+1]):
  #             # interior rear
  #             bdry_type[ii,jj,kk] = bdry_map["interiorrear"]
        
  #           elif (mask[ii-1,jj,kk-1] and mask[ii+1,jj,kk-1]
  #                 and mask[ii,jj-1,kk-1] and mask[ii,jj+1,kk-1]):
  #             # interior front
  #             bdry_type[ii,jj,kk] = bdry_map["interiorfront"]
        
  #         elif is_interior_pt_3d(ii+1, jj, kk, mask):
  #           # left
  #           bdry_type[ii,jj,kk] = bdry_map["left"]
        
  #         elif is_interior_pt_3d(ii-1, jj, kk, mask):
  #           # right
  #           bdry_type[ii,jj,kk] = bdry_map["right"]
        
  #         elif is_interior_pt_3d(ii, jj+1, kk, mask):
  #           # bottom
  #           bdry_type[ii,jj,kk] = bdry_map["bottom"]
        
  #         elif is_interior_pt_3d(ii, jj-1, kk, mask):
  #           # top
  #           bdry_type[ii,jj,kk] = bdry_map["top"]

  #         elif is_interior_pt_3d(ii, jj, kk+1, mask):
  #           # bottom
  #           bdry_type[ii,jj,kk] = bdry_map["rear"]
        
  #         elif is_interior_pt_3d(ii, jj, kk-1, mask):
  #           # top
  #           bdry_type[ii,jj,kk] = bdry_map["front"]
            
  #         elif is_interior_pt_3d(ii+1, jj+1, kk, mask):
  #           # left bottom
  #           bdry_type[ii,jj,kk] = bdry_map["bottomleft"]
        
  #         elif is_interior_pt_3d(ii+1, jj-1, kk, mask):
  #           # left top
  #           bdry_type[ii,jj,kk] = bdry_map["topleft"]
        
  #         elif is_interior_pt_3d(ii-1, jj+1, kk, mask):
  #           # right bottom
  #           bdry_type[ii,jj,kk] = bdry_map["bottomright"]
        
  #         elif is_interior_pt_3d(ii-1, jj-1, kk, mask):
  #           # right top
  #           bdry_type[ii,jj,kk] = bdry_map["topright"]
            
  #         elif is_interior_pt_3d(ii+1, jj, kk+1, mask):
  #           # left rear
  #           bdry_type[ii,jj,kk] = bdry_map["rearleft"]

  #         elif is_interior_pt_3d(ii+1, jj, kk-1, mask):
  #           # left front
  #           bdry_type[ii,jj,kk] = bdry_map["frontleft"]
        
  #         elif is_interior_pt_3d(ii-1, jj, kk+1, mask):
  #           # right rear
  #           bdry_type[ii,jj,kk] = bdry_map["rearright"]
        
  #         elif is_interior_pt_3d(ii-1, jj, kk-1, mask):
  #           # right front
  #           bdry_type[ii,jj,kk] = bdry_map["frontright"]

  #         elif is_interior_pt_3d(ii, jj+1, kk+1, mask):
  #           # bottom rear
  #           bdry_type[ii,jj,kk] = bdry_map["rearbottom"]

  #         elif is_interior_pt_3d(ii, jj+1, kk-1, mask):
  #           # bottom front
  #           bdry_type[ii,jj,kk] = bdry_map["frontbottom"]
        
  #         elif is_interior_pt_3d(ii, jj-1, kk+1, mask):
  #           # top rear
  #           bdry_type[ii,jj,kk] = bdry_map["reartop"]
        
  #         elif is_interior_pt_3d(ii, jj-1, kk-1, mask):
  #           # top front
  #           bdry_type[ii,jj,kk] = bdry_map["fronttop"]
  #         elif is_interior_pt_3d(ii+1, jj+1, kk+1, mask):
  #           # left bottom rear
  #           bdry_type[ii,jj,kk] = bdry_map["rearbottomleft"]

  #         elif is_interior_pt_3d(ii+1, jj+1, kk-1, mask):
  #           # left bottom front
  #           bdry_type[ii,jj,kk] = bdry_map["frontbottomleft"]
        
  #         elif is_interior_pt_3d(ii-1, jj+1, kk+1, mask):
  #           # right bottom rear
  #           bdry_type[ii,jj,kk] = bdry_map["rearbottomright"]
        
  #         elif is_interior_pt_3d(ii-1, jj+1, kk-1, mask):
  #           # right bottom front
  #           bdry_type[ii,jj,kk] = bdry_map["frontbottomright"]

  #         elif is_interior_pt_3d(ii+1, jj-1, kk+1, mask):
  #           # left top rear
  #           bdry_type[ii,jj,kk] = bdry_map["reartopleft"]

  #         elif is_interior_pt_3d(ii+1, jj-1, kk-1, mask):
  #           # left top front
  #           bdry_type[ii,jj,kk] = bdry_map["fronttopleft"]
        
  #         elif is_interior_pt_3d(ii-1, jj-1, kk+1, mask):
  #           # right top rear
  #           bdry_type[ii,jj,kk] = bdry_map["reartopright"]
        
  #         elif is_interior_pt_3d(ii-1, jj-1, kk-1, mask):
  #           # right top front
  #           bdry_type[ii,jj,kk] = bdry_map["fronttopright"]
            
  #         elif (mask[ii-1, jj-1, kk] and mask[ii-1, jj+1, kk]
  #               and mask[ii-1, jj, kk-1] and mask[ii-1, jj, kk+1]
  #               and mask[ii-1, jj, kk] and mask[ii-2, jj, kk]):
  #           # not left
  #           # TODO decide whether better to mask or not mask these pixels
  #           # If we set mask to 0 here, then we need to set new boundary pixels accordingly!
  #           #print("notleft masking pixel", ii, jj, kk)
  #           bdry_type[ii,jj,kk] = bdry_map["notleft"]
  #           #mask[ii,jj,kk] = 0
        
  #         elif (mask[ii+1, jj-1, kk] and mask[ii+1, jj+1, kk]
  #               and mask[ii+1, jj, kk-1] and mask[ii+1, jj, kk+1]
  #               and mask[ii+1, jj, kk] and mask[ii+2, jj, kk]):
  #           # not right
  #           #print("notright masking pixel", ii, jj, kk)
  #           bdry_type[ii,jj,kk] = bdry_map["notright"]
  #           #mask[ii,jj,kk] = 0
        
  #         elif (mask[ii-1, jj-1, kk] and mask[ii+1, jj-1, kk]
  #               and mask[ii, jj-1, kk-1] and mask[ii, jj-1, kk+1]
  #               and mask[ii, jj-1, kk] and mask[ii, jj-2, kk]):
  #           # not bottom
  #           #print("notbottom masking pixel", ii, jj, kk)
  #           bdry_type[ii,jj,kk] = bdry_map["notbottom"]
  #           #mask[ii,jj,kk] = 0
        
  #         elif (mask[ii-1, jj+1, kk] and mask[ii+1, jj+1, kk]
  #               and mask[ii, jj+1, kk-1] and mask[ii, jj+1, kk+1]
  #               and mask[ii, jj+1, kk] and mask[ii, jj+2, kk]):
  #           # not top
  #           #print("nottop masking pixel", ii, jj, kk)
  #           bdry_type[ii,jj,kk] = bdry_map["nottop"]
  #           #mask[ii,jj,kk] = 0

  #         elif (mask[ii-1, jj, kk-1] and mask[ii+1, jj, kk-1]
  #               and mask[ii, jj-1, kk-1] and mask[ii, jj+1, kk-1]
  #               and mask[ii, jj, kk-1] and mask[ii, jj, kk-2]):
  #           # not rear
  #           #print("notrear masking pixel", ii, jj, kk)
  #           bdry_type[ii,jj,kk] = bdry_map["notrear"]
  #           #mask[ii,jj,kk] = 0
        
  #         elif (mask[ii-1, jj, kk+1] and mask[ii+1, jj, kk+1]
  #               and mask[ii, jj-1, kk+1] and mask[ii, jj+1, kk+1]
  #               and mask[ii, jj, kk+1] and mask[ii, jj, kk+2]):
  #           # not front
  #           #print("notfront masking pixel", ii, jj, kk)
  #           bdry_type[ii,jj,kk] = bdry_map["notfront"]
  #           #mask[ii,jj,kk] = 0
        
  #         else:
  #           # can't take derivatives currently
  #           # Note that with care there are some cases here where we still can take derivatives
  #           # just not as currently implemented.  For now, mask these regions
  #           print("masking pixel", ii, jj, kk)
  #           mask[ii,jj,kk] = 0
  #           bdry_type[ii,jj,kk] = 0
          
  #       else:
  #         bdry_type[ii,jj,kk] = 0

  for ii in range(1,num_known_bdrys):
    bdry_idx[ii] = np.where(bdry_type == ii)

  return(bdry_type, bdry_idx, bdry_map)
# end determine_boundary_3d


def open_mask_2d(mask):
  # get rid of small components
  opened = ndimage.binary_opening(mask, structure=np.ones((2, 2)))
  eroded = ndimage.binary_erosion(mask, structure=np.ones((6, 6)))
  ndimage.binary_propagation(eroded, mask=opened, output=mask)
# end open_mask_2d

def open_mask_3d(mask):
  # get rid of small components
  opened = ndimage.binary_opening(mask, structure=np.ones((2, 2, 2)))
  eroded = ndimage.binary_erosion(mask, structure=np.ones((6, 6, 6)))
  ndimage.binary_propagation(eroded, mask=opened, output=mask)
# end open_mask_3d
