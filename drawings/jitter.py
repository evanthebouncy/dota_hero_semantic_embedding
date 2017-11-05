import numpy as np

# apply a small jitter so things don't completely overlap
def jitter(data_dict, ratio_sep_dist = 1.0 / 36):
  def get_max_length(data_dict):
    x_min = 99999
    x_max = -99999
    y_min = 99999
    y_max = -99999
    for name in data_dict:
      x, y = data_dict[name][0], data_dict[name][1]
      x_min = min(x_min, x)
      x_max = max(x_max, x)
      y_min = min(y_min, y)
      y_may = max(y_max, y)
    return max(x_max-x_min, y_max-y_min)

  # set the desired jitter distance
  max_length = get_max_length(data_dict)
  sep_dist_target = ratio_sep_dist * max_length
  
  # jitter the coordinates so they don't overlap 
  def _jitter_mutate(data_dict):
    for name1 in data_dict:
      for name2 in data_dict:
        if name1 != name2:
          xy1 = data_dict[name1]
          xy2 = data_dict[name2]
          sep_dist = np.linalg.norm(xy1-xy2)
          # if the seperation distance is too small, move them apart
          if sep_dist < sep_dist_target:
            xy11 = xy1 + sep_dist_target * (xy1 - xy2) / sep_dist / 4
            xy22 = xy2 + sep_dist_target * (xy2 - xy1) / sep_dist / 4
            print "separating ", name1, name2
            data_dict[name1] = xy11
            data_dict[name2] = xy22
            return False
    return True

  # run infinite loop until jitter mutate pull stuff apart
  while not _jitter_mutate(data_dict):
    True

  return data_dict
