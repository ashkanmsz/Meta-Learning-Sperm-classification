from imblearn.over_sampling import SMOTE
import collections

def balance(images, targets, label, random_state):
  
  counter = collections.Counter(targets)
  sampling_strategy_dic = {1: counter[0]}
  if len(label) == 2:
    sampling_strategy_dic = {1: counter[0], 3: counter[2]}
  elif len(label) == 3:
    sampling_strategy_dic = {1: counter[0], 3: counter[2], 5: counter[4]}
    
  over = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy_dic)
  size = images.shape[0]
  height = images.shape[1]
  X_res, Y_res = over.fit_resample(images.reshape((size, (height**2))), targets)
  data = {"x": None, "y": None}
  data["x"] = X_res.reshape((X_res.shape[0],height, height))
  data["y"] = Y_res
  return data