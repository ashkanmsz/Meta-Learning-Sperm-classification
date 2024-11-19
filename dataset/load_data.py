import numpy as np
import os
from .data_aug import balance
import cv2

def load_data(label='h', balancer=False, image_size=128, train_len=500):
    
    labels_dic = {
        'h': 'head',
        'a': 'acrosome',
        'v': 'vacuole',
        'hv' : 'head-vacuole',
        'ha' : 'head-acrosome',
        'av' : 'acrosome-vacuole',
        'hva' : 'head-vacuole-acrosome'
    }

    dataset_addr = '/content/drive/MyDrive/Colab Notebooks/sperm/dataset/mhsma' 

    ########## Loading y values ############  

    assert label in labels_dic.keys(), 'Label is not correct.'

    labels = labels_dic[label].split('-')

    y_train = np.array([])
    y_valid = np.array([])
    y_test = np.array([])

    for i in range(len(labels)):
      
      y_train_file = os.path.join(dataset_addr, 'y_{}_train.npy'.format(labels[i]))
      y_valid_file = os.path.join(dataset_addr, 'y_{}_valid.npy'.format(labels[i]))
      y_test_file = os.path.join(dataset_addr, 'y_{}_test.npy'.format(labels[i]))

      y_train = np.append(y_train,np.load(y_train_file)[0:train_len])
      y_valid = np.append(y_valid,np.load(y_valid_file))
      y_test = np.append(y_test,np.load(y_test_file))

    if y_train.shape[0] > train_len :
      y_train = change_label(y_train, train_len)
    
    if y_valid.shape[0] > 240 :
      y_valid = change_label(y_valid, 240)

    if y_test.shape[0] > 300:
      y_test = change_label(y_test, 300)

    ########## Loading x values ############  

    x_train_128_file = os.path.join(dataset_addr, 'x_{}_train.npy'.format(str(128)))
    x_valid_128_file = os.path.join(dataset_addr, 'x_{}_valid.npy'.format(str(128)))
    x_test_128_file = os.path.join(dataset_addr, 'x_{}_test.npy'.format(str(128)))

    x_train_64_file = os.path.join(dataset_addr, 'x_{}_train.npy'.format(str(64)))
    x_valid_64_file = os.path.join(dataset_addr, 'x_{}_valid.npy'.format(str(64)))
    x_test_64_file = os.path.join(dataset_addr, 'x_{}_test.npy'.format(str(64)))

    x_train = np.array([])
    x_valid = np.array([])
    x_test = np.array([])

    for i in range(len(labels)):

      if image_size == 128:
        train_set = np.load(x_train_128_file)[0:train_len]
        #train_set = draw_border(train_set, labels[i], image_size)
        
        val_set = np.load(x_valid_128_file)
        #val_set = draw_border(val_set, labels[i], image_size)
        
        test_set = np.load(x_test_128_file)
        #test_set = draw_border(test_set, labels[i], image_size)

        x_train = np.append(x_train, train_set).astype(np.float32)
        x_valid = np.append(x_valid, val_set).astype(np.float32)
        x_test  = np.append(x_test, test_set).astype(np.float32)

      else:
        train_set = np.load(x_train_64_file)[0:train_len]
       # train_set = draw_border(train_set, labels[i], image_size)
        
        val_set = np.load(x_valid_64_file)
        #val_set = draw_border(val_set, labels[i], image_size)
        
        test_set = np.load(x_test_64_file)
        #test_set = draw_border(test_set, labels[i], image_size)

        x_train = np.append(x_train, train_set).astype(np.float32)
        x_valid = np.append(x_valid, val_set).astype(np.float32)
        x_test  = np.append(x_test, test_set).astype(np.float32)

    if image_size == 128:
        x_train = np.reshape(x_train, (len(y_train),128,128))
        x_valid = np.reshape(x_valid, (len(y_valid),128,128))
        x_test = np.reshape(x_test, (len(y_test),128,128))
    
    else:
        x_train = np.reshape(x_train, (len(y_train),64,64))
        x_valid = np.reshape(x_valid, (len(y_valid),64,64))
        x_test = np.reshape(x_test, (len(y_test),64,64))

    ########## Balancer ############  

    if balancer:
        train_balance = balance(x_train, y_train, label, 0)
        val_balance = balance(x_valid, y_valid, label, 0)
        test_balance = balance(x_test, y_test, label, 0)
        
        x_train, y_train = train_balance["x"], train_balance["y"]
        x_valid, y_valid = val_balance["x"], val_balance["y"]
        x_test, y_test = test_balance["x"], test_balance["y"]
        
    return {
            "x_train": x_train,
            "x_val": x_valid,
            "x_test": x_test,
            "y_train": y_train,
            "y_val": y_valid,
            "y_test": y_test
        }

def draw_border(set, label, image_size):
    out_put = np.array([])

    for i in range(len(set)):

      if label == 'head':
        cv2.rectangle(set[i],(30,30), (90,90), (255,255,255), 1)

      elif label == 'acrosome':
        cv2.circle(set[i],(64,64), (30), (255,255,255), 1)

      elif label == 'vacuole':
        cv2.line(set[i], (64,30), (20, 85), (255, 255, 255), 1)
        cv2.line(set[i], (20, 85), (100,85), (255, 255, 255), 1)
        cv2.line(set[i], (64,30), (100,85), (255, 255, 255), 1)

      out_put = np.append(out_put, set[i])
    
    if image_size == 128:
      out_put = np.reshape(out_put, (len(set),128,128))
    else:
      out_put = np.reshape(out_put, (len(set),64,64))
      
    return out_put

# 0,1,0,1 --> 0,1,2,3
def change_label(y, original_len) :
  
    j = len(y)//original_len
    
    for k in range(1,j):
      start = k * original_len
      end = (k+1) * original_len

      for i in range(start, end):
        if y[i] == 0 :
          y[i] = k*2
        else :
          y[i] = k*2+1
      
    return y





