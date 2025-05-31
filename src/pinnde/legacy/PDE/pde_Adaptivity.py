from . import pde_Points
import numpy as np
import tensorflow as tf

def SelectAdaptivity(adaptivity, model, pdes, ds_init, ds, N_pde, i, f_bdry, s_bdry):
  if adaptivity[0] == "RAR":
    return AdaptiveRAR(adaptivity[1], adaptivity[2], model, pdes, ds_init, ds, N_pde, i, f_bdry, s_bdry)

def AdaptiveRAR(freq, N_adp, model1, pdes1, ds_init1, ds1, N_pde1, i, f_bdry, s_bdry):
  if ((np.mod(i, freq)==0) and (i != 0)): # controll how often
    print("here")
    X = pde_Points.defineCollocationPoints_2var([f_bdry[0],f_bdry[1]],[s_bdry[0],s_bdry[1]],10000)

    f = model1([X[:,:1], X[:,1:2]])
    err_eq = np.absolute(f)
    for j in range(N_adp): #controll how many
      x_id = np.argmax(err_eq)
      t_point = X[x_id][0]
      x_point = X[x_id][1]
      new_point = [[t_point, x_point]]
      pdes1 = tf.concat([pdes1, new_point], 0)
      err_eq = np.delete(err_eq, x_id, 0)
        # print(err_eq)
        # print(x_id)

      #print(pdes)
    bs_pdes1 = len(pdes1[:,:1])//10

    ds_pde1 = tf.data.Dataset.from_tensor_slices(pdes1)
    ds_pde1 = ds_pde1.cache().shuffle(N_pde1).batch(bs_pdes1)

    ds1 = tf.data.Dataset.zip((ds_pde1, ds_init1))
    ds1 = ds1.prefetch(tf.data.experimental.AUTOTUNE)
    print(len(pdes1))
  return ds1, pdes1