import scipy.io as sio
import numpy as np
import h5py

prefix='./feature/'

# labels start from zero
data=sio.loadmat(prefix+'Pascal_fea_whole.mat')
train_img_data_whole = data['I_tr']
val_img_data_whole = data['I_va']
test_img_data_whole = data['I_te']
train_img_lab_whole = data['I_tr_lab']-1
val_img_lab_whole = data['I_va_lab']-1
test_img_lab_whole = data['I_te_lab']-1
train_txt_data_whole = data['T_tr']
val_txt_data_whole = data['T_va']
test_txt_data_whole = data['T_te']
train_txt_lab_whole = data['T_tr_lab']-1
val_txt_lab_whole = data['T_va_lab']-1
test_txt_lab_whole = data['T_te_lab']-1
np.save(prefix+'train_img_data_whole.npy', train_img_data_whole)
np.save(prefix+'validation_img_data_whole.npy', val_img_data_whole)
np.save(prefix+'test_img_data_whole.npy', test_img_data_whole)
np.save(prefix+'train_img_lab_whole.npy', train_img_lab_whole)
np.save(prefix+'validation_img_lab_whole.npy', val_img_lab_whole)
np.save(prefix+'test_img_lab_whole.npy', test_img_lab_whole)
np.save(prefix+'train_txt_data_whole.npy', train_txt_data_whole)
np.save(prefix+'validation_txt_data_whole.npy', val_txt_data_whole)
np.save(prefix+'test_txt_data_whole.npy', test_txt_data_whole)
np.save(prefix+'train_txt_lab_whole.npy', train_txt_lab_whole)
np.save(prefix+'validation_txt_lab_whole.npy', val_txt_lab_whole)
np.save(prefix+'test_txt_lab_whole.npy', test_txt_lab_whole)

data=sio.loadmat(prefix+'Pascal_fea_part.mat')
train_img_data_part = data['I_tr']
val_img_data_part = data['I_va']
test_img_data_part = data['I_te']
train_img_lab_part = data['I_tr_lab']-1
val_img_lab_part = data['I_va_lab']-1
test_img_lab_part = data['I_te_lab']-1
train_txt_data_part = data['T_tr']
val_txt_data_part = data['T_va']
test_txt_data_part = data['T_te']
train_txt_lab_part = data['T_tr_lab']-1
val_txt_lab_part = data['T_va_lab']-1
test_txt_lab_part = data['T_te_lab']-1
np.save(prefix+'train_img_data_part.npy', train_img_data_part)
np.save(prefix+'validation_img_data_part.npy', val_img_data_part)
np.save(prefix+'test_img_data_part.npy', test_img_data_part)
np.save(prefix+'train_img_lab_part.npy', train_img_lab_part)
np.save(prefix+'validation_img_lab_part.npy', val_img_lab_part)
np.save(prefix+'test_img_lab_part.npy', test_img_lab_part)
np.save(prefix+'train_txt_data_part.npy', train_txt_data_part)
np.save(prefix+'validation_txt_data_part.npy', val_txt_data_part)
np.save(prefix+'test_txt_data_part.npy', test_txt_data_part)
np.save(prefix+'train_txt_lab_part.npy', train_txt_lab_part)
np.save(prefix+'validation_txt_lab_part.npy', val_txt_lab_part)
np.save(prefix+'test_txt_lab_part.npy', test_txt_lab_part)


