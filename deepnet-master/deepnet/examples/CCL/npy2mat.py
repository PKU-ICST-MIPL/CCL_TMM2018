import scipy.io as sio
import numpy as np

prefix_img='./ff_img/ff_reps/'
prefix_txt='./ff_txt/ff_reps/'

test_image=np.load(prefix_img+"test/hidden3-00001-of-00001.npy")
test_text=np.load(prefix_txt+"test/hidden3-00001-of-00001.npy")

train_image=np.load(prefix_img+"train/hidden3-00001-of-00001.npy")
train_text=np.load(prefix_txt+"train/hidden3-00001-of-00001.npy")

val_image=np.load(prefix_img+"validation/hidden3-00001-of-00001.npy")
val_text=np.load(prefix_txt+"validation/hidden3-00001-of-00001.npy")

sio.savemat('res/test_image_data.mat', {'test_image':test_image})
sio.savemat('res/test_text_data.mat', {'test_text':test_text})

sio.savemat('res/train_image_data.mat', {'train_image':train_image})
sio.savemat('res/train_text_data.mat', {'train_text':train_text})

sio.savemat('res/validation_image_data.mat', {'validation_image':val_image})
sio.savemat('res/validation_text_data.mat', {'validation_text':val_text})

train_img_lab=np.load('feature/train_img_lab_whole.npy')
val_img_lab=np.load('feature/validation_img_lab_whole.npy')
test_img_lab=np.load('feature/test_img_lab_whole.npy')
train_txt_lab=np.load('feature/train_txt_lab_whole.npy')
val_txt_lab=np.load('feature/validation_txt_lab_whole.npy')
test_txt_lab=np.load('feature/test_txt_lab_whole.npy')

sio.savemat('res/train_image_lab.mat',{'train_image_lab':train_img_lab})
sio.savemat('res/validation_image_lab.mat',{'validation_image_lab':val_img_lab})
sio.savemat('res/test_image_lab.mat',{'test_image_lab':test_img_lab})
sio.savemat('res/train_text_lab.mat',{'train_text_lab':train_txt_lab})
sio.savemat('res/validation_text_lab.mat',{'validation_text_lab':val_txt_lab})
sio.savemat('res/test_text_lab.mat',{'test_text_lab':test_txt_lab})
