import numpy as np
from numpy.matlib import repmat

def znorm(inMat, refMat):
  col=inMat.shape[0]
  row=inMat.shape[1]
  mean_val=np.mean(refMat, axis=0)
  std_val=np.std(refMat, axis=0)
  mean_val=repmat(mean_val, col, 1)
  std_val=repmat(std_val, col, 1)
  x = np.argwhere(std_val==0)
  for y in x:
    std_val[y[0],y[1]]=1
  return (inMat-mean_val)/std_val

prefix_img='./ff_img/ff_reps/'
prefix_txt='./ff_txt/ff_reps/'

test_image=np.load(prefix_img+"test/hidden3-00001-of-00001.npy")
test_text=np.load(prefix_txt+"test/hidden3-00001-of-00001.npy")

train_image=np.load(prefix_img+"train/hidden3-00001-of-00001.npy")
train_text=np.load(prefix_txt+"train/hidden3-00001-of-00001.npy")

val_image=np.load(prefix_img+"validation/hidden3-00001-of-00001.npy")
val_text=np.load(prefix_txt+"validation/hidden3-00001-of-00001.npy")

# norm
test_image = znorm(test_image, train_image)
test_text = znorm(test_text, train_text)
val_image = znorm(val_image, train_image)
val_text = znorm(val_text, train_text)
train_image = znorm(train_image, train_image)
train_text = znorm(train_text, train_text)

train_img_lab=np.load('feature/train_img_lab_whole.npy')
val_img_lab=np.load('feature/validation_img_lab_whole.npy')
test_img_lab=np.load('feature/test_img_lab_whole.npy')
train_txt_lab=np.load('feature/train_txt_lab_whole.npy')
val_txt_lab=np.load('feature/validation_txt_lab_whole.npy')
test_txt_lab=np.load('feature/test_txt_lab_whole.npy')

train_size = train_image.shape[0]
shuffle = np.random.permutation(train_size)
train_image = train_image[shuffle,:]
train_text = train_text[shuffle,:]
train_img_lab = train_img_lab[shuffle]
train_txt_lab = train_txt_lab[shuffle]

save_dir='../../../../caffe-master/examples/CCL/bin_data/'

test_image.tofile(save_dir+"test_image.bin")
train_image.tofile(save_dir+"train_image.bin")
val_image.tofile(save_dir+"val_image.bin")

test_text.tofile(save_dir+"test_text.bin")
train_text.tofile(save_dir+"train_text.bin")
val_text.tofile(save_dir+"val_text.bin")

test_img_lab.tofile(save_dir+"test_image_lab.bin")
train_img_lab.tofile(save_dir+"train_image_lab.bin")
val_img_lab.tofile(save_dir+"val_image_lab.bin")

test_txt_lab.tofile(save_dir+"test_text_lab.bin")
train_txt_lab.tofile(save_dir+"train_text_lab.bin")
val_txt_lab.tofile(save_dir+"val_text_lab.bin")
