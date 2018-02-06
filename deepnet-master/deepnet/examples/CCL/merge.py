import numpy as np
import os

prefix_img = './feature'
prefix_txt = './feature'

for dataset in ['train', 'validation', 'test']:
  if not os.path.exists('rbm_1024/data/image_rbm2_LAST/'+dataset):
    os.makedirs('rbm_1024/data/image_rbm2_LAST/'+dataset)
  if not os.path.exists('rbm_1024/data/text_rbm2_LAST/'+dataset):
    os.makedirs('rbm_1024/data/text_rbm2_LAST/'+dataset)

  imgList = open(os.path.join(prefix_img, dataset+'set_img.list'))
  imgs = []
  for img in imgList.readlines():
    img = img.strip('\n')
    img = img.strip('\r')
    imgs.append(img)
  imgSegList = open(os.path.join(prefix_img, dataset+'set_imgSeg.list'))
  imgSegs = []
  for imgSeg in imgSegList.readlines():
    imgSeg = imgSeg.strip('\n')
    imgSeg = imgSeg.strip('\r')
    imgSegs.append(imgSeg)

  imgPartFea = np.load('multimodal_dbn/data/dbn_reps/image_rbm2_LAST/'+dataset+'/image_hidden2-00001-of-00001.npy')
  d = (imgPartFea.shape)[1]

  l = len(imgs)
  print('l = %d, d = %d'%(l,d))
  imgMergedFea = np.zeros([l,d])

  cnt = 0
  start = 0
  i = 0
  for i in range(l-1):
    stop = imgSegs.index(imgs[i+1])-1
    x = stop - start
    if x==0:
      x = 1
    tmp = imgPartFea[cnt:cnt+x,:]
    cnt = cnt + x
    start = stop + 1
    tmp = np.mean(tmp, axis=0)
    imgMergedFea[i,:] = tmp
    if i%100 == 0:
      print('finish : %d/%d'%(i,l))

  last = (imgPartFea.shape)[0]
  tmp = imgPartFea[cnt:last,:]
  print(cnt)
  print(last)
  print(tmp.shape)
  tmp = np.mean(tmp, axis=0)
  imgMergedFea[l-1,:] = tmp

  np.save('rbm_1024/data/image_rbm2_LAST/'+dataset+'/image_hidden2-00001-of-00001.npy', imgMergedFea)


  txtList = open(os.path.join(prefix_txt, dataset+'set_txt.list'))
  txts = []
  for txt in txtList.readlines():
    txt = txt.strip('\n')
    txt = txt.strip('\r')
    txts.append(txt)
  txtSegList = open(os.path.join(prefix_txt, dataset+'set_txtSeg.list'))
  txtSegs = []
  for txtSeg in txtSegList.readlines():
    txtSeg = txtSeg.strip('\n')
    txtSeg = txtSeg.strip('\r')
    txtSegs.append(txtSeg)

  txtPartFea = np.load('multimodal_dbn/data/dbn_reps/text_rbm2_LAST/'+dataset+'/text_hidden2-00001-of-00001.npy')
  d = (txtPartFea.shape)[1]

  l = len(txts)
  print('l = %d, d = %d'%(l,d))
  txtMergedFea = np.zeros([l,d])

  cnt = 0
  start = 0
  for i in range(l-1):
    stop = txtSegs.index(txts[i+1])-1
    x = stop - start
    if x==0:
      x = 1
    tmp = txtPartFea[cnt:cnt+x,:]
    cnt = cnt + x
    start = stop + 1
    tmp = np.mean(tmp, axis=0)
    txtMergedFea[i,:] = tmp
    if i%100 == 0:
      print('finish : %d/%d'%(i,l))

  last = (txtPartFea.shape)[0]
  tmp = txtPartFea[cnt:last,:]
  print(cnt)
  print(last)
  print(tmp.shape)
  tmp = np.mean(tmp, axis=0)
  txtMergedFea[l-1,:] = tmp

  np.save('rbm_1024/data/text_rbm2_LAST/'+dataset+'/text_hidden2-00001-of-00001.npy', txtMergedFea)
