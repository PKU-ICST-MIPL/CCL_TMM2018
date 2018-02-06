import os, sys
import numpy as np
import math
import scipy.spatial
from operator import itemgetter, attrgetter
import scipy.io as sio

def fx_calc_map_label(image, text, label, k = 0, dist_method='L2'):
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)

def fx_calc_map_label_and_save(image, text, label, save_dir, k = 0, dist_method='L2'):
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  np.save(save_dir, ord)
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)

def fx_calc_map_label_and_savemat(image, text, label, save_dir, k = 0, dist_method='L2'):
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  sio.savemat(save_dir,{'ord':ord})
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)

def fx_calc_map_label_sorted(ord, label, k = 0, dist_method='L2'):
  numcases = ord.shape[0]
  print numcases
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)

def fx_calc_map_label_sorted_withoutVA(ord, label, va, k = 0, dist_method='L2'):
  numcases = ord.shape[0]
  print numcases
  for i in range(5000):
    label[va[i]]=1000
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    if i%1000==0:
        print i
    if i in va:
        continue
    order = ord[i]
    p = 0.0
    r = 0.0
    pos = 0
    for j in range(k):
    #  if order[j] in va:
    #      continue
      pos += 1
      if label[order[j]]==1000:
        pos -= 1
        continue
      if label[i] == label[order[j]]:
        r += 1
        p += (r / pos)
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)

def fx_calc_map_label_calPR_withoutVA(image, text, label, fileName, k = 0, dist_method='COS'):
  if dist_method=='COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  elif dist_method == 'Multi':
    dist = np.dot(image, text.T)
  ord = dist.argsort()
  numcases = ord.shape[0]
  numcases2 = ord.shape[1]
  recall = np.zeros([numcases, numcases2])
  precision = np.zeros([numcases, numcases2])
  print numcases
  label = np.array(label)
  if k == 0:
    k = numcases2
  res = []
  vaCnt = 0
  for i in range(numcases):
    vaCnt += 1
    #if i%1000==0:
    #    print i
    order = ord[i]
    p = 0.0
    r = 0.0
    pos = 0
    recallAll = np.sum(label == label[i])
    for j in range(k):
    #  if order[j] in va:
    #      continue
      pos += 1
      if label[order[j]]==20:
        pos -= 1
        continue
      if label[i] == label[order[j]]:
        r += 1
        p += (r / pos)
	precision[vaCnt-1,pos-1] = r / pos
	recall[vaCnt-1,pos-1] = r / recallAll
      else:
        precision[vaCnt-1,pos-1] = r / pos
	recall[vaCnt-1,pos-1] = r / recallAll
    if r > 0:
      res += [p / r]
    else:
      res += [0]

  #print len(res)
  #print pos
  print np.mean(res)

  newnumcases = numcases
  tmp1 = np.zeros([newnumcases, 1])
  tmp2 = np.ones([newnumcases, 1])
  recall = np.column_stack([tmp1, recall])
  precision = np.column_stack([tmp2, precision])
  precisionValue = np.zeros([1000, numcases])
  count = 0
  for recallValue in range(1,1000,1):
    #if recallValue%100==0:
    #    print recallValue
    recallValue=recallValue/1000.0
    count = count + 1
    flag = recall < recallValue
    flagPlace = np.sum(flag, 1)
    for j in range(numcases):
      precisionValue[count, j] = calPrecision(precision[j, flagPlace[j]-1], recall[j, flagPlace[j]-1], precision[j, flagPlace[j]], recall[j, flagPlace[j]],recallValue)

  recallValue = np.arange(0.001,1.001,0.001)
  recallValue = recallValue.T

  precision = np.mean(precisionValue, 1)
  prcurve = np.column_stack([precision, recallValue])

  np.savetxt(fileName, prcurve)

  return np.mean(res)

def calPrecision(y1, x1, y2, x2, x):
  return (y2 - y1) * (x - x1) / (x2 - x1) + y1

def fx_calc_map_label_k_dist(dist, label, k = 0, dist_method='L2'):
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)

def fx_calc_map_multilabel(image, text, label, k=0, n=1000, dist_method='L2'):
  image = image[:n,:]
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  # print k
  res = []
  for i in range(numcases):
    order = ord[i].reshape(-1)

    tmp_label = (np.dot(label[order], label[i]) > 0)
    if tmp_label.sum() > 0:
      prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
      total_pos = float(tmp_label.sum())
      if total_pos > 0:
        res += [np.dot(tmp_label, prec) / total_pos]

  return np.mean(res)

def fx_calc_map_multilabel_k(image, text, label, k=0, dist_method='L2'):
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i].reshape(-1)

    tmp_label = (np.dot(label[order], label[i]) > 0)
    if tmp_label.sum() > 0:
      prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
      total_pos = float(tmp_label.sum())
      if total_pos > 0:
        res += [np.dot(tmp_label, prec) / total_pos]

  return np.mean(res)

def fx_calc_map_nolabel(image, text, dist_method='COS'):
  if dist_method == 'L1':
    dist = scipy.spatial.distance.cdist(image, text, 'minkowski', 1)
  elif dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  order = dist.argsort()
  numcases = dist.shape[0]
  res = np.zeros((numcases, numcases))

  for i in range(numcases):
    res[i, order[i].tolist().index(i)] = 1

  res = (numcases - res.cumsum(axis=1).sum(axis=0)) / numcases
  area = 0.5 * (1 + res[0])
  for i in range(numcases-1):
    area += 0.5 * (res[i] + res[i+1])
  area /= numcases
  return area

def fx_calc_map_nolabel_top(image, text, per=0.2, top_k=0):
  numcases = image.shape[0]
  if per != 0:
    top_k = numcases*per
  if top_k == 0:
    print 'make_test error'
    return
  dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  order = dist.argsort()
  r = 0
  for i in xrange(numcases):
    if order[i,:].tolist().index(i) < top_k:
      r += 1
  return r / float(numcases)

def fx_calc_dcg_k(image, text, indexes, label, k, dist_method='COS'):
  dist = -(image*text).sum(axis=1)/np.sqrt((image**2).sum(axis=1))
  dist /= np.sqrt((text**2).sum(axis=1))

  # dist = ((image-text)**2).sum(axis=1)

  index2image = {}
  keys = set()
  for i,ind in enumerate(indexes):
    if ind not in keys:
      keys.add(ind)
      index2image[ind] = []
    index2image[ind] += [(dist[i], label[i])]

  res = []
  for ind,dist_lab in index2image.items():
    dist_lab.sort(key=itemgetter(0))
    s = 0.0
    ct = 1
    for d,lab in dist_lab:
      s += (2.0**lab-1)/math.log(ct+1,2)
      if ct == k:
        break
      ct += 1
    s *= 0.01757
    res += [s]
  return np.mean(res)

def fx_calc_dcg_k_dist(dist, indexes, label, k):
  index2image = {}
  keys = set()
  for i,ind in enumerate(indexes):
    if ind not in keys:
      keys.add(ind)
      index2image[ind] = []
    index2image[ind] += [(dist[i], label[i])]

  res = []
  for ind,dist_lab in index2image.items():
    dist_lab.sort(key=itemgetter(0))
    s = 0.0
    ct = 1
    for d,lab in dist_lab:
      s += (2.0**lab-1)/math.log(ct+1,2)
      if ct == k:
        break
      ct += 1
    s *= 0.01757
    res += [s]
  return np.mean(res)

def fx_calc_dcg_k_dist_onequery(dist_lab, k):
  dist_lab.sort(key=itemgetter(0))
  s = 0.0
  ct = 1
  for d,lab in dist_lab:
    s += (2.0**lab-1)/math.log(ct+1,2)
    if ct == k:
      break
    ct += 1
  s *= 0.01757
  return s
