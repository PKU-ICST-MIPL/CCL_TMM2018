# Introduction
This is the source code of our TMM 2018 paper "CCL: Cross-modal Correlation Learning with Multi-grained Fusion by Hierarchical Network", Please cite the following paper if you use our code.

Yuxin Peng, Jinwei Qi, Xin Huang, and Yuxin Yuan, "CCL: Cross-modal Correlation Learning with Multi-grained Fusion by Hierarchical Network", IEEE Transactions on Multimedia (TMM), Vol. 20, No. 2, pp. 405-420, Feb. 2018. [[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=20184)

# Install
deepnet : please follow ./deepnet-master/INSTALL.txt  
caffe : run make in ./caffe-master

# Data
all the feature data and list files should be put in ./deepnet-master/deepnet/examples/CCL/feature.  
we provide the pascal features and lists we used as an example, which can be download from the [link](http://59.108.48.34/mipl/tiki-download_file.php?fileId=1008) and unzipped to the above path.

# Run CCL
    - cd to ./deepnet-master/deepnet/examples/CCL and execute runall.sh
    - cd to ./caffe-master and execute run_caffe.sh
    
# Our Related Work
If you are interested in cross-media retrieval, you can check our recently published overview paper on IEEE TCSVT:

Yuxin Peng, Xin Huang, and Yunzhen Zhao, "An Overview of Cross-media Retrieval: Concepts, Methodologies, Benchmarks and Challenges", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2017.[[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=201823)

Welcome to our [Benchmark Website](http://59.108.48.34/mipl/xmedia) and [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information about our papers, source codes, and datasets.
