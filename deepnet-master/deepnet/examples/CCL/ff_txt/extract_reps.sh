# Extract representation from different layers and write them to disk.
#!/bin/bash
# 
# python ../../extract_neural_net_representation.py <model_file> <train_op> <output_dir> <list of layer names>
python ${deepnet}/extract_neural_net_representation.py\
  ff_txt/Output/mnist_3layer_relu_LAST \
  ff_txt/train.pbtxt \
  ff_txt/ff_reps \
  ff_txt/data.pbtxt\
  hidden3
