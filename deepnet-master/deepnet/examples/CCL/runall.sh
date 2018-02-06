python mat2npy.py

sh multimodal_dbn/runall_dbn_img.sh
sh multimodal_dbn/runall_dbn_txt.sh
sh multimodal_dbn_whole/runall_dbn_img.sh
sh multimodal_dbn_whole/runall_dbn_txt.sh

python merge.py
cp -r multimodal_dbn_whole/data/dbn_reps/image_rbm2_LAST rbm_1024_whole/data/
cp -r multimodal_dbn_whole/data/dbn_reps/text_rbm2_LAST rbm_1024_whole/data/
cp feature/*_lab_whole.npy rbm_1024/data
cp feature/*_lab_whole.npy rbm_1024_whole/data

sh rbm_1024/runall_corr_ae.sh
sh rbm_1024_whole/runall_corr_ae.sh

cp rbm_1024/data/ae_reps/corr_ae_LAST/0.7/train/image_tied_hidden-00001-of-00001.npy joint_img/data/part_img/train/
cp rbm_1024/data/ae_reps/corr_ae_LAST/0.7/validation/image_tied_hidden-00001-of-00001.npy joint_img/data/part_img/validation/
cp rbm_1024/data/ae_reps/corr_ae_LAST/0.7/test/image_tied_hidden-00001-of-00001.npy joint_img/data/part_img/test/
cp rbm_1024/data/ae_reps/corr_ae_LAST/0.7/train/text_tied_hidden-00001-of-00001.npy joint_txt/data/part_txt/train/
cp rbm_1024/data/ae_reps/corr_ae_LAST/0.7/validation/text_tied_hidden-00001-of-00001.npy joint_txt/data/part_txt/validation/
cp rbm_1024/data/ae_reps/corr_ae_LAST/0.7/test/text_tied_hidden-00001-of-00001.npy joint_txt/data/part_txt/test/
cp rbm_1024_whole/data/ae_reps/corr_ae_LAST/0.7/train/image_tied_hidden-00001-of-00001.npy joint_img/data/whole_img/train/
cp rbm_1024_whole/data/ae_reps/corr_ae_LAST/0.7/validation/image_tied_hidden-00001-of-00001.npy joint_img/data/whole_img/validation/
cp rbm_1024_whole/data/ae_reps/corr_ae_LAST/0.7/test/image_tied_hidden-00001-of-00001.npy joint_img/data/whole_img/test/
cp rbm_1024_whole/data/ae_reps/corr_ae_LAST/0.7/train/text_tied_hidden-00001-of-00001.npy joint_txt/data/whole_txt/train/
cp rbm_1024_whole/data/ae_reps/corr_ae_LAST/0.7/validation/text_tied_hidden-00001-of-00001.npy joint_txt/data/whole_txt/validation/
cp rbm_1024_whole/data/ae_reps/corr_ae_LAST/0.7/test/text_tied_hidden-00001-of-00001.npy joint_txt/data/whole_txt/test/

sh joint_img/runall.sh
sh joint_txt/runall.sh

cp joint_img/joint_reps/joint_img_LAST/train/joint_img_hidden-00001-of-00001.npy ff_img/data/train/
cp joint_img/joint_reps/joint_img_LAST/validation/joint_img_hidden-00001-of-00001.npy ff_img/data/validation/
cp joint_img/joint_reps/joint_img_LAST/test/joint_img_hidden-00001-of-00001.npy ff_img/data/test/
cp joint_txt/joint_reps/joint_txt_LAST/train/joint_txt_hidden-00001-of-00001.npy ff_txt/data/train/
cp joint_txt/joint_reps/joint_txt_LAST/validation/joint_txt_hidden-00001-of-00001.npy ff_txt/data/validation/
cp joint_txt/joint_reps/joint_txt_LAST/test/joint_txt_hidden-00001-of-00001.npy ff_txt/data/test/
cp feature/train_img_lab_whole.npy ff_img/data/train_lab_data.npy
cp feature/validation_img_lab_whole.npy ff_img/data/validation_lab_data.npy
cp feature/test_img_lab_whole.npy ff_img/data/test_lab_data.npy
cp feature/train_txt_lab_whole.npy ff_txt/data/train_lab_data.npy
cp feature/validation_txt_lab_whole.npy ff_txt/data/validation_lab_data.npy
cp feature/test_txt_lab_whole.npy ff_txt/data/test_lab_data.npy

sh ff_img/runall.sh
sh ff_txt/runall.sh

python npy2bin.py
