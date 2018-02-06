rm -rf ./examples/CCL/CCL_test_image_res
rm -rf ./examples/CCL/CCL_test_text_res
rm -rf ./examples/CCL/CCL_image_train_res
rm -rf ./examples/CCL/CCL_text_train_res
rm -rf ./examples/CCL/CCL_text_val_res
rm -rf ./examples/CCL/CCL_image_val_res
sh examples/CCL/extratfea_CCL_train_image.sh
sh examples/CCL/extratfea_CCL_train_text.sh
sh examples/CCL/extratfea_CCL_test_image.sh
sh examples/CCL/extratfea_CCL_test_text.sh
sh examples/CCL/extratfea_CCL_val_image.sh
sh examples/CCL/extratfea_CCL_val_text.sh
./build/examples/CCL/check_triplet_data.bin ./examples/CCL/CCL_test_image_res 100 res/test_image.txt
./build/examples/CCL/check_triplet_data.bin ./examples/CCL/CCL_test_text_res 100 res/test_text.txt
./build/examples/CCL/check_triplet_data.bin ./examples/CCL/CCL_image_train_res 800 res/train_image.txt
./build/examples/CCL/check_triplet_data.bin ./examples/CCL/CCL_text_train_res 800 res/train_text.txt
./build/examples/CCL/check_triplet_data.bin ./examples/CCL/CCL_text_val_res 100 res/val_text.txt
./build/examples/CCL/check_triplet_data.bin ./examples/CCL/CCL_image_val_res 100 res/val_image.txt
