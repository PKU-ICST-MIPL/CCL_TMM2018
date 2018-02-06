rm -rf ./examples/CCL/CCL_test_image_data
rm -rf ./examples/CCL/CCL_test_text_data
rm -rf ./examples/CCL/CCL_train_image_data
rm -rf ./examples/CCL/CCL_train_text_data
rm -rf ./examples/CCL/CCL_val_image_data
rm -rf ./examples/CCL/CCL_val_text_data

./build/examples/CCL/convert_float2leveldb.bin ./examples/CCL/bin_data/train_image.bin ./examples/CCL/bin_data/train_image_lab.bin  ./examples/CCL/CCL_train_image_data 20 800 1024
./build/examples/CCL/convert_float2leveldb.bin ./examples/CCL/bin_data/train_text.bin  ./examples/CCL/bin_data/train_text_lab.bin  ./examples/CCL/CCL_train_text_data  20 800 1024
./build/examples/CCL/convert_float2leveldb.bin ./examples/CCL/bin_data/test_image.bin ./examples/CCL/bin_data/test_image_lab.bin ./examples/CCL/CCL_test_image_data 20 100 1024
./build/examples/CCL/convert_float2leveldb.bin ./examples/CCL/bin_data/test_text.bin  ./examples/CCL/bin_data/test_text_lab.bin ./examples/CCL/CCL_test_text_data  20 100 1024
./build/examples/CCL/convert_float2leveldb.bin ./examples/CCL/bin_data/val_image.bin ./examples/CCL/bin_data/val_image_lab.bin  ./examples/CCL/CCL_val_image_data 20 100 1024
./build/examples/CCL/convert_float2leveldb.bin ./examples/CCL/bin_data/val_text.bin  ./examples/CCL/bin_data/val_text_lab.bin  ./examples/CCL/CCL_val_text_data  20 100 1024

