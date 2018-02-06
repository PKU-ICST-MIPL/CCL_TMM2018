// Usage:
// convert_cifar10_data input_image_file input_label_file output_db_file
#define USE_LEVELDB

#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <iostream>
using namespace std;
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#ifdef USE_LEVELDB
#include "leveldb/db.h"
#include "math.h"
#include "stdint.h"
//#include <chrono>


uint32_t swap_endian(uint32_t val) {
    printf("%d\n",val);
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}


//read pixels from image_file(size:rows*cols), read label from image_file(size:4)
void read_image(std::ifstream* image_file, std::ifstream* label_file,
        uint32_t index, uint32_t rows, uint32_t cols,
        float* pixels, char* label_temp, signed char* label, int rgb_use) {
  if (rgb_use == 0) {
    image_file->seekg(index * rows * cols * sizeof(float));
    image_file->read((char*)pixels, rows * cols * sizeof(float));
    label_file->seekg(index);
    label_file->read(label_temp, 1);
    for (int i = 0; i < 1; i++)
      *(label+i) = (signed char)*(label_temp+i);
  } else {
    image_file->seekg(3 * index * rows * cols + 16);
    image_file->read((char*)&pixels, 3 * rows * cols * sizeof(float));
    label_file->seekg(index * 4 + 8);
    label_file->read(label_temp, 4);
    for (int i = 0; i < 4; i++)
      *(label+i) = (signed char)*(label_temp+i);
  }
}

void convert_dataset(const char* image_filename, const char* label_filename,
        const char* db_filename,
                     const char* class_number, const char* dataset_num, const char* dim) {
  int rgb_use1 = 0;
  int class_num = atoi(class_number);
  // Open files
  std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
  std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
  CHECK(image_file) << "Unable to open file " << image_filename;
  CHECK(label_file) << "Unable to open file " << label_filename;
  // Read the magic and the meta data
  uint32_t magic;
  uint32_t num_items = atoi(dataset_num);
  uint32_t num_labels = 10;
  uint32_t rows = 1;
  uint32_t cols = atoi(dim);

  // Open leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";

  char* label_temp = new char[4];  // label for unsigned char*
  signed char* label_i = new signed char[4];  // label for triplet
  //for cifar10, we only use 3 images for anchor,positive,negative
  // signed char* label_l = new signed char[4];  // label for pair wise
  // signed char* label_m = new signed char[4];
  int db_size;
  if (rgb_use1 == 0)
    db_size = rows * cols;
  else
    db_size = 3 * rows * cols;
  float* pixels1 = new float[db_size];
  // char* pixels4 = new char[db_size];
  // char* pixels5 = new char[db_size];
  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];
  std::string value;
  caffe::Datum datum;
  if (rgb_use1 == 0)
    datum.set_channels(1);
  else
    datum.set_channels(3);
  datum.set_height(rows);
  datum.set_width(cols);
  for (int p = 0; p < cols; p++) {
    datum.add_float_data(0);
  }
  
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;

  for (unsigned int itemid = 0; itemid < num_items; ++itemid) {
    int i = itemid % num_items;
    read_image(&image_file, &label_file, i, rows, cols,  
        pixels1, label_temp, label_i, rgb_use1);
    printf("itemid=%d\n", itemid);
    for (int p = 0; p < cols; ++p) {
        //printf("%f ", pixels1[p]);
        datum.set_float_data(p, pixels1[p]);
    }
    printf("\n");
    printf("label=%d\n", (*label_i));
    datum.set_label(static_cast<int>(*label_i));
    datum.SerializeToString(&value);
    snprintf(key, kMaxKeyLength, "%08d", itemid);
    status = db->Put(leveldb::WriteOptions(), std::string(key), value);
    CHECK(status.ok())<<"cnt "<<itemid<<" , LevelDB put error: "<<status.ToString().c_str();
  }

  delete db;
  delete pixels1;
}

int main(int argc, char** argv) {
  if (argc != 7) {
    printf("This script converts the dataset to the leveldb format used\n"
           "by caffe to train a triplet network.\n"
           "Usage:\n"
           "    convert_cifar10_data input_image_file input_label_file "
           "output_db_file class_number  dataset_num dimension \n");
  } else {
    google::InitGoogleLogging(argv[0]);
    printf("call %s ...\n",argv[0]);
    convert_dataset(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
  }
  return 0;
}
#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires LevelDB; compile with USE_LEVELDB.";
}
#endif  // USE_LEVELDB
