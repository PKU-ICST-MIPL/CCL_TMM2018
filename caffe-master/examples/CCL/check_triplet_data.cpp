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

#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/highgui/highgui_c.h>  
#include <opencv2/imgproc/imgproc.hpp>

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
        char* pixels, char* label_temp, signed char* label, int rgb_use) {
  if (rgb_use == 0) {
    image_file->seekg(index * rows * cols + 16);
    image_file->read(pixels, rows * cols);
    label_file->seekg(index * 4 + 8);
    label_file->read(label_temp, 4);
    for (int i = 0; i < 4; i++)
      *(label+i) = (signed char)*(label_temp+i);
  } else {
    image_file->seekg(3 * index * rows * cols + 16);
    image_file->read(pixels, 3 * rows * cols);
    label_file->seekg(index * 4 + 8);
    label_file->read(label_temp, 4);
    for (int i = 0; i < 4; i++)
      *(label+i) = (signed char)*(label_temp+i);
  }
}

void check(const char* db_filename, int nlimit, const char* labelpath, const char* pixelpath){
    //const char* db_filename = "/home/junchao/triplet/data/svhn_triplet_train_leveldb_90w_shuffle";
    printf("check: %s\n",db_filename);
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    leveldb::Status status = leveldb::DB::Open(options,db_filename,&db);
    CHECK(status.ok())<<status.ToString().c_str();
    int iMax = 0;
    if(nlimit>=0){
      iMax = nlimit;
    }
    else{
      iMax = INT_MAX; 
    }
    cout << "iMax = " << iMax << endl;
    int i = 0;
    leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
    caffe::Datum datum;
	ofstream labelfile(labelpath,ios::trunc);
    ofstream pixelfile(pixelpath,ios::trunc);
    //ofstream labelfile("/home/qijinwei/caffe/examples/pascal_quad_new/check/train_labels.txt",ios::trunc);
    //ofstream pixelfile("/home/qijinwei/caffe/examples/pascal_quad_new/check/pixels.txt",ios::trunc);
    //int total = 150000;
    //const int kMaxKeyLength = 10;
    //char key[kMaxKeyLength];
    //string value; 
    /*for(int j = 0;j<total;j++){
	sprintf(key,"%08d",j);
	leveldb::Status s = db->Get(leveldb::ReadOptions(),string(key),&value);
	datum.ParseFromString(value);
	if(s.ok()){
		cout<<++i<<" ok: ("<<s.ToString()<<"),"<<value.length()<<",";
	}
	else{
		cout<<++i<<" not ok: ("<<s.ToString()<<"),";
		labelfile<<"not ok: ";
	}
	labelfile<<datum.label()<<endl;
	cout<<string(key)<<","<<datum.label()<<","<<datum.data().length()<<endl;	
    }
*/

    for(it->SeekToFirst();it->Valid();it->Next()){
        //std::cout<<i++<<","<<it->key().ToString()<<","<<it->value().ToString().length()<<endl;
        datum.ParseFromString(it->value().ToString());
	//if (i <= iMax) cout<<++i<<": "<<it->key().ToString()<<","<<datum.label()<<","<<datum.data().length()<<endl;
	labelfile<<datum.label()<<endl;
	i++;

	if(i<=iMax){
          //for(int j=0;j<datum.data().length();j++){
    for(int j=0;j<datum.float_data_size();j++){
	    //pixelfile<<(float)((unsigned char)(datum.data()[j]))<<" ";
            pixelfile << datum.float_data(j) << " ";
	  }
	  pixelfile<<endl;
	}
	else{
	  //break;
	}
	
    }
    labelfile.close();
    pixelfile.close();

}
int main(int argc, char** argv) {
  
  if (argc < 3) {
    printf("This script converts the dataset to the leveldb format used\n"
           "by caffe to train a triplet network.\n"
           "Usage:\n"
           "    check_triplet_data dbfilename nlimit\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    printf("call %s ...\n",argv[0]);
    int nlimit = atoi(argv[2]);
	char labelpath[256], pixelpath[256];
	cerr << argc << endl;
	if (argc > 4) 
	{
		strcpy(labelpath, "/home/qijinwei/CCL-code/caffe-master/examples/CCL/");
		strcat(labelpath, argv[4]);
	}
	else strcpy(labelpath, "/home/qijinwei/CCL-code/caffe-master/examples/CCL/check/train_labels.txt");
	if (argc > 3)
	{
		strcpy(pixelpath, "/home/qijinwei/CCL-code/caffe-master/examples/CCL/");
		cerr << pixelpath << endl;
		strcat(pixelpath, argv[3]);
	}
	else strcpy(pixelpath, "/home/qijinwei/CCL-code/caffe-master/examples/CCL/check/pixels.txt");
    check(argv[1],nlimit, labelpath, pixelpath);
  } 
  return 0;
}
#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires LevelDB; compile with USE_LEVELDB.";
}
#endif  // USE_LEVELDB
