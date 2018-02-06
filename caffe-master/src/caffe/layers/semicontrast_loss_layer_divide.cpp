// Initial version
#include <algorithm>
#include <vector>
#include <typeinfo>
#include <float.h>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

   template <typename Dtype>
     void SemiContrastLossLayer<Dtype>::LayerSetUp(
       const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::LayerSetUp(bottom, top);
        int dim = bottom[0]->count()/bottom[0]->num();
        int count = bottom[0]->count();
        int num = bottom[0]->num();
        int fea_index = this->layer_param_.semicontrast_loss_param().feature_index();
	LOG(INFO)<<"feature_index: "<<fea_index<<"\n";
        CHECK(fea_index==0||fea_index==1);
        LOG(INFO)<<"margin: "<<this->layer_param_.semicontrast_loss_param().margin()
          <<" --- lambda: "<<this->layer_param_.semicontrast_loss_param().lambda()
          <<" --- feature_index: "<<this->layer_param_.semicontrast_loss_param().feature_index()<<"\n";
        LOG(INFO)<<"count: "<<count<<" --- num: "<<num<<" --- dim: "<<dim<<"--- feature dim: "<<bottom[fea_index]->channels()<<"\n";
        CHECK_EQ(bottom[0]->channels(), dim);   // image branch
        CHECK_EQ(bottom[0]->height(), 1);
        CHECK_EQ(bottom[0]->width(), 1);
        CHECK_EQ(bottom[1]->channels(), dim);   // text branch
        CHECK_EQ(bottom[1]->height(), 1);
        CHECK_EQ(bottom[1]->width(), 1);
        CHECK_EQ(bottom[1]->num(),bottom[0]->num());
        CHECK_EQ(bottom[2]->channels(), 1);
        CHECK_EQ(bottom[2]->height(),1);
        CHECK_EQ(bottom[2]->width(),1);
        simi_s.Reshape(num,num,1,1);
        dist_sq_.Reshape(num,num,1,1);
        fea_diff_.Reshape(dim,1,1,1);
        data_diff_.Reshape(bottom[0]->channels(),1,1,1);
     }

   template <typename Dtype>
     void SemiContrastLossLayer<Dtype>::Forward_cpu(
       const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top) {
        //printf("Forward_cpu... ");
        //int count = bottom[0]->count();
        int num = bottom[0]->num();
        int dim = bottom[0]->channels();
        //int fea_index = this->layer_param_.semicontrast_loss_param().feature_index();
        //fea_index = 0;
        //int fea_dim = bottom[fea_index]->channels();
        //printf("fea_dim:%d -- ",dim);
        //reset
        caffe_set(dist_sq_.count(), Dtype(0.0), dist_sq_.mutable_cpu_data());
        caffe_set(simi_s.count(), Dtype(-1.0), simi_s.mutable_cpu_data());

        //const Dtype* bottom_data = bottom[0]->cpu_data(); // bottom_data is n*dim matrix
        //const Dtype* bottom_label = bottom[1]->cpu_data();
        //const Dtype* bottom_fea = bottom[fea_index]->cpu_data();

        const Dtype* image_data = bottom[0]->cpu_data();
        const Dtype* text_data = bottom[1]->cpu_data();
        const Dtype* label = bottom[2]->cpu_data();

        Dtype* simi_s_data = simi_s.mutable_cpu_data();
        Dtype* dist_sq_data = dist_sq_.mutable_cpu_data();

        // Calculate pairwise distance
        caffe_set(simi_s.count(), Dtype(-1.0), simi_s_data);
        for (int i = 0; i < num; i++) {
          for (int j = 0; j < num; j++) {
            caffe_sub(dim, image_data + i * dim, text_data + j * dim, data_diff_.mutable_cpu_data());
            dist_sq_data[i * num + j] = caffe_cpu_dot(dim, data_diff_.cpu_data(), data_diff_.cpu_data());
            //if (label[i] == Dtype(-1.0) || label[j] == Dtype(-1.0)) {

            //}
            //else {
              if (label[i] == label[j])
                simi_s_data[i * num + j] = Dtype(1.0);
              else {
		//int ind = caffe_rng_rand() % 4;
    int ind = caffe_rng_rand() % 1;
		if (ind == 0)
                    simi_s_data[i * num + j] = Dtype(0.0);
		else
                    simi_s_data[i * num + j] = Dtype(-1.0);
		    
	      }
            //}
          }
        }
/*
        // Calculate kNN for I->T
        vector < pair<Dtype, int> > simi_vec;
        int vec_ind = 0;
        int index = 0;
        for (int i = 0; i < num; i++) {
          simi_vec.clear();
          CHECK_EQ(simi_vec.size(), 0);   // clear the vector
          for (int j = 0; j < num; j++) {
            if (label[i] == Dtype(-1.0) || label[j] == Dtype(-1.0)) {
              pair<Dtype, int> pitem(simi_s_data[i * num + j], j);
              simi_vec.push_back(pitem);
            }
          }
          std::sort(simi_vec.begin(), simi_vec.end());
          int length = static_cast<int>(simi_vec.size()) / 3;
          //select semi similar
          while(true) {
            vec_ind = caffe_rng_rand() % std::min(10, length);
            index = simi_vec[vec_ind].second;
            if (simi_s_data[i * num + index] < 0) break;
          }
          CHECK(index >= 0 && index < simi_s.num());
          if(simi_s_data[i * num + index] < 0){
            simi_s_data[i * num + index] = 3;
          }
          //select semi dissimilar
          while(true){
            vec_ind = length * 2 + caffe_rng_rand() % length;
            index = simi_vec[vec_ind].second;
            if(simi_s_data[i * num + index] < 0) break;
          }
          if(simi_s_data[i * num + index] < 0){
            simi_s_data[i * num + index] = 2;
          }
        }

        // Calculate kNN for T->I
        for (int j = 0; j < num; j++) {   // for each text
          simi_vec.clear();
          for (int i = 0; i < num; i++) {   // for each image
            if (label[j] == Dtype(-1.0) || label[i] == Dtype(-1.0)) {
              pair<Dtype, int> pitem(simi_s_data[i * num + j], i);
              simi_vec.push_back(pitem);
            }
          }
          std::sort(simi_vec.begin(), simi_vec.end());
          int length = static_cast<int>(simi_vec.size()) / 3;
          //select semi similar
          while(true) {
            vec_ind = caffe_rng_rand() % std::min(10, length);
            index = simi_vec[vec_ind].second;
            if (simi_s_data[index * num + j] < 0) break;
          }
          CHECK(index >= 0 && index < simi_s.num());
          if(simi_s_data[index * num + j] < 0){
            simi_s_data[index * num + j] = 3;
          }
          //select semi dissimilar
          while(true){
            vec_ind = length * 2 + caffe_rng_rand() % length;
            index = simi_vec[vec_ind].second;
            if(simi_s_data[index * num + j] < 0) break;
          }
          if(simi_s_data[index * num + j] < 0){
            simi_s_data[index * num + j] = 2;
          }
        }
*/
        // Calculate loss
        Dtype margin = this->layer_param_.semicontrast_loss_param().margin();
        Dtype lambda = this->layer_param_.semicontrast_loss_param().lambda();
        Dtype loss(0.0);
	Dtype loss_sim(0.0);
	Dtype loss_dis(0.0);
        int num_simi = 0, num_dissimi = 0;
        //int cntzero = 0, cntequal = 0;
        for(int i = 0; i < num; i++){
           for(int j = 0; j < num; j++){
              //if (i == j) continue;
              //CHECK_EQ(simi_s_data[i*num+j],simi_s_data[j*num+i]);
              //Dtype alpha = simi_s_data[i * num + j] >= Dtype(2.0) ? lambda:Dtype(1.0);
              Dtype alpha = Dtype(1.0);
              if(simi_s_data[i * num + j] == Dtype(1.0)){// || simi_s_data[i * num + j] == Dtype(3.0)){
                //cntequal++; 
                loss += alpha * dist_sq_data[i * num + j];
		 num_simi++;
              }
              else if(simi_s_data[i * num + j] == Dtype(0.0)){// || simi_s_data[i * num + j] == Dtype(2.0)){
                 Dtype dist_d = std::max(static_cast<Dtype>(margin - sqrt(dist_sq_data[i * num + j])), Dtype(0.0));
                 //cntequal++;
                 //if(dist_d == Dtype(0.0))
                 //  cntzero++;
                 loss += alpha * dist_d * dist_d;
                 num_dissimi++;
              }
           }
	   //if (i % 20 == 0)
	   //	printf("dist = %f current loss = %f\n", dist_sq_data[i * num + 0], loss);
        }
        //printf("cntequal = %d, cntzero = %d\n", cntequal, cntzero);

        //printf("num_simi:%d, num_dissimi:%d,--",num_simi,num_dissimi);
        loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
        top[0]->mutable_cpu_data()[0] = loss;
        //printf("--ave loss: %f\n",top[0]->cpu_data()[0]);
        //CHECK_EQ(1,0);
     }


   template <typename Dtype>
     void SemiContrastLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        //printf("Backward_cpu... ");
        Dtype* image_diff = bottom[0]->mutable_cpu_diff();
        Dtype* text_diff = bottom[1]->mutable_cpu_diff();

        const Dtype* image_data = bottom[0]->cpu_data();
        const Dtype* text_data = bottom[1]->cpu_data();

        //const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* simi_s_data = simi_s.cpu_data();
        const Dtype* dist_sq_data = dist_sq_.cpu_data();

        int count = bottom[0]->count();
        int num = bottom[0]->num();
        int dim = count/num;

        Dtype margin = this->layer_param_.semicontrast_loss_param().margin();
        Dtype lambda = this->layer_param_.semicontrast_loss_param().lambda();
        Dtype index = this->layer_param_.semicontrast_loss_param().feature_index();

        Dtype alpha = 2 * top[0]->cpu_diff()[0] / static_cast<Dtype>(num);
        //caffe_set(count,Dtype(0.0),bottom_diff);
        caffe_set(count, Dtype(0.0), image_diff);
        caffe_set(count, Dtype(0.0), text_diff);

        for(int i = 0; i < num; i++){
           for(int j = 0; j < num; j++){
              //if (i == j) continue;
              //CHECK_EQ(simi_s_data[i * num + j], simi_s_data[j * num + i]);
              //Dtype gama = (simi_s_data[i * num + j] >= Dtype(2.0)) ? lambda * alpha :alpha;
              Dtype gama = alpha;
              if(simi_s_data[i * num + j] == Dtype(1.0)){// || simi_s_data[i * num + j] == Dtype(3.0)){
                 // loss for image branch
                 caffe_sub(dim, image_data + i * dim, text_data + j * dim, data_diff_.mutable_cpu_data());
                 caffe_cpu_axpby(dim, gama, data_diff_.cpu_data(), Dtype(1.0), image_diff + i * dim);
                 // loss for text branch
		 if (index == Dtype(1.0)) {
                 caffe_sub(dim, text_data + j * dim, image_data + i * dim, data_diff_.mutable_cpu_data());
                 caffe_cpu_axpby(dim, gama, data_diff_.cpu_data(), Dtype(1.0), text_diff + j * dim);
		 }
              }
              else if(simi_s_data[i * num + j] == Dtype(0.0)){// || simi_s_data[i * num + j] == Dtype(2.0)){
                 Dtype dist = sqrt(dist_sq_data[i * num + j]);
                 Dtype mdist = margin - dist;
                 Dtype beta = -gama * mdist / (dist + Dtype(1e-4));
                 if(mdist > Dtype(0.0)){
                    // loss for image branch
                    caffe_sub(dim, image_data + i * dim, text_data + j * dim, data_diff_.mutable_cpu_data());
                    caffe_cpu_axpby(dim, beta, data_diff_.cpu_data(), Dtype(1.0), image_diff + i * dim);
                    // loss for text branch
		    if (index == Dtype(1.0)){
                    caffe_sub(dim, text_data + j * dim, image_data + i * dim, data_diff_.mutable_cpu_data());
                    caffe_cpu_axpby(dim, beta, data_diff_.cpu_data(), Dtype(1.0), text_diff + j * dim);
		    }
                 }
              }
           }
        }

        //printf("diff:%f %f %f %f\n",bottom_diff[0],bottom_diff[1],bottom_diff[dim-1],bottom_diff[count-1]);
     }

#ifdef CPU_ONLY
   STUB_GPU(SemiContrastLossLayer);
#endif

   INSTANTIATE_CLASS(SemiContrastLossLayer);
   REGISTER_LAYER_CLASS(SemiContrastLoss);

}  // namespace caffe
