#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	
	// number of triplet in a batch

	int num_triplets = this->layer_param_.triplet_loss_param().num_triplets();

	// dimension of each descriptor
	int dim = bottom[0]->count()/bottom[0]->num();
	CHECK_EQ(bottom[0]->channels(), dim);
	CHECK_EQ(bottom[0]->height(), 1);
	CHECK_EQ(bottom[0]->width(), 1);
	
	CHECK_EQ(bottom[1]->channels(), dim);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);

	CHECK_EQ(bottom[2]->channels(), 1);
	CHECK_EQ(bottom[2]->height(), 1);
	CHECK_EQ(bottom[2]->width(), 1);
	
	
	// In each set, we have:
	// the descriptor of reference sample, closest sample, and negative samples
	// number of sets in the whole batch
	int num_set = bottom[0]->num() * num_triplets;	// we only use reference sample, closest sample, and negative sample

	printf("num_set_layer:%d\n",num_set);

	ind_ori.Reshape(num_set, 3, 1, 1);
	dist_sq_.Reshape(num_set, 1, 1, 1);
	diff_pos.Reshape(num_set, dim, 1, 1);
	dist_sq_pos.Reshape(num_set, 1, 1, 1);
	diff_neg.Reshape(num_set, dim, 1, 1);
	dist_sq_neg.Reshape(num_set, 1, 1, 1);
	
	// vector of ones used to sum along channels
	summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
	for (int i = 0; i < bottom[0]->channels(); ++i)
	summer_vec_.mutable_cpu_data()[i] = Dtype(1);

	summer_vec_.Reshape(bottom[1]->channels(), 1, 1, 1);
	for (int i = 0; i < bottom[1]->channels(); ++i)
	summer_vec_.mutable_cpu_data()[i] = Dtype(1);


}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	Dtype margin = this->layer_param_.triplet_loss_param().margin();
	int losstype = this->layer_param_.triplet_loss_param().losstype();
	int num_triplets = this->layer_param_.triplet_loss_param().num_triplets();
	Dtype loss(0.0);
	int dim = bottom[0]->count()/bottom[0]->num();
	int num_set = bottom[0]->num() * num_triplets;
	int num = bottom[0]->num();

	const Dtype* image_data = bottom[0]->cpu_data();
	const Dtype* text_data = bottom[1]->cpu_data();
	const Dtype* label = bottom[2]->cpu_data();

	if (losstype == 1) std::swap(image_data, text_data);

	// Generate quad samples
	Dtype* ind_ori_data = ind_ori.mutable_cpu_data();
	for(int id = 0; id < num_set; id++) {
		while(true) {
			int i = caffe::caffe_rng_rand() % num;
			int j = caffe::caffe_rng_rand() % num;
			int k = caffe::caffe_rng_rand() % num;
			bool quad1_pass = label[i] == label[j];
			bool quad2_pass = label[i] != label[k];
			if (quad1_pass && quad2_pass) {
				ind_ori_data[id * 3] = Dtype(i);						   // I+
				ind_ori_data[id * 3 + 1] = Dtype(j);				   // T+
				ind_ori_data[id * 3 + 2] = Dtype(k);				   // T-
				break;
			}
		}
	}
	
	
	
	for (int id = 0; id < num_set; ++id) {

	int i = static_cast<int>(ind_ori_data[id * 3]);
	int j = static_cast<int>(ind_ori_data[id * 3 + 1]);
	int k = static_cast<int>(ind_ori_data[id * 3 + 2]);
	
	// I+ - T+
	caffe_sub(
		dim,
		image_data + i * dim, //reference
		text_data + j * dim, //positive
		diff_pos.mutable_cpu_data() + id * dim);
	
	dist_sq_pos.mutable_cpu_data()[id] = caffe_cpu_dot(dim,
		diff_pos.cpu_data() + id * dim,
		diff_pos.cpu_data() + id * dim);
	dist_sq_.mutable_cpu_data()[id] = 2 * dist_sq_pos.cpu_data()[id];
	
	// I+ - T-
	caffe_sub(
		dim,
		image_data + i * dim, //reference
		text_data + k * dim, //negative
		diff_neg.mutable_cpu_data() + id * dim);
	dist_sq_neg.mutable_cpu_data()[id] = caffe_cpu_dot(dim,
		diff_neg.cpu_data() + id * dim,
		diff_neg.cpu_data() + id * dim);
	dist_sq_.mutable_cpu_data()[id] -= dist_sq_neg.cpu_data()[id];

	loss += std::max(margin + dist_sq_.cpu_data()[id],Dtype(0.0));
	
	}
	loss = loss / static_cast<Dtype>(num_set) / Dtype(2);
	top[0]->mutable_cpu_data()[0] = loss;
	//printf("loss %f\n",loss);
	//printf("dist_sq_ [0]:%f [%d]%f\tdim:%d\tloss:%f\n", dist_sq_.cpu_data()[0],num_set - 1,dist_sq_.cpu_data()[num_set-1],dim,loss);

}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	Dtype margin = this->layer_param_.triplet_loss_param().margin();
	int losstype = this->layer_param_.triplet_loss_param().losstype();
	int num_triplets = this->layer_param_.triplet_loss_param().num_triplets();
	int dim = bottom[0]->count()/bottom[0]->num();
	//int num_set = bottom[0]->num()/(2 + num_triplets);
	int num_set = bottom[0]->num() * num_triplets;
	//printf("backward dist_sq_ %f %f\n", dist_sq_.cpu_data()[0],dist_sq_.cpu_data()[num_set-1]);
	Dtype* ind_ori_data = ind_ori.mutable_cpu_data(); // quad order
	
	if (propagate_down[0]) {
		const Dtype sign = 1;
		const Dtype alpha = sign * top[0]->cpu_diff()[0] / static_cast<Dtype>(num_set);
		caffe_set(bottom[0]->count(),Dtype(0.0),bottom[0]->mutable_cpu_diff());
		caffe_set(bottom[1]->count(),Dtype(0.0),bottom[1]->mutable_cpu_diff());
		for (int id = 0; id < num_set; id ++) {
			Dtype* bout1 = bottom[0]->mutable_cpu_diff();
			Dtype* bout2 = bottom[1]->mutable_cpu_diff();
			
			if (losstype == 1) std::swap(bout1, bout2);
			
			int i = static_cast<int>(ind_ori_data[id * 3]);			// I+
			int j = static_cast<int>(ind_ori_data[id * 3 + 1]);		// T+
			int k = static_cast<int>(ind_ori_data[id * 3 + 2]);		// T-

			if (dist_sq_.cpu_data()[id] + margin > 0) {
				// BP for feat1(extracted from reference)
				//I+
					caffe_cpu_axpby(
						dim,
						alpha,
						diff_pos.cpu_data() + (id * dim),
						Dtype(1.0),
						bout1 + i * dim);
					caffe_cpu_axpby(
						dim,
						-alpha,
						diff_neg.cpu_data() + (id * dim),
						Dtype(1.0),
						bout1 + i * dim);
				//T+
					caffe_cpu_axpby(
						dim,
						-alpha,
						diff_pos.cpu_data() + (id * dim),
						Dtype(1.0),
						bout2 + j * dim);
				
				// BP for feat2(extracted from positive)
				// T-
					caffe_cpu_axpby(
						dim,
						alpha,
						diff_neg.cpu_data() + (id * dim),
						Dtype(1.0),
						bout2 + k * dim);
				
			}
			else {	// 0
				caffe_set(dim,Dtype(0.0),bout1 + (i) * dim);
				caffe_set(dim,Dtype(0.0),bout1 + (k) * dim);
				caffe_set(dim,Dtype(0.0),bout2 + (j) * dim);
			}
		}
	}
	
	//printf("alpha%f\n",top[0]->cpu_diff()[0] / static_cast<Dtype>(num_set));
	//printf("diff:pos-neg:%f\tpos:%f\tneg:%f\n",diff_pos.cpu_data()[0]*2-diff_neg.cpu_data()[0]*2,diff_pos.cpu_data()[0]*2,diff_neg.cpu_data()[0]*2);
	//printf("bout_[0]:%f[1]:%f[2]:%f\n",bottom[0]->mutable_cpu_diff()[0],bottom[0]->mutable_cpu_diff()[1],bottom[0]->mutable_cpu_diff()[2]);
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}	 // namespace caffe
