#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <algorithm>
//#include <iosfwd>
//#include <memory>
#include <string>
//#include <utility>
#include <vector>
#include <cassert>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

#define stride	16
#define face_window	48		// 宽 & 高
#define threshold	0.96f
#define factor	0.793700526 // 缩小因子

typedef struct _BoundingBox
{
	_BoundingBox()
	{
		left_ = top_ = right_ = bottom_ = probability_ = -1.f;
	}

	_BoundingBox(float l, float t, float r, float b, float p) 
		: left_(l), top_(t), right_(r), bottom_(b), probability_(p) {}

	float left_;
	float top_;
	float right_;
	float bottom_;

	float probability_;	// 概率

} BoundingBox;


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void sort_idx(const BoundingBox* boxes, int *idxes, int n);

int sort_stable(int *arr, int n);

void fast_nms(BoundingBox* boxes, int num, float overlap_th, std::vector<int>& picks);

struct data_t
{
    float score;
    int idx;
};

bool cmp(const data_t& data1, const data_t& data2)
{
    return data1.score < data2.score;
}

void sort_idx(const BoundingBox* boxes, int *idxes, int n) 
{
    std::vector<data_t> datas(n);

    for (int i = 0; i < n; ++i) {
        datas[i].score = boxes[i].probability_;
        datas[i].idx = i;
    }

    std::sort(datas.begin(), datas.end(), cmp);

    for (int i = 0; i < n; ++i) {
        idxes[i] = datas[i].idx;
    }
}

int sort_stable(int *arr, int n)
{
    // stable move all -1 to the end
    int i = 0, j = 0;

    while (i < n)
    {
		if (arr[i] == -1)
        {
		    if (j  < i+1)
            {
				j = i+1;
			}
	    	
			while (j < n)
            {
				if (arr[j] == -1)
                {
                    ++j;
                }
				else
                {
		    		arr[i] = arr[j];
		   			arr[j] = -1;
		   	 		j++;
		    		break;
				}
	    	}
	    	if (j == n)
            {
                return i;
            }
		}
		++i;
  	}
    return i;
}

//#define fast_max(x,y) (x - ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1))))
//#define fast_min(x,y) (y + ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1))))
#define simple_max(x, y) ((x) > (y) ? (x) : (y))
#define simple_min(x, y) ((x) < (y) ? (x) : (y))

void fast_nms(BoundingBox* boxes, int num, float overlap, std::vector<int>& picks) 
{
    void *pmem = malloc(sizeof(int) * (num + num));
    int *idxes = (int *)pmem;
    int *pick = idxes + num;
	float *areas = (float*)malloc(sizeof(float) * num);

	int idx_count = num;
    int counter = 0, last_idx;
    int x0,y0,x1,y1;
    int tx0, ty0, tx1, ty1;
    
    for (int i = 0; i < num; ++i)
	{
		idxes[i] = i;
	}
    
    sort_idx(boxes, idxes, num);

    for (int i = 0; i < num; ++i) 
	{
		int ti = idxes[i];
		BoundingBox b = boxes[ti];
		areas[ti] = (b.right_ - b.left_ + 1) * (b.bottom_ - b.top_ + 1); 
	}
    
    while (idx_count > 0) 
	{
        int tmp_idx = idx_count - 1;
		last_idx = idxes[ tmp_idx ];
		pick[counter++] = last_idx;
	
		x0 = boxes[last_idx].left_;
		y0 = boxes[last_idx].top_;
		x1 = boxes[last_idx].right_;
		y1 = boxes[last_idx].bottom_;

		idxes[ tmp_idx ] = -1;

		for (int i = tmp_idx-1; i != -1; i--)
		{
	    	BoundingBox b = boxes[idxes[i]];
	    	//tx0 = fast_max(x0, r.x0);
	    	//ty0 = fast_max(y0, r.y0);
	    	//tx1 = fast_min(x1, r.x1);
	    	//ty1 = fast_min(y1, r.y1);

	    	tx0 = simple_max(x0, b.left_);
	    	ty0 = simple_max(y0, b.top_);
	    	tx1 = simple_min(x1, b.right_);
	    	ty1 = simple_min(y1, b.bottom_);
	    	
			tx0 = tx1 - tx0 + 1;
	    	ty0 = ty1 - ty0 + 1;
	    	if (tx0 > 0 && ty0 >0) 
			{
 				float inter_area = tx0 * ty0;
			    if (inter_area / (areas[ idxes[i] ] + areas[ last_idx ] - inter_area) > overlap)
				{
					idxes[i] = -1;
				} 
			}
		}

		idx_count = sort_stable(idxes, idx_count);
    }

    // just give the selected boxes' indexes, modification needed for real use
    for (int i = 0; i < counter; ++i) 
	{
		picks.push_back(pick[i]);
	}

    free(pmem);
	free(areas);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
class FaceDetector
{
private:
	//shared_ptr<Net<float> > net_;		// 二分类网络
	shared_ptr<Net<float> > fc_net_;	// 全卷积网络

	int num_channels_;					// 3(RGB) or 1(Gray)

public:
	FaceDetector(const string& model_file,		// 二分类网络定义文件
				const string& trained_file,
				const string& fc_model_file,	// 全卷积网络定义文件
				const string& fc_trained_file)	
	{
	#ifdef CPU_ONLY
		Caffe::set_mode(Caffe::CPU);
	#else
		Caffe::set_mode(Caffe::GPU);
	#endif
		
		// load the network.
		//net_.reset(new Net<float>(model_file, TEST));
		//net_->CopyTrainedLayersFrom(trained_file);
		fc_net_.reset(new Net<float>(fc_model_file, TEST));
		fc_net_->CopyTrainedLayersFrom(fc_trained_file);
		
		Blob<float>* input_layer = fc_net_->input_blobs()[0];
		num_channels_ = input_layer->channels();
		assert(3 == num_channels_ || 1 == num_channels_);
	}

	// 将一张图片划分出多尺度（金字塔）,因为是基于滑动窗口的方式，所以一个尺度不准确
	void detect(const cv::Mat& img, std::vector<BoundingBox>& boxes, std::vector<int>& picked)
	{
		std::vector<float> scaleVector;
		int width = img.cols;
		int height = img.rows;
		int min = width < height ? width : height;
		int max = width > height ? width : height;
		
		// 放大的尺度(scale >= 1)
		float scale = float(2400) / max;
		while (scale >= 1)
		{
			scaleVector.push_back(scale);
			scale -= 0.5f;
		}
		
		// 缩小的尺度(scale < 1)
		min *= factor;
		int cnt = 1;
		while (min >= face_window)
		{
			scale = pow(factor, cnt++);
			scaleVector.push_back(scale);
			min *= factor;
		}
		
		//for (int i = 0; i < scaleVector.size(); ++i)
		//{
		//	printf("scale %d = %f\n", i, scaleVector[i]);
		//}
		//return;

		//std::vector<BoundingBox> boxes;
		for (int i = 0; i < scaleVector.size(); ++i)
		{
			float scale = scaleVector[i];
			cv::Mat img_resized;
			cv::resize(img, img_resized, cv::Size(width * scale, height * scale), 0, 0, CV_INTER_LINEAR);
			doDetect(img_resized, scale, boxes);
		}

		BoundingBox* p_boxes = new BoundingBox[boxes.size()];
		for (int i = 0; i < boxes.size(); ++i)
		{
			p_boxes[i] = boxes[i];
		}
		
		const float overlap = 0.25f;
		printf("before nms, box number = %d\n", boxes.size());
		//std::vector<int> picked;
		fast_nms(p_boxes, boxes.size(), overlap, picked);
		printf("after nms, box number = %d\n", picked.size());
		
		delete[] p_boxes;
	}

	void draw_rect(cv::Mat& img, const std::vector<BoundingBox>& boxes)
	{
		for (int i = 0; i < boxes.size(); ++i)
		{
			cv::rectangle(img, cv::Rect(boxes[i].left_, 
										boxes[i].top_,
										boxes[i].right_ - boxes[i].left_,
										boxes[i].bottom_ - boxes[i].top_),
										cv::Scalar(255, 0, 0), 2);

			cv::putText(img, std::to_string(boxes[i].probability_), 
											cv::Point(boxes[i].left_, boxes[i].top_), 
											3, 0.8, cv::Scalar(255, 0, 0), 1);		
		}
	}

private:
	void doDetect(const cv::Mat& img, const float scale, std::vector<BoundingBox>& boxes)
	{
		Blob<float>* input_layer = fc_net_->input_blobs()[0];
		input_layer->Reshape(1, num_channels_, img.rows, img.cols);
		
		/* Forward dimension change to all layers. */
		fc_net_->Reshape();
		std::vector<cv::Mat> input_channels;
		wrapInputLayer(img, input_channels);
		preprocess(img, input_channels);
		
		printf("sg.xu: forward ...\n");
		fc_net_->Forward();
		generateBoundingBox(boxes, scale);
	}


	/* Wrap the input layer of the network in separate cv::Mat objects
	 * (one per channel). This way we save one memcpy operation and we
	 * don't need to rely on cudaMemcpy2D. The last preprocessing
	 * operation will write the separate channels directly to the input
	 * layer. */
	void wrapInputLayer(const cv::Mat& img, std::vector<cv::Mat>& input_channels)
	{
		Blob<float>* input_layer = fc_net_->input_blobs()[0];
		int w = input_layer->width();
		int h = input_layer->height();
		assert(img.cols == w && img.rows == h);

		float* input_data = input_layer->mutable_cpu_data();
		for (int i = 0; i < input_layer->channels(); ++i)
		{
			cv::Mat channel(h, w, CV_32FC1, input_data);
			input_channels.push_back(channel);
			input_data += w * h;
		}
	
	}

	void preprocess(const cv::Mat& img, std::vector<cv::Mat>& input_channels)
	{
		/* Convert the input image to the input image format of the network. */
		cv::Mat sample;
  		if (img.channels() == 3 && num_channels_ == 1)
		{
			cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
		}
		else if (img.channels() == 4 && num_channels_ == 1)
		{
    		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
		}
		else if (img.channels() == 4 && num_channels_ == 3)
		{
			cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
		}
		else if (img.channels() == 1 && num_channels_ == 3)
		{
			cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
		}
		else
		{
			sample = img;
		}
		
		//cv::Mat sample_resized;
		//if (sample.size() != input_geometry_)
		//{
		//	cv::resize(sample, sample_resized, input_geometry_);
		//}
		//else
		//{
		//	sample_resized = sample;
		//}
		cv::Mat sample_resized = sample;
		
		cv::Mat sample_float;
		if (num_channels_ == 3)
		{
    		sample_resized.convertTo(sample_float, CV_32FC3);
		}
		else
		{
	  		sample_resized.convertTo(sample_float, CV_32FC1);
		}
	
		//cv::Mat sample_normalized;
		//cv::subtract(sample_float, mean_, sample_normalized);
		cv::Mat sample_normalized = sample_float;
	
		/* This operation will write the separate BGR planes directly to the
		 * input layer of the network because it is wrapped by the cv::Mat
		 * objects in input_channels. */
		cv::split(sample_normalized, input_channels);
		
		assert(reinterpret_cast<float*>(input_channels.at(0).data) == fc_net_->input_blobs()[0]->cpu_data());
	}

	void generateBoundingBox(std::vector<BoundingBox>& boxes, const float scale)
	{
		Blob<float>* output_layer = fc_net_->output_blobs()[0];
		const float* output_data = output_layer->cpu_data();
		
		//int num = output_layer->num();
		//int channels = output_layer->channels();
		//int width = output_layer->width();
		//int height = output_layer->height();
		//
		//printf("output_layer->num = %d\n", num);
		//printf("output_layer->channels = %d\n", channels);
		//printf("output_layer->width = %d\n", width);
		//printf("output_layer->height = %d\n", height);

		//for (int n = 0; n < num; ++n)
		//{
		//	for (int c = 0; c < channels; ++c)
		//	{
		//		for (int h = 0; h < height; ++h)
		//		{
		//			for (int w = 0; w < width; ++w)
		//			{
		//				int idx = ((n * channels +c) * height + h) * width + w;
		//				printf("%f ", output_data[idx]);
		//			}
		//			printf("\n");
		//		}
		//		printf("\n--------------------------------------\n");
		//	}
		//}
		int num = output_layer->num();
		int channels = output_layer->channels();
		int width = output_layer->width();
		int height = output_layer->height();
		
		assert(2 == channels);
		for (int n = 0; n < num; ++n)
		{
			for (int c = 0; c < channels - 1; ++c)
			{
				// 还原到原图的坐标
				for (int h = 0; h < height; ++h)
				{
					for (int w = 0; w < width; ++w)
					{
						int idx = ((n * channels + c) * height + h) * width + w;
						float p = output_data[idx];
						if (p < threshold)
						{
							continue;
						}
						int x = w - 1;	// 有负数??
						int y = h - 1;
						
						BoundingBox box((stride * x) / scale, 
										(stride * y) / scale,
										(stride * x + face_window - 1) / scale,
										(stride * y + face_window - 1) / scale,
										p);
						
						boxes.push_back(box);	
					}	// for w
				}	// for h
			} // for c
		} // for n
		
		return;
	}
};

int main()
{
	string model_file = "model/deploy.prototxt";
	string trained_file = "model/snapshot_iter_100000.caffemodel";
	string fc_model_file = "model/deploy_fc.prototxt";
	string fc_trained_file = "model/snapshot_iter_100000_fc.caffemodel";
	FaceDetector d(model_file, trained_file, fc_model_file,	fc_trained_file);
	
	cv::Mat img;
	img = cv::imread("images/0.jpg");
	if (NULL == img.data)
	{
		printf("read image error!\n");
		return -1;
	}
	
	printf("sg.xu: before detect!\n");
	
	std::vector<BoundingBox> boxes;
	std::vector<int> picked;	// index
	d.detect(img, boxes, picked);

	std::vector<BoundingBox> valid_boxes;
	for (int i = 0; i < picked.size(); ++i)
	{
		valid_boxes.push_back(boxes[picked[i]]);
	}

	d.draw_rect(img, valid_boxes);
	
	cv::imwrite("results/0.jpg", img);
}
