#include <iostream>
#include <stdio.h>
#include <fstream>

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#include <opencv2/opencv.hpp>

//#include <nppi.h>
#include <nppcore.h>
#include <nppi_geometry_transforms.h>

using namespace std;
using namespace cv;

Mat read_BMP_opencv(char* filename, int& w, int& h);

int main()
{
	int f_width, f_height;
	char buf[256];

	cudaEvent_t start, stop;
	float  elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const int img_num = 32; // ceil(16384.0f / float(crop_size)) * ceil(8192.0f / float(crop_size))
	// 2048 -> 32, 1536 -> 66, 1024 -> 128, 512 -> 512
	const int stream_num = 4;
	int n_iter = 50; // 시간 측정을 위한 반복 횟수

	cudaStream_t stream[stream_num];
	for (int n = 0; n < stream_num; n++)
	{
		cudaStreamCreate(&stream[n]);
	}

	// imread시 메모리를 pinned memory에 할당하도록 설정
	cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

	double angle = 3.8;

	///////////////////////////*********************** Memory Allocation ***********************/////////////////////////
	Mat* img = new Mat[img_num];
	Mat* result = new Mat[img_num];
	uchar* *data = new uchar*[img_num];
	uchar* *d_data = new uchar*[img_num];
	uchar* *d_rotated = new uchar*[img_num];
	uchar* *h_result = new uchar*[img_num];


	for (int i = 0; i < img_num; i++)
	{
		sprintf(buf, "input_images/src2048.bmp");
		img[i] = read_BMP_opencv(buf, f_width, f_height);
		data[i] = img[i].data;
	}

	NppiSize srcSize = { f_width, f_height };
	NppiRect srcROI = { 0, 0, f_width, f_height };
	NppiRect dstROI = { 0, 0, f_width, f_height };

	for (int i = 0; i < img_num; i++)
	{
		cudaMalloc((void**)&d_data[i], sizeof(uchar)*f_width*f_height);
		cudaMalloc((void**)&d_rotated[i], sizeof(uchar)*f_width*f_height);
		cudaMallocHost((void**)&h_result[i], sizeof(uchar)*f_width*f_height);
	}

	///////////////////////////*********************** NPP rotation ***********************/////////////////////////
	cudaEventRecord(start, 0);
	for (int n = 0; n < n_iter; n++)
	{
		for (int i = 0; i < int(img_num / stream_num); i++)
		{
			for (int j = 0; j < stream_num; j++)
			{
				cudaMemcpyAsync(d_data[i*stream_num+j], data[i*stream_num + j], sizeof(uchar)*f_width*f_height, cudaMemcpyHostToDevice, stream[j]);

				// NPP 10.2 and beyond contain an additional element in the NppStreamContext structure
				nppSetStream(stream[j]);
				nppiRotate_8u_C1R(d_data[i*stream_num + j], srcSize, f_width, srcROI, d_rotated[i*stream_num + j], f_width, dstROI, angle, 0, 0, NPPI_INTER_LINEAR);

				cudaMemcpyAsync(h_result[i*stream_num + j], d_rotated[i*stream_num + j], sizeof(uchar)*f_width*f_height, cudaMemcpyDeviceToHost, stream[j]);
			}
		}
	}
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	for (int i = 0; i < img_num; i++)
	{
		result[i] = Mat(f_height, f_width, CV_8UC1);
		result[i].data = h_result[i];

		sprintf(buf, "output_images/test2048_rotated_nppi_%d.bmp", i);
		imwrite(buf, result[i]);
	}

	return 0;
}

Mat read_BMP_opencv(char* filename, int& w, int& h)
{
	Mat input_img = imread(filename, 0);
	if (input_img.empty())
		throw "Argument Exception";

	int width = input_img.cols;
	int height = input_img.rows;

	w = width;
	h = height;

	return input_img;
}