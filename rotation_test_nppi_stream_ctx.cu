#include <iostream>
#include <stdio.h>

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

	int img_num = 32; 
	const int stream_num = 4;
	int n_iter = 50; 

	cudaStream_t stream[stream_num];
	NppStreamContext nppStreamContext[stream_num];
	cudaDeviceProp oDeviceProperties;

	for (int n = 0; n < stream_num; n++)
	{
		cudaStreamCreate(&stream[n]);
		nppStreamContext[n].nCudaDeviceId = 0;
		nppStreamContext[n].hStream = stream[n];

		cudaDeviceGetAttribute(&nppStreamContext[n].nCudaDevAttrComputeCapabilityMajor,
			cudaDevAttrComputeCapabilityMajor,
			nppStreamContext[n].nCudaDeviceId);

		cudaDeviceGetAttribute(&nppStreamContext[n].nCudaDevAttrComputeCapabilityMinor,
			cudaDevAttrComputeCapabilityMinor,
			nppStreamContext[n].nCudaDeviceId);

		cudaStreamGetFlags(nppStreamContext[n].hStream, &nppStreamContext[n].nStreamFlags);

		cudaGetDevice(&nppStreamContext[n].nCudaDeviceId);
		cudaGetDeviceProperties(&oDeviceProperties, nppStreamContext[n].nCudaDeviceId);

		nppStreamContext[n].nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
		nppStreamContext[n].nMaxThreadsPerMultiProcessor = oDeviceProperties.maxThreadsPerMultiProcessor;
		nppStreamContext[n].nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
		nppStreamContext[n].nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;
	}

	cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

	double angle = 3.8;

	///////////////////////////*********************** Memory Allocation ***********************/////////////////////////
	Mat* img = new Mat[img_num];
	Mat* result = new Mat[img_num];

	Npp8u* *data = new Npp8u*[img_num];
	Npp8u* *d_data = new Npp8u*[img_num];
	Npp8u* *d_rotated = new Npp8u*[img_num];
	Npp8u* *h_result = new Npp8u*[img_num];



	for (int i = 0; i < img_num; i++)
	{
		img[i] = read_BMP_opencv("input_images/test_2048_2.bmp", f_width, f_height);
		data[i] = img[i].data;
	}

	NppiSize srcSize = { f_width, f_height };
	NppiRect srcROI = { 0, 0, f_width, f_height };
	NppiRect dstROI = { 0, 0, f_width, f_height };

	for (int i = 0; i < img_num; i++)
	{
		cudaMalloc((void**)&d_data[i], sizeof(Npp8u)*f_width*f_height);
		cudaMalloc((void**)&d_rotated[i], sizeof(Npp8u)*f_width*f_height);
		cudaMallocHost((void**)&h_result[i], sizeof(Npp8u)*f_width*f_height);
	}

	///////////////////////////*********************** NPP rotation ***********************/////////////////////////
	cudaEventRecord(start, 0);
	for (int n = 0; n < n_iter; n++)
	{
		for (int i = 0; i < int(img_num / stream_num); i++)
		{
			for (int j = 0; j < stream_num; j++)
			{
				cudaMemcpyAsync(d_data[i*stream_num+j], data[i*stream_num + j], sizeof(Npp8u)*f_width*f_height, cudaMemcpyHostToDevice, stream[j]);
				
				// NPP 10.2 and beyond contain an additional element in the NppStreamContext structure
				nppiRotate_8u_C1R_Ctx(d_data[i*stream_num + j], srcSize, f_width, srcROI, d_rotated[i*stream_num + j], f_width, dstROI, angle, 0, 0, NPPI_INTER_LINEAR, nppStreamContext[j]);
				
				cudaMemcpyAsync(h_result[i*stream_num + j], d_rotated[i*stream_num + j], sizeof(Npp8u)*f_width*f_height, cudaMemcpyDeviceToHost, stream[j]);
			}
		}
	}
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Average Rotation Time: %3.1f ms\n", elapsedTime / n_iter);

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