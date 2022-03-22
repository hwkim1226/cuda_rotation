#include <iostream>
#include <stdio.h>
#include <fstream>
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define PI 3.14159265f

Mat read_BMP_opencv(char* filename, int& w, int& h);
__global__ void rotateImage(uchar* idata, uchar* odata, float sine, float cosine, int xCenter, int yCenter, int cols, int rows);

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
	for (int n = 0; n < stream_num; n++)
	{
		cudaStreamCreate(&stream[n]);
	}
	cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

	Mat* img = new Mat[img_num];
	Mat* result = new Mat[img_num];

	uchar* *data = new uchar*[img_num];
	uchar* *d_data = new uchar*[img_num];
	uchar* *d_rotated = new uchar*[img_num];
	uchar* *h_result = new uchar*[img_num];

	for (int i = 0; i < img_num; i++)
	{
		img[i] = read_BMP_opencv("input_images/src2048.bmp", f_width, f_height);
		data[i] = img[i].data;
	}

	for (int i = 0; i < img_num; i++)
	{
		cudaMalloc((void**)&d_data[i], sizeof(uchar)*f_width*f_height);
		cudaMalloc((void**)&d_rotated[i], sizeof(uchar)*f_width*f_height);
		cudaMallocHost((void**)&h_result[i], sizeof(uchar)*f_width*f_height);
	}

	float angle = 3.8f * PI / 180.0f;
	float sine = sin(angle);
	float cosine = cos(angle);
	
	dim3 threadsPerBlock(32, 32, 1);
	dim3 numBlocks(int(f_width / threadsPerBlock.x), int(f_height / threadsPerBlock.y), 1);

	// define rotation center
	//int xCenter = (int)(f_width / 2);
	//int yCenter = (int)(f_height / 2);
	int xCenter = 0;
	int yCenter = 0;


	cudaEventRecord(start, 0);
	for (int n = 0; n < n_iter; n++)
	{
		for (int i = 0; i < int(img_num / stream_num); i++)
		{
			for (int j = 0; j < stream_num; j++)
			{
				cudaMemcpyAsync(d_data[i*stream_num + j], data[i*stream_num + j], sizeof(uchar)*f_width*f_height, cudaMemcpyHostToDevice, stream[j]);
				rotateImage << <numBlocks, threadsPerBlock, 0, stream[j] >> > (d_data[i*stream_num + j], d_rotated[i*stream_num + j], sine, cosine, xCenter, yCenter, f_width, f_height);
				cudaMemcpyAsync(h_result[i*stream_num + j], d_rotated[i*stream_num + j], sizeof(uchar)*f_width*f_height, cudaMemcpyDeviceToHost, stream[j]);
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

		sprintf(buf, "output_images/test2048_rotated_kernel_%d.bmp", i);
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

__global__ void rotateImage(uchar* idata, uchar* odata, float sine, float cosine, int xCenter, int yCenter, int cols, int rows)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float orig_x, orig_y, offset_x, offset_y;
	int round_x, round_y;
	float fx = (float)x;
	float fy = (float)y;

	orig_x = xCenter - (fy - yCenter)*sine + (fx - xCenter)*cosine;
	orig_y = yCenter + (fy - yCenter)*cosine + (fx - xCenter)*sine;
	round_x = rintf(orig_x);
	round_y = rintf(orig_y);

	int dx = 1;
	int dy = 1;
	offset_x = orig_x - round_x;
	if (offset_x < 0) { dx = -1; offset_x *= -1.0f; }
	offset_y = orig_y - round_y;
	if (offset_y < 0) { dy = -1; offset_y *= -1.0f; }

	if ((round_y >= 0 && round_y < cols) && (round_x >= 0 && round_x < rows)) // (4)
	{
		float val = idata[round_y*cols + round_x] * (1.0f - offset_x) * (1.0f - offset_y)
			+ idata[round_y*cols + round_x + dx] * offset_x * (1.0 - offset_y)
			+ idata[(round_y+dy)*cols + round_x] * (1.0f - offset_x) * offset_y
			+ idata[(round_y+dy)*cols + round_x + dx] * offset_x * offset_y;

		odata[y*cols + x] = (uchar)val;
	}
	else
	{
		odata[y*cols + x] = 0;
	}
}
