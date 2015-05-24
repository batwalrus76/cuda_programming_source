/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

static const int WORK_SIZE = 256;

#define NUM_ELEMENTS 4096

typedef struct {
	u32 a;
	u32 b;
	u32 c;
	u32 d;
} INTERLEAVED_T;

typedef INTERLEAVED_T INTERLEAVED_ARRAY_T[NUM_ELEMENTS];

typedef u32 ARRAY_MEMBER_T[NUM_ELEMENTS];

typedef struct {
	ARRAY_MEMBER_T a;
	ARRAY_MEMBER_T b;
	ARRAY_MEMBER_T c;
	ARRAY_MEMBER_T d;
} NON_INTERLEAVED_T;

__host__ float add_test_non_interleaved_cpu(
		NON_INTERLEAVED_T * const host_dest_ptr,
		const NON_INTERLEAVED_T * const host_src_ptr, const u32 iter,
		const u32 num_elements) {
	float start_time = get_time();

	for (u32 tid = 0; tid < num_elements; tid++) {
		for (u32 i = 0; i < iter; i++) {
			host_dest_ptr->a[tid] += host_src_ptr->a[tid];
			host_dest_ptr->b[tid] += host_src_ptr->b[tid];
			host_dest_ptr->c[tid] += host_src_ptr->c[tid];
			host_dest_ptr->d[tid] += host_src_ptr->d[tid];
		}
	}

	const float delta = get_time() - start_time;

	return delta;
}

__host__ float add_test_interleaved_cpu(INTERLEAVED_T * const host_dest_ptr,
		const INTERLEAVED_T * const host_src_ptr, const u32 iter,
		const u32 num_elements) {
	float start_time = get_time();

	for (u32 tid = 0; tid < num_elements; tid++) {
		for (u32 i = 0; i < iter; i++) {
			host_dest_ptr[tid].a += host_src_ptr[tid].a;
			host_dest_ptr[tid].b += host_src_ptr[tid].b;
			host_dest_ptr[tid].c += host_src_ptr[tid].c;
			host_dest_ptr[tid].d += host_src_ptr[tid].d;
		}
	}

	const float delta = get_time() - start_time;

	return delta;
}

__global__ void add_kernel_interleaved(INTERLEAVED_T * const dest_ptr,
		const INTERLEAVED_T * const src_ptr, const u32 iter,
		const u32 num_elements) {
	float start_time = get_time();

	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(tid < num_elements)
	{
		for(u32 i=0; i<iter; i++)
		{
			dest_ptr[tid].a += src_ptr[tid].a;
			dest_ptr[tid].b += src_ptr[tid].b;
			dest_ptr[tid].c += src_ptr[tid].c;
			dest_ptr[tid].d += src_ptr[tid].d;
		}
	}
}

__global__ void add_kernel_non_interleaved(
		NON_INTERLEAVED_T * const dest_ptr,
		const NON_INTERLEAVED_T * const src_ptr, const u32 iter,
		const u32 num_elements) {
	float start_time = get_time();

	for (u32 tid = 0; tid < num_elements; tid++) {
		for (u32 i = 0; i < iter; i++) {
			dest_ptr->a[tid] += src_ptr->a[tid];
			dest_ptr->b[tid] += src_ptr->b[tid];
			dest_ptr->c[tid] += src_ptr->c[tid];
			dest_ptr->d[tid] += src_ptr->d[tid];
		}
	}
}

__host__ float add_test_interleaved(INTERLEAVED_T * const host_dest_ptr,
		const INTERLEAVED_T * const host_src_ptr, const u32 iter,
		const u32 num_elements)
{
	const u32 num_threads = 256;
	const u32 num_blocks = (num_elements + (num_threads-1)) / num_threads;

	const size_t num_bytes = (sizeof(INTERLEAVED_T) * num_elements);
	INTERLEAVED_T * device_dest_ptr;
	INTERLEAVED_T * device_src_ptr;

	CUDA_CALL(cudaMalloc((void **) &device_src_ptr, num_bytes));
	CUDA_CALL(cudaMalloc((void **) &device_dest_ptr, num_bytes));

	cudaEvent_t kernel_start, kernel_stop;
	cudaEventCreate(&kernel_start1,0);
	cudaEventCreate(&kernel_start2,0);

	cudaStream_t test_stream;
	CUDA_CALL(cudaStreamCreate(&test_stream));

	CUDA_CALL(cudaMemcpy(device_src_ptr, host_src_ptr, num_bytes,cudaMemcpyHostToDevice));

	CUDA_CALL(cudaEventRecord(kernel_start, 0));

	add_kernel_interleaved<<<num_blocks,num_threads>>>(device_dest_ptr, device_src_ptr, iter, num_elements);

	CUDA_CALL(cudaEventRecord(kernel_stop, 0));

	CUDA_CALL(cudaEventSynchronize(kernel_stop));

	float delta = 0.0F;
	CUDA_CALL(cudaEventElapsedTime(&delta, kernel_start, kernel_stop));

	CUDA_CALL(cudaFree(device_src_ptr));
	CUDA_CALL(cudaFree(device_dest_ptr));
	CUDA_CALL(cudaEventDestroy(kernel_start));
	CUDA_CALL(cudaEventDestroy(kernel_stop));
	CUDA_CALL(cudaStreamDestroy(test_stream));

	return delta;
}


/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CALL(x) {														\
	cudaError_t _m_cudaStat = x;											\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__host__ __device__ unsigned int bitreverse(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bitreverse(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse(idata[threadIdx.x]);
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	void *d = NULL;
	int i;
	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];

	for (i = 0; i < WORK_SIZE; i++)
		idata[i] = (unsigned int) i;

	CUDA_CHECK_RETURN(cudaMalloc((void** ) &d, sizeof(int) * WORK_SIZE));
	CUDA_CHECK_RETURN(
			cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE,
					cudaMemcpyHostToDevice));

	bitreverse<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(
			cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE,
					cudaMemcpyDeviceToHost));

	for (i = 0; i < WORK_SIZE; i++)
		printf("Input value: %u, device output: %u, host output: %u\n",
				idata[i], odata[i], bitreverse(idata[i]));

	CUDA_CHECK_RETURN(cudaFree((void* ) d));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}
