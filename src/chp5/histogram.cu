#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 256
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

unsigned int cpu_block[ARRAY_SIZE];

/* Each thread writes to one block of 256 elements of global memory and contends for write access */
__global__ void myhistogram256kernel_01(unsigned int * hist_data,
		unsigned int * bin_data) {
	for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		bin_data[i] = 0;
	}

	/* Work out our thread id */
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned int tid = ((gridDim.x * blockDim.x) * idy) + idx;

	/* Fetch the data value */
	const unsigned char value = hist_data[tid];
	atomicAdd(&(bin_data[value]), 1);
}

/* Each read is 4 bytes, not one, 32 x 4 = 128 byte reads */
__global__ void myhistogram256kernel_02(unsigned int * d_hist_data,
		unsigned int *d_bin_data) {
	/* Work out our thread id */
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

	/* Fetch the data value */
	const unsigned int value_u32 = d_hist_data[tid];
	atomicAdd(&(d_bin_data[((value_u32 & 0x000000FF))]), 1);
	atomicAdd(&(d_bin_data[((value_u32 & 0x0000FF00 >> 8))]), 1);
	atomicAdd(&(d_bin_data[((value_u32 & 0x00FF0000 >> 16))]), 1);
	atomicAdd(&(d_bin_data[((value_u32 & 0xFF000000 >> 24))]), 1);
}

__shared__ unsigned int d_bin_data_shared[256];

/* Each read is 4 bytes, not one, 32 x 4 = 128 byte reads */
__global__ void myhistogram256kernel_03(unsigned int * d_hist_data,
		unsigned int * d_bin_data) {
	/* Work out our thread id */
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

	/* Clear shared memory */
	d_bin_data_shared[threadIdx.x] = 0;

	/* Fetch the data value */
	const unsigned int value_u32 = d_hist_data[tid];

	/* Wait for all threads to update shared memory */
	__syncthreads();

	atomicAdd(&(d_bin_data[((value_u32 & 0x000000FF))]), 1);
	atomicAdd(&(d_bin_data[((value_u32 & 0x0000FF00 >> 8))]), 1);
	atomicAdd(&(d_bin_data[((value_u32 & 0x00FF0000 >> 16))]), 1);
	atomicAdd(&(d_bin_data[((value_u32 & 0xFF000000 >> 24))]), 1);

	/* Wait for all threads to update shared memory */
	__syncthreads();

	/* The write the accumulated data back to global memory in blocks, not scattered */
	atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_shared[threadIdx.x]);
}

/* Each read is 4 bytes, not one, 32 x 4 = 128 byte reads */
__global__ void myhistogram256kernel_04(unsigned int * d_hist_data,
		unsigned int * d_bin_data, unsigned int N) {
	/* Work out our thread id */
	const unsigned int idx = (blockIdx.x * (blockDim.x * N)) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int tid = idx + idy * (blockDim.x * N) * gridDim.x;

	/* Clear shared memory */
	d_bin_data_shared[threadIdx.x] = 0;

	/* Wait for all threads to update shared memory */
	__syncthreads();

	for (unsigned int i = 0, tid_offset = 0; i < N; i++, tid_offset += 256) {
		const unsigned int value_u32 = d_hist_data[tid + tid_offset];
		atomicAdd(&(d_bin_data[((value_u32 & 0x000000FF))]), 1);
		atomicAdd(&(d_bin_data[((value_u32 & 0x0000FF00 >> 8))]), 1);
		atomicAdd(&(d_bin_data[((value_u32 & 0x00FF0000 >> 16))]), 1);
		atomicAdd(&(d_bin_data[((value_u32 & 0xFF000000 >> 24))]), 1);
	}

	/* Wait for all threads to update shared memory */
	__syncthreads();

	/* The write the accumulated data back to global memory in blocks, not scattered */
	atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_shared[threadIdx.x]);
}

void generate_random_pointers(unsigned int * data)
{
	for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		unsigned int randNumber = rand() % ARRAY_SIZE;
		data[i] = randNumber;
	}
}

void perform_histogram_kernel(void (* kernel_function)(unsigned int *, unsigned int *))
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	const unsigned int num_blocks = ARRAY_SIZE/32;
	const unsigned int num_threads = ARRAY_SIZE/num_blocks;

	/* Declare pointers for GPU based params */
	unsigned int *hist_data;
	unsigned int *bin_data;

	cudaMalloc(&hist_data, ARRAY_SIZE_IN_BYTES);
	cudaMalloc(&bin_data, ARRAY_SIZE_IN_BYTES);
	generate_random_pointers(cpu_block);

	cudaMemcpy( hist_data, cpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

	/* Execute our kernel */
	cudaEventRecord(start);
	kernel_function<<<num_blocks, num_threads>>>(hist_data, bin_data);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time elapsed: %fn\n", milliseconds);

	cudaMemcpy(cpu_block, bin_data, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	cudaFree(hist_data);
	cudaFree(bin_data);

	/* Iterate through the arrays and print */
	for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("histogram[%2u]=%2u\n",i,cpu_block[i]);
	}
}

void perform_histogram_kernel_N(void (* kernel_function)(unsigned int *, unsigned int *, unsigned int), unsigned int N)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	const unsigned int num_blocks = ARRAY_SIZE/32;
	const unsigned int num_threads = ARRAY_SIZE/num_blocks;

	/* Declare pointers for GPU based params */
	unsigned int *hist_data;
	unsigned int *bin_data;

	cudaMalloc(&hist_data, ARRAY_SIZE_IN_BYTES);
	cudaMalloc(&bin_data, ARRAY_SIZE_IN_BYTES);
	generate_random_pointers(cpu_block);

	cudaMemcpy( hist_data, cpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

	/* Execute our kernel */
	cudaEventRecord(start);
	kernel_function<<<num_blocks, num_threads>>>(hist_data, bin_data, N);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time elapsed: %fn\n", milliseconds);

	cudaMemcpy(cpu_block, bin_data, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	cudaFree(hist_data);
	cudaFree(bin_data);

	/* Iterate through the arrays and print */
	for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("histogram[%2u]=%2u\n",i,cpu_block[i]);
	}
}
void main_sub0()
{
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
	printf("myhistogram256kernel_01\n");
//	cudaEventRecord(start);
	perform_histogram_kernel(&myhistogram256kernel_01);
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	float milliseconds = 0;
//	cudaEventElapsedTime(&milliseconds, start, stop);
//	printf("Time elapsed: %fn\n", milliseconds);

	printf("myhistogram256kernel_02\n");
	perform_histogram_kernel(&myhistogram256kernel_02);

	printf("myhistogram256kernel_03\n");
	perform_histogram_kernel(&myhistogram256kernel_03);

	printf("myhistogram256kernel_04\n");
	perform_histogram_kernel_N(&myhistogram256kernel_04, 2);

}

int main()
{
	main_sub0();

	return EXIT_SUCCESS;
}

