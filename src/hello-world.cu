// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA
// with an array of offsets. Then the offsets are added in parallel
// to produce the string "World!"
// By Ingemar Ragnemalm 2010

// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world

#include <stdio.h>

#define N 16
#define BLOCK_SIZE 16

#define ARRAY_SIZE BLOCK_SIZE
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

/* Declare  statically four arrays of ARRAY_SIZE each */
unsigned int cpu_block[ARRAY_SIZE];
//unsigned int cpu_thread[ARRAY_SIZE];
//unsigned int cpu_warp[ARRAY_SIZE];
//unsigned int cpu_calc_thread[ARRAY_SIZE];

__global__ 
void hello(char *a, int *b) 
{
	a[threadIdx.x] += b[threadIdx.x];
}

__global__
void hello2(int * block)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = threadIdx.x;
}

void main_sub0()
{
	char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	char *ad;
	int *bd;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);

	printf("%s", a);

	cudaMalloc( (void**)&ad, csize ); 
	cudaMalloc( (void**)&bd, isize ); 
	cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ); 
	cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 
	
	dim3 dimBlock( BLOCK_SIZE, 1 );
	dim3 dimGrid( 1, 1 );
	hello<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 
	cudaFree( ad );
	cudaFree( bd );
	
	printf("%s\n", a);
	sleep(1);
}

void main_sub1()
{

	/* Declare pointers for GPU based params */
	int *gpu_block;

	cudaMalloc((void **)&gpu_block, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy( cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

	dim3 dimBlock( BLOCK_SIZE/2, 1 );
	dim3 dimGrid( 2, 1 );

	/* Execute our kernel */
	hello2<<<dimGrid, dimBlock>>>(gpu_block);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy( cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	cudaFree(gpu_block);

	/* Iterate through the arrays and print */
	for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("Calculated Thread: - Block: %2u\n",cpu_block[i]);
	}
}

int main()
{
	main_sub0();
	main_sub1();

	return EXIT_SUCCESS;
}
