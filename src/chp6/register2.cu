#include <stdio.h>
#include <stdlib.h>
#include <conio.h>

#define ARRAY_SIZE 128

__device__ static u32 d_tmp[NUM_ELEM];

__global__void test_gpu_register(u32 * const data, const u32 num_elements)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(tid < num_elements)
	{
		u32 d_tmp = 0;
		
		for(int i=0;i<KERNEL_LOOP;i++)
		{
			d_tmp |= (packed_array[i] << i);
		}
		
		data[tid] = d_tmp;
	}
}
