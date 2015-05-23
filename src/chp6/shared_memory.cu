#include <stdio.h>
#include <stdlib.h>
#include <conio.h>

__host__ void cpu_sort(u32 * const data, const u32 num_elements)
{
	static u32 cpu_tmp_0[NUM_ELEM];
	static u32 cpu_tmp_1[NUM_ELEM];

	for(u32 bit=0;bit<32;bit++)
	{
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;
		for(u32 i=0; i<num_elements; i++)
		{
			const u32 d = data[i];
			const u32 bit_mask = (1 << bit_mask);
			if((d & bit_mask) > 0)
			{
				cpu_tmp_1[base_cnt_1] = d;
				base_cnt_1++;
			}
			else
			{
				cpu_tmp_0[base_cnt_0] = d;
				base_cnt_0++;
			}
		}
	}

	for(u32 i=0; i<base_cnt_0; i++)
	{
		data[i] = cpu_tmp_0[i];
	}

	// Copy data back to the source - then the one list
	for(u32 i = 0; i<base_cnt_1; i++)
	{
		data[base_cnt_0+i] = cpu_tmp_1[i];
	}
}

__device__ void radix_sort(u32 * const sort_tmp,
				const u32 num_lists,
				const u32 num_elements,
				const u32 tid,
				u32 * const sort_tmp_0,
				u32 * const sort_tmp_1)
{
	//Sort into num_list, listd
	//Apply radix sort on 32 bits of data
	for(u32 bit=0;bit<32;bit++)
	{
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;
	
		for(u32 i=0; i<num_elements; i+=num_lists)
		{
			const u32 elem = sort_tmp[i+tid];
			const u32 bit_mask = (1 << bit);
			if((elem & bit_mask) > 0)
			{
				sort_tmp_1[base_cnt_1+tid] = elem;
				base_cnt_1+=num_lists;
			}
			else
			{
				sort_tmp_0[base_cnt_0+tid] = elem;
				base_cnt_0+=num_lists;
			}
		}
		
		// Copy data back to source - first the zero list
		for(u32 i=0;i<base_cnt_0;i+=num_lists)
		{
			sort_tmp[i+tid] = sort_tmp_0[i+tid];
		}
		
		//Copy data back to source - then the one list
		for(u32 i=0;i<base_cnt_1; i+=num_lists)
		{
			sort_tmp[base_cnt_0+i+tid] = sort_tmp_1[i+tid];
		}
	}
	__syncthreads();
}

__device__ void radix_sort2(u32 * const sort_tmp,
				const u32 num_lists,
				const u32 num_elements,
				const u32 tid,
				u32 * const sort_tmp_1)
{
	//Sort into num_list, listd
	//Apply radix sort on 32 bits of data
	for(u32 bit=0;bit<32;bit++)
	{
		const u32 bit_mask = (1 << bit);
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;
	
		for(u32 i=0; i<num_elements; i+=num_lists)
		{
			const u32 elem = sort_tmp[i+tid];
			if((elem & bit_mask) > 0)
			{
				sort_tmp_1[base_cnt_1+tid] = elem;
				base_cnt_1+=num_lists;
			}
			else
			{
				sort_tmp_0[base_cnt_0+tid] = elem;
				base_cnt_0+=num_lists;
			}
		}
		
		//Copy data back to source - then the one list
		for(u32 i=0;i<base_cnt_1; i+=num_lists)
		{
			sort_tmp[base_cnt_0+i+tid] = sort_tmp_1[i+tid];
		}
	}
	__syncthreads();
}

#define MAX_NUM_LISTS = 2

void merge_array(const u32 * const src_array,
			u32 * const dest_array,
			const u32 num_lists,
			const u32 num_elements)
{
	const u32 num_elements_per_list = (num_elements / num_lists);
	
	u32 list_indexes[MAX_NUM_LISTS];
	
	for(u32 list=0; list < num_lists; list++)
	{
		list_indexes[list] = 0;
	}

	for(u32 i=0; i<num_elements; i++)
	{
		dest_array[i] = find_min(src_array,
					list_indexes,
					num_lists,
					num_elements_per_list);
	}
}

u32 find_min(const u32 * const src_array,
		u32 * const list_indexes,
		const u32 num_lists,
		const u32 num_elements_per_list)
{
	u32 min_val = 0xFFFFFFF;
	u32 min_idx = 0;
	// Iterate over each of the lists
	for(u32 i=0; i<num_lists; i++)
	{
		// If the current list ahs already been emptied
		// then ignore it
		if(list_indexes[i] < num_elements_per_list)
		{
			const u32 src_idx = i + (list_indexes[i] * num_lists);

			const u32 data = src_array[src_idx];
	
			if(data <= min_val)
			{
				min_val = data;
				min_idx = i
			}
		}
	}
	list_indexes[min_idx]++;
	return min_val;
}

__global__ void gpu_sort_array_array(u32 * const data,
					const u32 num_lists,
					const u32 num_elements)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	__shared__ u32 sort_tmp[NUM_ELEM];
	__shared__ u32 sort_tmp_1[NUM_ELEM];

	copy_data_to_shared(data, sort_tmp, num_lists, 
				num_elements, tid);

	radix_sort2(sort_tmp, num_lists, num_elements,
			tid, sort_tmp_1);

	merge_array(sort_tmp, data, num_lists,
			num_elements, tid);
}

