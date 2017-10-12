/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/


namespace KokkosSparse{


namespace Impl{


template <typename HandleType,
typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
template <typename c_row_view_t, typename c_lno_nnz_view_t, typename c_scalar_nnz_view_t>
void KokkosSPGEMM
  <HandleType, a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
    b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_>::
    KokkosSPGEMM_multi_mem_numeric(
      c_row_view_t &rowmapC_,
      c_lno_nnz_view_t &entriesC_,
      c_scalar_nnz_view_t &valuesC_){
	  Kokkos::Impl::Timer timer1, timer2;

    //get the algorithm and execution space.
    //SPGEMMAlgorithm spgemm_algorithm = this->handle->get_spgemm_handle()->get_algorithm_type();
    KokkosKernels::Impl::ExecSpaceType my_exec_space = KokkosKernels::Impl::get_exec_space_type<MyExecSpace>();

    if (KOKKOSKERNELS_VERBOSE){
      std::cout << "use_dynamic_schedule:" << use_dynamic_schedule << std::endl;
      std::cout << "MULTIMEM Numeric PHASE" << std::endl;
    }

    /*
    if (spgemm_algorithm == SPGEMM_KK_SPEED)
    {
      this->KokkosSPGEMM_numeric_speed(rowmapC_, entriesC_, valuesC_, my_exec_space);
    }
    else if ( spgemm_algorithm == SPGEMM_KK_COLOR ||
              spgemm_algorithm == SPGEMM_KK_MULTICOLOR ||
              spgemm_algorithm == SPGEMM_KK_MULTICOLOR2){
      this->KokkosSPGEMM_numeric_color(rowmapC_, entriesC_, valuesC_, spgemm_algorithm);
    }
    else if (spgemm_algorithm == SPGEMM_KK_MEMORY2){
        this->KokkosSPGEMM_numeric_hash2(rowmapC_, entriesC_, valuesC_, my_exec_space);
    }
    else if (spgemm_algorithm == SPGEMM_KK_OUTERMULTIMEM ){
      this->KokkosSPGEMM_numeric_outer(rowmapC_, entriesC_, valuesC_, my_exec_space);
    }
    else
    */if (spgemm_algorithm == SPGEMM_KK_MULTIMEMCACHE  ){
    	this->KokkosSPGEMM_numeric_multimem_cache_hash(rowmapC_, entriesC_, valuesC_, my_exec_space);
    }
    else if (spgemm_algorithm == SPGEMM_KK_CACHECACHE ){
    	this->KokkosSPGEMM_numeric_cache_cache_hash(rowmapC_, entriesC_, valuesC_, my_exec_space);
    }
    else if (spgemm_algorithm == SPGEMM_KK_MULTIMEMBBLOCK ){
    	this->KokkosSPGEMM_numeric_multimem_bblock_hash(rowmapC_, entriesC_, valuesC_, my_exec_space);
    }
    else if (spgemm_algorithm == SPGEMM_KK_CACHEBBLOCK  ){
    	this->KokkosSPGEMM_numeric_cache_bblock_hash(rowmapC_, entriesC_, valuesC_, my_exec_space);
    }
    else if (spgemm_algorithm == SPGEMM_KK_MULTIMEMABLOCK){
    	this->KokkosSPGEMM_numeric_multimem_ablock_hash(rowmapC_, entriesC_, valuesC_, my_exec_space);

    }
    else if (spgemm_algorithm == SPGEMM_KK_CACHEABLOCK  ){
    	this->KokkosSPGEMM_numeric_cache_ablock_hash(rowmapC_, entriesC_, valuesC_, my_exec_space);
    }
    else {
      this->KokkosSPGEMM_numeric_hash(rowmapC_, entriesC_, valuesC_, my_exec_space);
    }

  }

template <typename HandleType,
typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
template <typename c_row_view_t>
void KokkosSPGEMM
  <HandleType, a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
    b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_>::
    KokkosSPGEMM_multi_mem_symbolic(c_row_view_t rowmapC_)
  //SPGEMMAlgorithm spgemm_algorithm = this->handle->get_spgemm_handle()->get_algorithm_type();
  {

    Kokkos::Impl::Timer timer1;
    bool apply_compression = true;
	KokkosKernels::Impl::ExecSpaceType my_exec_space = KokkosKernels::Impl::get_exec_space_type<MyExecSpace>();
	if (my_exec_space == KokkosKernels::Impl::Exec_CUDA) {
		apply_compression = true;
	}
    if (apply_compression)
    {
    	//number of rows and nnzs
    	nnz_lno_t n = this->row_mapB.dimension_0() - 1;
    	size_type nnz = this->entriesB.dimension_0();
    	bool compress_in_single_step = this->handle->get_spgemm_handle()->get_compression_step();
    	if (my_exec_space == KokkosKernels::Impl::Exec_CUDA) {
    		compress_in_single_step = true;
    	}
    	//compressed b
    	row_lno_persistent_work_view_t new_row_mapB(Kokkos::ViewAllocateWithoutInitializing("new row map"), n+1);
    	row_lno_persistent_work_view_t new_row_mapB_begins;

    	nnz_lno_persistent_work_view_t set_index_entries; //will be output of compress matrix.
    	nnz_lno_persistent_work_view_t set_entries; //will be output of compress matrix


    	if (KOKKOSKERNELS_VERBOSE){
    		std::cout << "SYMBOLIC PHASE with COMPRESSION " << std::endl;
    	}
    	//First Compress B.

    	if (KOKKOSKERNELS_VERBOSE){
    		std::cout << "\tCOMPRESS MATRIX-B PHASE" << std::endl;
    	}
    	//get the compressed matrix.
    	this->compressMatrix(n, nnz, this->row_mapB, this->entriesB, new_row_mapB, set_index_entries, set_entries, compress_in_single_step);

    	if (KOKKOSKERNELS_VERBOSE){
    		std::cout << "\t\tCOMPRESS MATRIX-B overall time:" << timer1.seconds()
                                				  << std::endl << std::endl;
    	}

    	timer1.reset();

    	//first get the max flops for a row, which will be used for max row size.
    	//If we did compression in single step, row_mapB[i] points the begining of row i,
    	//and new_row_mapB[i] points to the end of row i.
    	nnz_lno_t maxNumRoughZeros = 0;
    	if (compress_in_single_step){
    		maxNumRoughZeros = this->getMaxRoughRowNNZ(a_row_cnt, row_mapA, entriesA, row_mapB, new_row_mapB);
    		if (KOKKOSKERNELS_VERBOSE){
    			std::cout << "\tMax Row Flops:" << maxNumRoughZeros  << std::endl;
    			std::cout << "\tMax Row Flop Calc Time:" << timer1.seconds()  << std::endl;
    		}

    		//calling symbolic structure
    		this->symbolic_c(a_row_cnt, row_mapA, entriesA,
    				row_mapB, new_row_mapB, set_index_entries, set_entries,
					rowmapC_, maxNumRoughZeros);

    	}
    	else {
    		nnz_lno_t begin = 0;
    		auto new_row_mapB_begin = Kokkos::subview (new_row_mapB, std::make_pair (begin, n - 1));
    		auto new_row_mapB_end = Kokkos::subview (new_row_mapB, std::make_pair (begin + 1, n));
    		//KokkosKernels::Impl::print_1Dview(new_row_mapB);
    		//KokkosKernels::Impl::print_1Dview(new_row_mapB_begin);
    		//KokkosKernels::Impl::print_1Dview(new_row_mapB_end);
    		//But for 2 step it is a bit different.
    		//new_row_mapB is complete and holds content of row i is in between new_row_mapB[i] - new_row_mapB[i+1]
    		maxNumRoughZeros = this->getMaxRoughRowNNZ(a_row_cnt, row_mapA, entriesA, new_row_mapB_begin, new_row_mapB_end);
    		if (KOKKOSKERNELS_VERBOSE){
    			std::cout << "\tMax Row Flops:" << maxNumRoughZeros  << std::endl;
    			std::cout << "\tMax Row Flop Calc Time:" << timer1.seconds()  << std::endl;
    			std::cout << "\t Compression Ratio: " << set_index_entries.dimension_0() << " / " << nnz
    					<< " = " << set_index_entries.dimension_0() / double (nnz) << std::endl;
    		}

    		//calling symbolic structure
    		this->symbolic_c(a_row_cnt, row_mapA, entriesA,
    				new_row_mapB_begin, new_row_mapB_end, set_index_entries, set_entries,
					rowmapC_, maxNumRoughZeros);
    	}


    }
    else {

    	//number of rows and nnzs
    	nnz_lno_t n = this->row_mapB.dimension_0() - 1;
    	size_type nnz = this->entriesB.dimension_0();

    	if (KOKKOSKERNELS_VERBOSE){
    		std::cout << "MULTIMEM SYMBOLIC PHASE NO COMPRESSION " << std::endl;
    	}
    	//First Compress B.

    	//first get the max flops for a row, which will be used for max row size.
    	//If we did compression in single step, row_mapB[i] points the begining of row i,
    	//and new_row_mapB[i] points to the end of row i.
    	nnz_lno_t maxNumRoughZeros = 0;
    	nnz_lno_t begin = 0;
    	auto new_row_mapB_begin = Kokkos::subview (this->row_mapB, std::make_pair (begin, n - 1));
    	auto new_row_mapB_end = Kokkos::subview (this->row_mapB, std::make_pair (begin + 1, n));
    	//KokkosKernels::Impl::print_1Dview(new_row_mapB);
    	//KokkosKernels::Impl::print_1Dview(new_row_mapB_begin);
    	//KokkosKernels::Impl::print_1Dview(new_row_mapB_end);
    	//But for 2 step it is a bit different.
    	//new_row_mapB is complete and holds content of row i is in between new_row_mapB[i] - new_row_mapB[i+1]
    	maxNumRoughZeros = this->getMaxRoughRowNNZ(a_row_cnt, row_mapA, entriesA, new_row_mapB_begin, new_row_mapB_end);
    	if (KOKKOSKERNELS_VERBOSE){
    		std::cout << "\tMax Row Flops:" << maxNumRoughZeros  << std::endl;
    		std::cout << "\tMax Row Flop Calc Time:" << timer1.seconds()  << std::endl;
    	}

    	//calling symbolic structure
    	this->symbolic_c_no_compression(a_row_cnt, row_mapA, entriesA,
    			new_row_mapB_begin, new_row_mapB_end, this->entriesB,
				rowmapC_, maxNumRoughZeros);


    }
#ifdef KOKKOSKERNELS_ANALYZE_MEMORYACCESS
    	double read_write_cost = this->handle->get_spgemm_handle()->get_read_write_cost_calc();
    	if (read_write_cost){
    		this->print_read_write_cost(rowmapC_);
    	}
#endif
   timer1.reset();
   switch (this->spgemm_algorithm){
   case SPGEMM_KK_MULTIMEMCACHE:
   case SPGEMM_KK_CACHECACHE:
	   this->prepare_multi_mem_cache();
	   break;
   case SPGEMM_KK_MULTIMEMBBLOCK:
   case SPGEMM_KK_CACHEBBLOCK:
	   this->prepare_multi_mem_block_b();
	   break;
   case SPGEMM_KK_MULTIMEMABLOCK:
   case SPGEMM_KK_CACHEABLOCK:
   	   this->prepare_multi_mem_block_a();
   	   break;
   default:
	   break;
   }

    if (KOKKOSKERNELS_VERBOSE){
    	std::cout << "\t\tSymbolic MultiMEM Calc Time:" << timer1.seconds() << std::endl;
    }
    //3- get max row size of b. maxb.
    //current_row_begin = 0
    //while not done
    	//f
    	//while all rows are processed
    	//4- available rows = fastrows / maxb
    	//fill available rows.
    	//
    //
  }


	template <typename HandleType, typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
	typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >

	size_t KokkosSPGEMM <HandleType, a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
									 b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_>::
	partition_b_sequential(nnz_lno_t current_b_col_begin, nnz_lno_t current_b_col_end,
				const_b_lno_row_view_t row_mapB_, const_b_lno_nnz_view_t entriesB_,
				row_lno_persistent_work_view_t outrowmapB){

		size_type overall_size = 0;
		nnz_lno_t b_rows = row_mapB_.dimension_0() - 1;
		for (nnz_lno_t i = 0; i< b_rows; ++i){
			nnz_lno_t out_row_size = 0;
			size_type row_begin = row_mapB_(i);
			nnz_lno_t row_size = row_mapB_(i + 1) - row_begin;
			for (nnz_lno_t colind = 0; colind < row_size; ++colind){
				nnz_lno_t col = entriesB_(colind + row_begin);
				if (col >= current_b_col_begin && col < current_b_col_end){
					++out_row_size;
				}
			}
			outrowmapB(i) = out_row_size;
			overall_size += out_row_size;
		}
		KokkosKernels::Impl::kk_exclusive_parallel_prefix_sum<row_lno_persistent_work_view_t, MyExecSpace>(b_rows + 1, outrowmapB);
		MyExecSpace::fence();
		return overall_size;

	}

	template <typename HandleType, typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
			  typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
	void KokkosSPGEMM <HandleType, a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
	b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_>::prepare_multi_mem_block_a(){

		nnz_lno_t n = this->row_mapB.dimension_0() - 1;
		//so far we found the output structure.
		//now for multilevel memory algorithm, we want to setup how we will proceed the computation.
		//1- FM = get fast memory size.
		size_t fast_memory_size = this->handle->get_fast_memory_size();


		if (KOKKOSKERNELS_VERBOSE){
			std::cout << "\t\tfast_memory_size:" << fast_memory_size << std::endl;
		}

		// b's row pointers size is br = size_type * (brows + 1) . We need only one as we can compact it.
		size_t b_row_map_sizes = sizeof(size_type) * (this->b_row_cnt + 1);
		// (sizeof (lno_t) + sizeof(scalar_t)) * max_row_size_of_b = pool_chunk_size
		size_t available_size_for_b = fast_memory_size - b_row_map_sizes;

		// (FM - br ) / pool_chunk_size  == fastrows that can fit into fast memory.

		size_type max_num_entries_in_fast_memory = (available_size_for_b / (sizeof (nnz_lno_t) + sizeof(scalar_t)));

		size_type overall_b_size = this->entriesB.dimension_0();
		double b_ratio = static_cast <double> (max_num_entries_in_fast_memory) / overall_b_size ;

		nnz_lno_t num_ideal_b_row_in_fast_memory = static_cast <nnz_lno_t> (b_ratio * this->b_row_cnt);
		nnz_lno_t num_partitions = static_cast <nnz_lno_t> (1.0 / b_ratio) + 1;


		if (KOKKOSKERNELS_VERBOSE){
			std::cout << "\t\tnum_ideal_b_columns_in_fast_memory:" << num_ideal_b_row_in_fast_memory << std::endl;
			std::cout << "\t\tnum_partitions:" << num_partitions << std::endl;
		}

		nnz_lno_t current_a_row_begin = 0 ;


		std::vector<nnz_lno_t> multi_mem_b_column_ranges;
		multi_mem_b_column_ranges.push_back(0);


		size_type size_on_the_left = 0;


		size_type max_required_size = 0;
		for (int counter = 0; ; ++counter)
		{
			nnz_lno_t b_row_begin_to_test = multi_mem_b_column_ranges[counter];
			nnz_lno_t b_row_end_to_test = b_row_begin_to_test + num_ideal_b_row_in_fast_memory;
			b_row_end_to_test = KOKKOSKERNELS_MACRO_MIN(b_row_end_to_test, this->b_row_cnt);


			size_type lower_bound = size_on_the_left;
			nnz_lno_t lower_bound_column = b_row_begin_to_test;

			size_type upper_bound = overall_b_size;
			nnz_lno_t upper_bound_column = this->b_col_cnt;

			bool done = false;
			size_type b_size_for_the_range = 0;
			while (!done){

				b_size_for_the_range = this->row_mapB(b_row_end_to_test) - size_on_the_left;
#if 0
				if (KOKKOSKERNELS_VERBOSE){
					std::cout << "\n\t\t\tb_row_begin_to_test:" << b_row_begin_to_test << " b_row_end_to_test:" << b_row_end_to_test << std::endl;
					std::cout << "\t\t\tb_size_for_the_range:" << b_size_for_the_range << " max_num_entries_in_fast_memory:" << max_num_entries_in_fast_memory << std::endl;
					std::cout << "\t\t\tlower_bound_column:" << lower_bound_column << " lower_bound:" << lower_bound << std::endl;
					std::cout << "\t\t\tupper_bound_column:" << upper_bound_column << " upper_bound:" << upper_bound << std::endl;
				}
#endif
				if (b_size_for_the_range < max_num_entries_in_fast_memory){
					lower_bound_column = b_row_end_to_test;
					lower_bound = b_size_for_the_range + size_on_the_left;

					if (upper_bound_column == lower_bound_column ){
						break;
					}

					nnz_lno_t shift = ((upper_bound_column - lower_bound_column) / static_cast <double>(upper_bound - lower_bound))
													* (max_num_entries_in_fast_memory - b_size_for_the_range);
					if (shift == 0) shift = 1;
					if (shift == upper_bound_column - lower_bound_column) break;

					//now we need to increase
					b_row_end_to_test = shift + lower_bound_column;
				}
				else if (b_size_for_the_range > max_num_entries_in_fast_memory){

					upper_bound_column = b_row_end_to_test;
					upper_bound = b_size_for_the_range + size_on_the_left;

					if (upper_bound_column <= lower_bound_column +1){
						b_size_for_the_range = lower_bound;
						b_row_end_to_test = lower_bound_column;
						break;
					}

					nnz_lno_t shift = ((upper_bound_column - lower_bound_column) / static_cast <double>(upper_bound - lower_bound))
													* (b_size_for_the_range - max_num_entries_in_fast_memory);
					if (shift == 0) shift = 1;
					if (shift == upper_bound_column - lower_bound_column) break;
					b_row_end_to_test = -shift + upper_bound_column;
				}
				else {
					break;
				}
			}


			if (KOKKOSKERNELS_VERBOSE){
				std::cout << "\t\tPart:" << counter << " col_end_to_test:" << b_row_end_to_test << " b_size_for_the_range:" << b_size_for_the_range << std::endl;
				std::cout << "\t\tmax_num_entries_in_fast_memory:" << max_num_entries_in_fast_memory << std::endl;
			}
			if (this->row_mapB(b_row_end_to_test) - size_on_the_left > max_required_size) max_required_size = this->row_mapB(b_row_end_to_test) - size_on_the_left;
			size_on_the_left = this->row_mapB(b_row_end_to_test);
			multi_mem_b_column_ranges.push_back(b_row_end_to_test);

			if (b_row_end_to_test == this->b_col_cnt) break;
		}

		this->handle->get_spgemm_handle()->multi_mem_ranges = multi_mem_b_column_ranges;
		this->handle->get_spgemm_handle()->max_b_size_in_fast_memory = max_required_size;
		/*
		this->handle->get_spgemm_handle()->pool_reverse_pointers = b_rowmap_entries;
		this->handle->get_spgemm_handle()->multi_mem_ranges = multi_mem_b_column_ranges;
		this->handle->get_spgemm_handle()->max_b_row_size = max_brow_size;
		this->handle->get_spgemm_handle()->num_rows_of_in_fast_memory = num_rows_of_in_fast_memory;
		 */
}


	template <typename HandleType, typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
			  typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
	void KokkosSPGEMM <HandleType, a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
	b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_>::prepare_multi_mem_block_b(){

		nnz_lno_t n = this->row_mapB.dimension_0() - 1;
		//so far we found the output structure.
		//now for multilevel memory algorithm, we want to setup how we will proceed the computation.
		//1- FM = get fast memory size.
		size_t fast_memory_size = this->handle->get_fast_memory_size();


		if (KOKKOSKERNELS_VERBOSE){
			std::cout << "\t\tfast_memory_size:" << fast_memory_size << std::endl;
		}

		// b's row pointers size is br = size_type * (brows + 1) . We need only one as we can compact it.
		size_t b_row_map_sizes = sizeof(size_type) * (this->b_row_cnt + 1);
		// (sizeof (lno_t) + sizeof(scalar_t)) * max_row_size_of_b = pool_chunk_size
		size_t available_size_for_b = fast_memory_size - b_row_map_sizes;

		// (FM - br ) / pool_chunk_size  == fastrows that can fit into fast memory.

		size_type max_num_entries_in_fast_memory = (available_size_for_b / (sizeof (nnz_lno_t) + sizeof(scalar_t)));

		size_type overall_b_size = this->entriesB.dimension_0();
		double b_ratio = static_cast <double> (max_num_entries_in_fast_memory) / overall_b_size ;

		nnz_lno_t num_ideal_b_columns_in_fast_memory = static_cast <nnz_lno_t> (b_ratio * this->b_col_cnt);
		nnz_lno_t num_partitions = static_cast <nnz_lno_t> (1.0 / b_ratio) + 1;


		if (KOKKOSKERNELS_VERBOSE){
			std::cout << "\t\tnum_ideal_b_columns_in_fast_memory:" << num_ideal_b_columns_in_fast_memory << std::endl;
			std::cout << "\t\tnum_partitions:" << num_partitions << std::endl;
		}

		nnz_lno_t current_a_row_begin = 0 ;


		std::vector<row_lno_persistent_work_view_t> b_rowmap_entries;
		std::vector<nnz_lno_t> multi_mem_b_column_ranges;
		multi_mem_b_column_ranges.push_back(0);

		row_lno_persistent_work_view_t prev_b_rowmap_entry("b_rowmap entry", b_row_cnt + 1);
		row_lno_persistent_work_view_t tmp_b_rowmap_entry;


		size_type size_on_the_left = 0;

		size_type max_required_size = 0;

		for (int counter = 0; ; ++counter)
		{
			nnz_lno_t col_begin_to_test = multi_mem_b_column_ranges[counter];
			nnz_lno_t col_end_to_test = col_begin_to_test + num_ideal_b_columns_in_fast_memory;
			col_end_to_test = KOKKOSKERNELS_MACRO_MIN(col_end_to_test, this->b_col_cnt);


			size_type lower_bound = size_on_the_left;
			nnz_lno_t lower_bound_column = col_begin_to_test;

			size_type upper_bound = overall_b_size;
			nnz_lno_t upper_bound_column = this->b_col_cnt;


			row_lno_persistent_work_view_t b_rowmap_entry("b_rowmap entry", b_row_cnt + 1);


			MyExecSpace::fence();

			bool done = false;
			size_type b_size_for_the_range = 0;
			while (!done){

				b_size_for_the_range =
						partition_b_sequential(
								col_begin_to_test, col_end_to_test,
								row_mapB, entriesB,
								b_rowmap_entry);
#if 0
				if (KOKKOSKERNELS_VERBOSE){
					std::cout << "\n\t\t\tcol_begin_to_test:" << col_begin_to_test << " col_end_to_test:" << col_end_to_test << std::endl;
					std::cout << "\t\t\tb_size_for_the_range:" << b_size_for_the_range << " max_num_entries_in_fast_memory:" << max_num_entries_in_fast_memory << std::endl;
					std::cout << "\t\t\tlower_bound_column:" << lower_bound_column << " lower_bound:" << lower_bound << std::endl;
					std::cout << "\t\t\tupper_bound_column:" << upper_bound_column << " upper_bound:" << upper_bound << std::endl;
				}
#endif
				if (b_size_for_the_range < max_num_entries_in_fast_memory){
					lower_bound_column = col_end_to_test;
					lower_bound = b_size_for_the_range + size_on_the_left;

					if (upper_bound_column == lower_bound_column ){
						break;
					}

					nnz_lno_t shift = ((upper_bound_column - lower_bound_column) / static_cast <double>(upper_bound - lower_bound))
											* (max_num_entries_in_fast_memory - b_size_for_the_range);
					if (shift == 0) shift = 1;
					if (shift == upper_bound_column - lower_bound_column) break;

					tmp_b_rowmap_entry = prev_b_rowmap_entry;
					prev_b_rowmap_entry = b_rowmap_entry;
					b_rowmap_entry = tmp_b_rowmap_entry;
					//now we need to increase
					col_end_to_test = shift + lower_bound_column;
				}
				else if (b_size_for_the_range > max_num_entries_in_fast_memory){

					upper_bound_column = col_end_to_test;
					upper_bound = b_size_for_the_range + size_on_the_left;

					if (upper_bound_column <= lower_bound_column +1){
						tmp_b_rowmap_entry = prev_b_rowmap_entry;
						prev_b_rowmap_entry = b_rowmap_entry;
						b_rowmap_entry = tmp_b_rowmap_entry;
						b_size_for_the_range = lower_bound;
						col_end_to_test = lower_bound_column;
						break;
					}

					nnz_lno_t shift = ((upper_bound_column - lower_bound_column) / static_cast <double>(upper_bound - lower_bound))
											* (b_size_for_the_range - max_num_entries_in_fast_memory);
					if (shift == 0) shift = 1;
					if (shift == upper_bound_column - lower_bound_column) break;
					col_end_to_test = -shift + upper_bound_column;
				}
				else {
					break;
				}
			}

			if (b_size_for_the_range > max_required_size) max_required_size = b_size_for_the_range;

			if (KOKKOSKERNELS_VERBOSE){
				std::cout << "\t\tPart:" << counter << " col_end_to_test:" << col_end_to_test << " b_size_for_the_range:" << b_size_for_the_range << std::endl;
				std::cout << "\t\tmax_num_entries_in_fast_memory:" << max_num_entries_in_fast_memory << std::endl;
			}

			multi_mem_b_column_ranges.push_back(col_end_to_test);
			b_rowmap_entries.push_back(b_rowmap_entry);
			if (col_end_to_test == this->b_col_cnt) break;
		}

		this->handle->get_spgemm_handle()->b_rowmap_pointers = b_rowmap_entries;
		this->handle->get_spgemm_handle()->multi_mem_ranges = multi_mem_b_column_ranges;
		this->handle->get_spgemm_handle()->max_b_size_in_fast_memory = max_required_size;

		/*
		this->handle->get_spgemm_handle()->pool_reverse_pointers = b_rowmap_entries;
		this->handle->get_spgemm_handle()->multi_mem_ranges = multi_mem_b_column_ranges;
		this->handle->get_spgemm_handle()->max_b_row_size = max_brow_size;
		this->handle->get_spgemm_handle()->num_rows_of_in_fast_memory = num_rows_of_in_fast_memory;
		*/
}




  template <typename HandleType, typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
            typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
  void KokkosSPGEMM <HandleType, a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
                   b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_>::prepare_multi_mem_cache(){

	    nnz_lno_t n = this->row_mapB.dimension_0() - 1;
	    //so far we found the output structure.
	    //now for multilevel memory algorithm, we want to setup how we will proceed the computation.
	    //1- FM = get fast memory size.
	    size_t fast_memory_size = this->handle->get_fast_memory_size();


	    if (KOKKOSKERNELS_VERBOSE){
	    	std::cout << "\t\tfast_memory_size:" << fast_memory_size << std::endl;
	    }
	    //2- get maximum row size of B.
	    size_type max_brow_size = 0;
	    KokkosKernels::Impl::view_reduce_maxsizerow<const_b_lno_row_view_t, MyExecSpace>(this->b_row_cnt, row_mapB, max_brow_size);
	    MyExecSpace::fence();
	    //2- get maximum row size of A.
	    //std::cout << "\t\t rowmap a"; KokkosKernels::Impl::print_1Dview(row_mapA);

	    //2- get maximum row size of A.
	    //std::cout << "\t\t rowmap b"; KokkosKernels::Impl::print_1Dview(row_mapB);
	    size_type max_arow_size = 0;
	    KokkosKernels::Impl::view_reduce_maxsizerow<const_a_lno_row_view_t, MyExecSpace>(this->a_row_cnt, row_mapA, max_arow_size);
	    MyExecSpace::fence();

	    if (KOKKOSKERNELS_VERBOSE){
	    	std::cout << "\t\tmax_brow_size:" << max_brow_size << std::endl;
	    	std::cout << "\t\tmax_arow_size:" << max_arow_size << std::endl;
	    }

	    // b's row pointers size is br = size_type * (brows + 1) * 2. 1 for begin, 1 for end.
	    size_t b_row_map_sizes = sizeof(size_type) * (this->b_row_cnt + 1) * 2;
	    // (sizeof (lno_t) + sizeof(scalar_t)) * max_row_size_of_b = pool_chunk_size
	    size_t available_size_for_b = fast_memory_size - b_row_map_sizes;

	    // (FM - br ) / pool_chunk_size  == fastrows that can fit into fast memory.

	    nnz_lno_t num_rows_of_in_fast_memory = (available_size_for_b / (sizeof (nnz_lno_t) + sizeof(scalar_t)))/ max_brow_size;

	    num_rows_of_in_fast_memory = KOKKOSKERNELS_MACRO_MIN(num_rows_of_in_fast_memory, b_row_cnt);



	    if (KOKKOSKERNELS_VERBOSE){
	    	std::cout << "\t\tnum_rows_of_b_in_fast_memory:" << num_rows_of_in_fast_memory << std::endl;
	    }
	    nnz_lno_t current_a_row_begin = 0 ;

	    int counter = 0;
	    bool alldone = false;
	    std::vector<nnz_lno_persistent_work_view_t> pool_reverse_pointers;
	    std::vector<nnz_lno_t> multi_mem_ranges;
	    multi_mem_ranges.push_back(0);
	    nnz_lno_persistent_work_view_t previous_row_to_pool_index;

	    while (!alldone)
	    {
#if 0
	    	std::cout << "counter:" << counter << std::endl;
#endif
	        nnz_lno_t available_fast_memory_row_count = num_rows_of_in_fast_memory;
	    	nnz_lno_t min_space_for_a_rows = available_fast_memory_row_count / max_arow_size;


	    	nnz_lno_t current_a_row_end = 0; //KOKKOSKERNELS_MACRO_MIN(current_a_row_begin + min_space_for_a_rows, a_row_cnt);

	    	nnz_lno_persistent_work_view_t row_to_pool_index(Kokkos::ViewAllocateWithoutInitializing("b_row_to_pool"), b_row_cnt);
	    	nnz_lno_persistent_work_view_t pool_to_row_index(Kokkos::ViewAllocateWithoutInitializing("pool_to_brow"), num_rows_of_in_fast_memory);
	    	Kokkos::deep_copy(row_to_pool_index, -1);
	    	Kokkos::deep_copy(pool_to_row_index, -1);
	    	MyExecSpace::fence();


			nnz_lno_t num_alloc = 0;
#if 0
    		std::cout << "current_a_row_begin:" << current_a_row_begin << " a_row_cnt:" << a_row_cnt << " num_alloc:" << num_alloc
    				<< " available_fast_memory_row_count:" << available_fast_memory_row_count << std::endl;
#endif
    		bool space_available_in_fast_memory = true;
			for (current_a_row_end = current_a_row_begin; current_a_row_end < a_row_cnt ; ++current_a_row_end){
				size_type a_row_begin = row_mapA(current_a_row_end);
				nnz_lno_t row_size =  row_mapA(current_a_row_end + 1) - a_row_begin;
				for (nnz_lno_t j = 0; j < row_size; ++j){
					nnz_lno_t b_row = entriesA(j + a_row_begin);
					if (row_to_pool_index(b_row) == -1)
					{
						if (num_alloc >= num_rows_of_in_fast_memory){
							space_available_in_fast_memory = false;
							break;
						}
						pool_to_row_index[num_alloc] = b_row;
						row_to_pool_index[b_row] = num_alloc++;
					}

				}
				if (!space_available_in_fast_memory) break;
			}
			current_a_row_begin = current_a_row_end;
#if 0

			std::cout << "current_a_row_begin:" << current_a_row_begin << std::endl;
			std::cout << "num_alloc:" << num_alloc << std::endl;
#endif
	    	/*
	    	while (min_space_for_a_rows > 0){

	    		std::cout << "current_a_row_begin:" << current_a_row_begin << " current_a_row_end:" << current_a_row_end
	    				<< " available_fast_memory_row_count:" << available_fast_memory_row_count << std::endl;

	    		nnz_lno_t allocated_row_count = fill_rows_of_pool_sequential(
	    				current_a_row_begin, current_a_row_end,
						row_mapA, entriesA, row_mapB, entriesB,
						row_to_pool_index, pool_to_row_index);

	    		std::cout << "allocated_row_count:" << allocated_row_count << std::endl;

	    		available_fast_memory_row_count -=  allocated_row_count;
	    		std::cout << "available_fast_memory_row_count:" << available_fast_memory_row_count << std::endl;
	    		if (current_a_row_end == a_row_cnt){
	    			alldone = true;
	    			break;
	    		}
	    		min_space_for_a_rows = available_fast_memory_row_count / max_arow_size;
	    		current_a_row_begin = current_a_row_end;
	    		current_a_row_end += min_space_for_a_rows;
	    		current_a_row_end = KOKKOSKERNELS_MACRO_MIN(current_a_row_end, a_row_cnt);
	    	}
	    	*/



	    	multi_mem_ranges.push_back(current_a_row_end);
	    	//now store it in a vector.
	    	if (counter++ == 0){
	    		pool_reverse_pointers.push_back(pool_to_row_index);
	    	}
	    	else {
	    		//but order it to minimize the copy by checking the previous state of the fast memory.

		    	if (KOKKOSKERNELS_VERBOSE){
		    		std::cout << "\t\tSTEP:" << counter - 1  << " ";
		    	}
	    		maximize_overlap(num_rows_of_in_fast_memory, row_to_pool_index, pool_to_row_index, previous_row_to_pool_index);
	    		pool_reverse_pointers.push_back(pool_to_row_index);

	    	}

#if 0
	    	if (KOKKOSKERNELS_VERBOSE){
	    		std::cout << "\t\tPOOL AND ROWS FOR STEP:" << counter << std::endl;
	    		std::cout << "\t\t"; KokkosKernels::Impl::print_1Dview(row_to_pool_index);
	    		std::cout << "\t\t"; KokkosKernels::Impl::print_1Dview(pool_to_row_index);
	    		std::cout << "\t\t#######################" << std::endl;
	    	}
#endif

	    	previous_row_to_pool_index = row_to_pool_index;
	    	if (current_a_row_end >= a_row_cnt) alldone = true;
	    }

	    this->handle->get_spgemm_handle()->pool_reverse_pointers = pool_reverse_pointers;
	    this->handle->get_spgemm_handle()->multi_mem_ranges = multi_mem_ranges;
	    this->handle->get_spgemm_handle()->max_b_row_size = max_brow_size;
	    this->handle->get_spgemm_handle()->num_rows_of_in_fast_memory = num_rows_of_in_fast_memory;
  }

  template <typename HandleType, typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
            typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
  template <typename row_to_pool_view_t, typename pool_to_row_view_t>
  void KokkosSPGEMM <HandleType, a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
                     b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_>::
   maximize_overlap(nnz_lno_t available_fast_memory_row_count,
			  	  	  	  row_to_pool_view_t row_to_pool_index, pool_to_row_view_t pool_to_row_index,
						  row_to_pool_view_t previous_row_to_pool_index){
	  nnz_lno_t num_avoid = 0;
	  for (nnz_lno_t i = 0; i < available_fast_memory_row_count;){
		  nnz_lno_t real_row = pool_to_row_index(i);
		  if (real_row == -1) {
			  ++i;
			  continue;
		  }
		  nnz_lno_t previous_pool_index = previous_row_to_pool_index(real_row);
		  if (previous_pool_index == -1 || previous_pool_index == i) {
			  previous_row_to_pool_index(real_row) = -1;
			  ++i;
			  continue;
		  }
		  ++num_avoid;
		  nnz_lno_t row_to_swap = pool_to_row_index(previous_pool_index);
		  pool_to_row_index(previous_pool_index) = -1;
		  pool_to_row_index(i) = row_to_swap;
		  row_to_pool_index(real_row) = previous_pool_index;
		  row_to_pool_index(row_to_swap) = i;
	  }

	  if (KOKKOSKERNELS_VERBOSE){
		  std::cout << "num_avoid:" << num_avoid << " available_fast_memory_row_count:" << available_fast_memory_row_count << std::endl;
	  }
  }

  template <typename HandleType, typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
            typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
  template <typename row_to_pool_view_t, typename pool_to_row_view_t, typename fast_memory_lno_view_t, typename fast_memory_scalar_view_t>
  struct  KokkosSPGEMM <HandleType, a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
  	  	  	  	  b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_>::
				  FastMemoryCopier{
	  nnz_lno_t _b_max_row_size;
	  nnz_lno_t _team_work_size;
	  nnz_lno_t _available_fast_memory_row_count;
	  row_to_pool_view_t _row_to_pool_index_begin; row_to_pool_view_t _row_to_pool_index_end;
	  pool_to_row_view_t _current_pool_reverse_pointers;
	  fast_memory_lno_view_t _b_pool_entries; fast_memory_scalar_view_t _b_pool_values;
	  const_b_lno_row_view_t _rowmapB;
	  const_b_lno_nnz_view_t _entriesB;
	  const_b_scalar_nnz_view_t _valsB;

	  FastMemoryCopier (
			  nnz_lno_t b_max_row_size_,
			  nnz_lno_t team_work_size_,
			  nnz_lno_t available_fast_memory_row_count_,
			  row_to_pool_view_t row_to_pool_index_begin_, row_to_pool_view_t row_to_pool_index_end_,
			  pool_to_row_view_t current_pool_reverse_pointers_,
			  fast_memory_lno_view_t b_pool_entries_, fast_memory_scalar_view_t b_pool_values_,
			  const_b_lno_row_view_t rowmapB_, const_b_lno_nnz_view_t entriesB_,   const_b_scalar_nnz_view_t valsB_):
				  _b_max_row_size(b_max_row_size_),
				  _team_work_size(team_work_size_),
				  _available_fast_memory_row_count(available_fast_memory_row_count_),
				  _row_to_pool_index_begin(row_to_pool_index_begin_),
				  _row_to_pool_index_end(row_to_pool_index_end_),
				  _current_pool_reverse_pointers(current_pool_reverse_pointers_),
				  _b_pool_entries(b_pool_entries_),
				  _b_pool_values(b_pool_values_),
				  _rowmapB(rowmapB_),
				  _entriesB(entriesB_),
				  _valsB(valsB_){

	  }


	  KOKKOS_INLINE_FUNCTION
	  void operator()(const team_member_t & teamMember) const {

		  const nnz_lno_t team_row_begin = teamMember.league_rank() * _team_work_size;
		  const nnz_lno_t team_row_end = KOKKOSKERNELS_MACRO_MIN(team_row_begin + _team_work_size, _available_fast_memory_row_count);

		  Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, team_row_begin, team_row_end), [&] (const nnz_lno_t& fast_row_index) {
			  nnz_lno_t row_index = _current_pool_reverse_pointers(fast_row_index);
			  if (row_index >= 0)  {
				  const size_type col_begin = _rowmapB[row_index];
				  const nnz_lno_t left_work = _rowmapB[row_index + 1] - col_begin;
				  size_type pool_begin = fast_row_index * _b_max_row_size;
				  size_type pool_end = left_work + pool_begin;
				  Kokkos::single(Kokkos::PerThread(teamMember),[&] () {
					  _row_to_pool_index_begin(row_index) = pool_begin;
					  _row_to_pool_index_end(row_index) = pool_end;
				  });
				  Kokkos::parallel_for( Kokkos::ThreadVectorRange(teamMember, left_work),
						  [&] (nnz_lno_t work_to_handle) {
					  _b_pool_entries(pool_begin + work_to_handle) = _entriesB(col_begin + work_to_handle);
					  _b_pool_values(pool_begin + work_to_handle) = _valsB(col_begin + work_to_handle);
				  });
			  }
		  });
	  }
  };



  template <typename HandleType, typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
            typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
  template <typename fast_memory_row_view_t, typename fast_memory_lno_view_t, typename fast_memory_scalar_view_t>
  struct  KokkosSPGEMM <HandleType, a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
  	  	  	  	  b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_>::
				  FastMemoryCopierBBlock{
	  nnz_lno_t _team_work_size;
	  nnz_lno_t _b_row_count;
	  nnz_lno_t _b_col_begin, _b_col_end;
	  fast_memory_row_view_t _b_fast_rowmap;
	  fast_memory_lno_view_t _b_pool_entries; fast_memory_scalar_view_t _b_pool_values;
	  const_b_lno_row_view_t _rowmapB;
	  const_b_lno_nnz_view_t _entriesB;
	  const_b_scalar_nnz_view_t _valsB;
	  KokkosKernels::Impl::ExecSpaceType _MyEnumExecSpace;

	  FastMemoryCopierBBlock (
	 		  nnz_lno_t team_work_size_,
			  nnz_lno_t b_row_count,
	 		  nnz_lno_t b_col_begin, nnz_lno_t b_col_end,
	 		  fast_memory_row_view_t row_mapB_fast,
	 		  fast_memory_lno_view_t b_pool_entries_, fast_memory_scalar_view_t b_pool_values_,
	 		  const_b_lno_row_view_t rowmapB_, const_b_lno_nnz_view_t entriesB_,   const_b_scalar_nnz_view_t valsB_,
			  KokkosKernels::Impl::ExecSpaceType MyEnumExecSpace_):
				  _team_work_size(team_work_size_),
				  _b_row_count(b_row_count),
				  _b_col_begin(b_col_begin),
				  _b_col_end(b_col_end),
				  _b_fast_rowmap(row_mapB_fast),
				  _b_pool_entries(b_pool_entries_),
				  _b_pool_values(b_pool_values_),
				  _rowmapB(rowmapB_),
				  _entriesB(entriesB_),
				  _valsB(valsB_), _MyEnumExecSpace(MyEnumExecSpace_){

	  }


	  KOKKOS_INLINE_FUNCTION
	  void operator()(const team_member_t & teamMember) const {

		  const nnz_lno_t team_row_begin = teamMember.league_rank() * _team_work_size;
		  const nnz_lno_t team_row_end = KOKKOSKERNELS_MACRO_MIN(team_row_begin + _team_work_size, _b_row_count);
		  nnz_lno_t *my_row_size = (nnz_lno_t *) (teamMember.team_shmem().get_shmem(teamMember.team_size() * sizeof (nnz_lno_t)));
		  my_row_size += sizeof (nnz_lno_t) * teamMember.team_rank();



		  Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, team_row_begin, team_row_end), [&] (const nnz_lno_t& row_index) {

				  const size_type col_begin = _rowmapB[row_index];
				  const nnz_lno_t left_work = _rowmapB[row_index + 1] - col_begin;

				  size_type pool_begin = _b_fast_rowmap(row_index);
				  size_type pool_end = _b_fast_rowmap(row_index + 1);
				  Kokkos::single(Kokkos::PerThread(teamMember),[&] () {
					  my_row_size[0] = 0;
				  });
				  Kokkos::parallel_for( Kokkos::ThreadVectorRange(teamMember, left_work),
						  [&] (nnz_lno_t work_to_handle) {
					  nnz_lno_t col =  _entriesB(col_begin + work_to_handle);
					  if (col >= _b_col_begin && col < _b_col_end){
						  nnz_lno_t wind = 0;
						  switch (_MyEnumExecSpace){
						  case KokkosKernels::Impl::Exec_CUDA:
						  	  wind = Kokkos::atomic_fetch_add(my_row_size , 1);
						  	  break;
						  default:
							  wind = my_row_size[0]++;
							  break;
						  }
						  _b_pool_entries(pool_begin + wind) = col;
						  _b_pool_values(pool_begin + wind) = _valsB(col_begin + work_to_handle);
					  }

				  });

		  });
	  }

	  size_t team_shmem_size (int team_size) const {
	    return team_size * sizeof (nnz_lno_t);
	  }
  };

  template <typename HandleType, typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
            typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
  template <typename fast_memory_row_view_t, typename fast_memory_lno_view_t, typename fast_memory_scalar_view_t>
  void KokkosSPGEMM <HandleType, a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
                     b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_>::
					 fill_fast_memory_bblock(
					 		  int suggested_team_size, int suggested_vector_size,
					 		  nnz_lno_t team_work_size_,
					 		  nnz_lno_t b_col_begin, nnz_lno_t b_col_end,
					 		  fast_memory_row_view_t row_mapB_fast,
					 		  fast_memory_lno_view_t b_pool_entries_, fast_memory_scalar_view_t b_pool_values_,
					  		  const_b_lno_row_view_t rowmapB_, const_b_lno_nnz_view_t entriesB_,   const_b_scalar_nnz_view_t valsB_){
	  FastMemoryCopierBBlock <fast_memory_row_view_t, fast_memory_lno_view_t,  fast_memory_scalar_view_t> fmc(
			  team_work_size_,
			  row_mapB_fast.dimension_0()-1,
			  b_col_begin, b_col_end,
			  row_mapB_fast,
			  b_pool_entries_, b_pool_values_,
			  rowmapB_, entriesB_,   valsB_, this->MyEnumExecSpace) ;

      Kokkos::parallel_for( team_policy_t((row_mapB_fast.dimension_0()-1) / team_work_size_ + 1 , suggested_team_size, suggested_vector_size), fmc);
      MyExecSpace::fence();
  }

  template <typename HandleType, typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
            typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
  template <typename row_to_pool_view_t, typename pool_to_row_view_t, typename fast_memory_lno_view_t, typename fast_memory_scalar_view_t>
  void KokkosSPGEMM <HandleType, a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
                     b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_>::
   fill_fast_memory(
		   	   int suggested_team_size, int suggested_vector_size,
			  nnz_lno_t b_max_row_size_,
			  nnz_lno_t team_work_size_,
			  nnz_lno_t available_fast_memory_row_count_,
			  row_to_pool_view_t row_to_pool_index_begin_, row_to_pool_view_t row_to_pool_index_end_,
			  pool_to_row_view_t current_pool_reverse_pointers_,
			  fast_memory_lno_view_t b_pool_entries_, fast_memory_scalar_view_t b_pool_values_,
			  const_b_lno_row_view_t rowmapB_, const_b_lno_nnz_view_t entriesB_,   const_b_scalar_nnz_view_t valsB_){

	  FastMemoryCopier<row_to_pool_view_t, pool_to_row_view_t, fast_memory_lno_view_t,  fast_memory_scalar_view_t> fmc(
			  b_max_row_size_,
			  team_work_size_,
			  available_fast_memory_row_count_,
			  row_to_pool_index_begin_, row_to_pool_index_end_,
			  current_pool_reverse_pointers_,
			  b_pool_entries_, b_pool_values_,
			  rowmapB_, entriesB_,   valsB_) ;

      Kokkos::parallel_for( team_policy_t(available_fast_memory_row_count_ / team_work_size_ + 1 , suggested_team_size, suggested_vector_size), fmc);
      MyExecSpace::fence();
  }
/*

	template <typename HandleType,
			  typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
			  typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
	struct KokkosSPGEMM <HandleType, a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
						   b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_>::
						   FillFastMemory{


		const_a_lno_row_view_t row_mapA_;
		const_a_lno_nnz_view_t entriesA_;
		const_b_lno_row_view_t row_mapB_;
		const_b_lno_nnz_view_t entriesB_;
		nnz_lno_t *prow_to_pool_index;
		nnz_lno_t *ppool_to_row_index;
		nnz_lno_t num_slots_;
		nnz_lno_t team_row_chunk_size_;
		nnz_lno_t row_end_index_; //num rows

		FillFastMemory(
				const_a_lno_row_view_t row_mapA__, const_a_lno_nnz_view_t entriesA__,
				const_b_lno_row_view_t row_mapB__, const_b_lno_nnz_view_t entriesB__,
				nnz_lno_t *prow_to_pool_index_, nnz_lno_t *pool_to_row_index_, nnz_lno_t num_slots__,
				nnz_lno_t team_row_chunk_size__, nnz_lno_t row_end_index__):
					row_mapA_(row_mapA__), entriesA_(entriesA__), row_mapB_(row_mapB__), entriesB_(entriesB__),
					prow_to_pool_index(prow_to_pool_index_), ppool_to_row_index(pool_to_row_index_),
					num_slots_(num_slots__), team_row_chunk_size_(team_row_chunk_size__), row_end_index_(row_end_index__){

		}


		KOKKOS_INLINE_FUNCTION
		void operator()(const team_member_t & teamMember, nnz_lno_t &overal_alloc) const {

			//get the range of rows for team.
			const nnz_lno_t team_row_begin = teamMember.league_rank() * team_row_chunk_size_;
			const nnz_lno_t team_row_end = KOKKOSKERNELS_MACRO_MIN(team_row_begin + team_row_chunk_size_, row_end_index_);
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, team_row_begin, team_row_end), [&] (const nnz_lno_t& row_index)
					{

				size_type a_row_begin = row_mapA_(row_index);
				nnz_lno_t row_size =  row_mapA_(row_index + 1) - a_row_begin;
				for (nnz_lno_t j = 0; j < row_size; ++j){
					nnz_lno_t a_col = entriesA_(j + a_row_begin);

					size_type b_row_begin = row_mapB_(a_col);
					nnz_lno_t brow_size =  row_mapB_(a_col + 1) - b_row_begin;

					Kokkos::parallel_reduce( Kokkos::ThreadVectorRange(teamMember, brow_size),
							[&] (const nnz_lno_t k, nnz_lno_t &num_alloc) {

						nnz_lno_t b_col = entriesB_(k + b_row_begin);

						if (row_to_pool_index(b_col) == -1 &&
								Kokkos::atomic_compare_exchange_strong(prow_to_pool_index + b_col, -1, b_col))
						{
							++num_alloc;
							nnz_lno_t b_search_index = b_col;
							bool run_second_phase = false;
							while(!Kokkos::atomic_compare_exchange_strong(ppool_to_row_index + b_search_index, -1, b_col)){
								b_search_index++;
								if (b_search_index < num_slots_) {
									run_second_phase = true;
									break;
								}
							}
							b_search_index = 0;
							while(run_second_phase && !Kokkos::atomic_compare_exchange_strong(ppool_to_row_index + b_search_index, -1, b_col)){
								b_search_index++;
							}
							row_to_pool_index(b_col) = b_search_index;
						}
					});
				}
			});
		}
	};

	template <	typename HandleType, typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
				typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
	template <typename row_to_pool_view_t, typename pool_to_row_view_t>
	size_t KokkosSPGEMM <HandleType, a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
						 b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_>::
	fill_rows_of_pool_parallel(
			nnz_lno_t current_a_row_begin, nnz_lno_t current_a_row_end,
			const_a_lno_row_view_t row_mapA_, const_a_lno_nnz_view_t entriesA_,
			const_b_lno_row_view_t row_mapB_, const_b_lno_nnz_view_t entriesB_,
			row_to_pool_view_t row_to_pool_index, pool_to_row_view_t pool_to_row_index){



		nnz_lno_t *prow_to_pool_index = row_to_pool_index.data();
		nnz_lno_t *ppool_to_row_index = pool_to_row_index.data();
		nnz_lno_t num_slots = pool_to_row_index.dimension_0();

		nnz_lno_t num_alloc = 0;
		for (nnz_lno_t i = current_a_row_begin; i < current_a_row_end; ++i){
			size_type a_row_begin = row_mapA_(i);
			nnz_lno_t row_size =  row_mapA_(i + 1) - a_row_begin;
			for (nnz_lno_t j = 0; j < row_size; ++j){
				nnz_lno_t a_col = entriesA_(j + a_row_begin);

				size_type b_row_begin = row_mapB_(a_col);
				nnz_lno_t brow_size =  row_mapB_(a_col + 1) - b_row_begin;

				for (nnz_lno_t k = 0; k < brow_size; ++k){
					nnz_lno_t b_col = entriesB_(k + b_row_begin);

					if (row_to_pool_index(b_col) == -1 &&
							Kokkos::atomic_compare_exchange_strong(prow_to_pool_index + b_col, -1, b_col))
					{
						++num_alloc;
						nnz_lno_t b_search_index = b_col;
						bool run_second_phase = false;
						while(!Kokkos::atomic_compare_exchange_strong(ppool_to_row_index + b_search_index, -1, b_col)){
							b_search_index++;
							if (b_search_index < num_slots) {
								run_second_phase = true;
								break;
							}
						}
						b_search_index = 0;
						while(run_second_phase && !Kokkos::atomic_compare_exchange_strong(ppool_to_row_index + b_search_index, -1, b_col)){
							b_search_index++;
						}
						row_to_pool_index(b_col) = b_search_index;
					}
				}
			}
		}
		return num_alloc;

	}
	*/

	template <typename HandleType, typename a_row_view_t_, typename a_lno_nnz_view_t_, typename a_scalar_nnz_view_t_,
				typename b_lno_row_view_t_, typename b_lno_nnz_view_t_, typename b_scalar_nnz_view_t_  >
	template <typename row_to_pool_view_t, typename pool_to_row_view_t>
	size_t KokkosSPGEMM <HandleType, a_row_view_t_, a_lno_nnz_view_t_, a_scalar_nnz_view_t_,
			b_lno_row_view_t_, b_lno_nnz_view_t_, b_scalar_nnz_view_t_>::
	fill_rows_of_pool_sequential(nnz_lno_t current_a_row_begin, nnz_lno_t current_a_row_end,
		const_a_lno_row_view_t row_mapA_, const_a_lno_nnz_view_t entriesA_, const_b_lno_row_view_t row_mapB_, const_b_lno_nnz_view_t entriesB_,
		row_to_pool_view_t row_to_pool_index, pool_to_row_view_t pool_to_row_index){

		//nnz_lno_t *prow_to_pool_index = row_to_pool_index.data();
		//nnz_lno_t *ppool_to_row_index = pool_to_row_index.data();
		nnz_lno_t num_slots = pool_to_row_index.dimension_0();

		nnz_lno_t num_alloc = 0;
		for (nnz_lno_t i = current_a_row_begin; i < current_a_row_end; ++i){
			size_type a_row_begin = row_mapA_(i);
			nnz_lno_t row_size =  row_mapA_(i + 1) - a_row_begin;
			for (nnz_lno_t j = 0; j < row_size; ++j){
				nnz_lno_t b_row = entriesA_(j + a_row_begin);
				//nnz_lno_t a_col = entriesA_(j + a_row_begin);

				//size_type b_row_begin = row_mapB_(a_col);
				//nnz_lno_t brow_size =  row_mapB_(a_col + 1) - b_row_begin;

				//for (nnz_lno_t k = 0; k < brow_size; ++k){
				//	nnz_lno_t b_col = entriesB_(k + b_row_begin);
					//std::cout << "b_col:" << b_row << std::endl;
					if (row_to_pool_index(b_row) == -1)
					{
						++num_alloc;

						nnz_lno_t b_col_mod = b_row % num_slots;

						nnz_lno_t b_search_index = b_col_mod;
						bool run_second_phase = false;



						while(pool_to_row_index[b_search_index] != -1){
							b_search_index++;
							if (b_search_index >= num_slots) {
								run_second_phase = true;
								break;
							}
						}
						if (run_second_phase){
							b_search_index = 0;
							while(pool_to_row_index[b_search_index] != -1){
								b_search_index++;
								if (b_search_index >= b_col_mod) {
									std::cout<< "CANNOT FIND A SPACE: " << i << " b_row:" << b_row << std::endl ;

									break;
								}
							}
						}
						pool_to_row_index[b_search_index] = b_row;
						row_to_pool_index[b_row] = b_search_index;
					}
				//}
			}
		}
		return num_alloc;
	}

}
}

