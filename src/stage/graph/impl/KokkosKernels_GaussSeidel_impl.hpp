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

#include "KokkosKernels_GraphColor.hpp"
#include "KokkosKernels_Utils.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_MemoryTraits.hpp>

#ifndef _KOKKOSGSIMP_HPP
#define _KOKKOSGSIMP_HPP

namespace KokkosKernels{

namespace Experimental{

namespace Graph{


namespace Impl{


template <typename HandleType, typename lno_row_view_t_, typename lno_nnz_view_t_, typename scalar_nnz_view_t_>
class GaussSeidel{

public:

  typedef lno_row_view_t_ in_lno_row_view_t;
  typedef lno_nnz_view_t_ in_lno_nnz_view_t;
  typedef scalar_nnz_view_t_ in_scalar_nnz_view_t;

  typedef typename HandleType::HandleExecSpace MyExecSpace;
  typedef typename HandleType::HandleTempMemorySpace MyTempMemorySpace;
  typedef typename HandleType::HandlePersistentMemorySpace MyPersistentMemorySpace;


  typedef typename in_lno_row_view_t::non_const_value_type row_lno_t;

  typedef typename HandleType::size_type size_type;
  typedef typename HandleType::nnz_lno_t nnz_lno_t;
  typedef typename HandleType::nnz_scalar_t nnz_scalar_t;


  typedef typename HandleType::const_lno_row_view_t const_lno_row_view_t;
  typedef typename HandleType::non_const_lno_row_view_t non_const_lno_row_view_t;

  typedef typename HandleType::const_lno_nnz_view_t const_lno_nnz_view_t;
  typedef typename HandleType::non_const_lno_nnz_view_t non_const_lno_nnz_view_t;

  typedef typename HandleType::const_scalar_nnz_view_t const_scalar_nnz_view_t;
  typedef typename HandleType::non_const_scalar_nnz_view_t non_const_scalar_nnz_view_t;




  typedef typename HandleType::row_lno_temp_work_view_t row_lno_temp_work_view_t;
  typedef typename HandleType::row_lno_persistent_work_view_t row_lno_persistent_work_view_t;
  typedef typename HandleType::row_lno_persistent_work_host_view_t row_lno_persistent_work_host_view_t; //Host view type



  typedef typename HandleType::nnz_lno_temp_work_view_t nnz_lno_temp_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_view_t nnz_lno_persistent_work_view_t;
  typedef typename HandleType::nnz_lno_persistent_work_host_view_t nnz_lno_persistent_work_host_view_t; //Host view type


  typedef typename HandleType::scalar_temp_work_view_t scalar_temp_work_view_t;
  typedef typename HandleType::scalar_persistent_work_view_t scalar_persistent_work_view_t;

  typedef Kokkos::RangePolicy<MyExecSpace> my_exec_space;
  typedef nnz_lno_t color_t;
  typedef Kokkos::View<color_t *, MyTempMemorySpace> color_view_t;

  typedef Kokkos::TeamPolicy<MyExecSpace> team_policy_t ;
  typedef typename team_policy_t::member_type team_member_t ;

  struct ForwardBlockedSGS{};
  struct ForwardAysncSGS{};
  struct BackwardBlockedSGS{};
  struct BackwardAysncSGS{};
private:
  HandleType *handle;
  nnz_lno_t num_rows, num_cols;

  const_lno_row_view_t row_map;
  const_lno_nnz_view_t entries;
  const_scalar_nnz_view_t values;
  bool is_symmetric;
public:



  struct PSGS{

    row_lno_persistent_work_view_t _xadj;
    nnz_lno_persistent_work_view_t _adj; // CSR storage of the graph.
    scalar_persistent_work_view_t _adj_vals; // CSR storage of the graph.

    scalar_persistent_work_view_t _Xvector /*output*/;
    scalar_persistent_work_view_t _Yvector;

    scalar_persistent_work_view_t _permuted_diagonals;

    PSGS(row_lno_persistent_work_view_t xadj_, nnz_lno_persistent_work_view_t adj_, scalar_persistent_work_view_t adj_vals_,
        scalar_persistent_work_view_t Xvector_, scalar_persistent_work_view_t Yvector_, nnz_lno_persistent_work_view_t color_adj_,
        scalar_persistent_work_view_t permuted_diagonals_):
          _xadj( xadj_),
          _adj( adj_),
          _adj_vals( adj_vals_),
          _Xvector( Xvector_),
          _Yvector( Yvector_), _permuted_diagonals(permuted_diagonals_){}

    KOKKOS_INLINE_FUNCTION
    void operator()(const nnz_lno_t &ii) const {

      size_type row_begin = _xadj[ii];
      size_type row_end = _xadj[ii + 1];

      nnz_scalar_t sum = _Yvector[ii];

      for (size_type adjind = row_begin; adjind < row_end; ++adjind){
        nnz_lno_t colIndex = _adj[adjind];
        nnz_scalar_t val = _adj_vals[adjind];
        sum -= val * _Xvector[colIndex];
      }


      nnz_scalar_t diagonalVal = _permuted_diagonals[ii];
      //_Xvector[ii] = (sum + diagonalVal * _Xvector[ii])/ diagonalVal;
      _Xvector[ii] += sum * diagonalVal ;//

      //if (ii < 200)
      //std::cout << "row_index:" << ii << " _Xvector[ii]:" << _Xvector[ii] << std::endl;

    }
  };

  struct Team_PSGS{

    row_lno_persistent_work_view_t _xadj;
    nnz_lno_persistent_work_view_t _adj; // CSR storage of the graph.
    scalar_persistent_work_view_t _adj_vals; // CSR storage of the graph.

    scalar_persistent_work_view_t _Xvector /*output*/;
    scalar_persistent_work_view_t _Yvector;
    nnz_lno_t _color_set_begin;
    nnz_lno_t _color_set_end;

    scalar_persistent_work_view_t _permuted_diagonals;


    Team_PSGS(row_lno_persistent_work_view_t xadj_, nnz_lno_persistent_work_view_t adj_, scalar_persistent_work_view_t adj_vals_,
        scalar_persistent_work_view_t Xvector_, scalar_persistent_work_view_t Yvector_,
        nnz_lno_t color_set_begin, nnz_lno_t color_set_end,
        scalar_persistent_work_view_t permuted_diagonals_):
          _xadj( xadj_),
          _adj( adj_),
          _adj_vals( adj_vals_),
          _Xvector( Xvector_),
          _Yvector( Yvector_),
          _color_set_begin(color_set_begin),
          _color_set_end(color_set_end), _permuted_diagonals(permuted_diagonals_){}

    KOKKOS_INLINE_FUNCTION
    void operator()(const team_member_t & teamMember) const {
      //idx ii = _color_adj[i];
      //int ii = teamMember.league_rank()  + _shift_index;

      nnz_lno_t ii = teamMember.league_rank()  * teamMember.team_size()+ teamMember.team_rank() + _color_set_begin;
      //check ii is out of range. if it is, just return.
      if (ii >= _color_set_end)
        return;



      size_type row_begin = _xadj[ii];
      size_type row_end = _xadj[ii + 1];

      //bool am_i_the_diagonal = false;
      //nnz_scalar_t diagonal = 1;
      nnz_scalar_t product = 0 ;
      Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(teamMember, row_end - row_begin),
          //Kokkos::TeamThreadRange(teamMember, row_end - row_begin),
          [&] (size_type i, nnz_scalar_t & valueToUpdate) {
        size_type adjind = i + row_begin;
        nnz_lno_t colIndex = _adj[adjind];
        nnz_scalar_t val = _adj_vals[adjind];
        valueToUpdate += val * _Xvector[colIndex];
      },
      product);

      Kokkos::single(Kokkos::PerThread(teamMember),[=] () {
        nnz_scalar_t diagonalVal = _permuted_diagonals[ii];
        //_Xvector[ii] = (_Yvector[ii] - product + diagonalVal * _Xvector[ii])/ diagonalVal;
        _Xvector[ii] += (_Yvector[ii] - product) * diagonalVal;//
      });
     }
  };

  template<typename size_type, typename lno_t, typename scalar_t>
  struct Async_SGS{

    const size_type *_xadj;
    const lno_t *_adj; // CSR storage of the graph.
    const scalar_t  *_adj_vals; // CSR storage of the graph.

    scalar_t  * _initial_Xvector /*output*/;
    scalar_t  * _temporal_Xvector /*output*/;

    const scalar_t  * _Yvector;

    const scalar_t *_diagonals;
    lno_t _num_rows;
    lno_t _block_size;


    Async_SGS(const size_type *xadj_, const lno_t *adj_,const  scalar_t *adj_vals_,
              scalar_t * initial_Xvector_, scalar_t * temporal_Xvector_,
              const scalar_t *Yvector_,
              const scalar_t *diagonals_,
              lno_t num_rows_,
              lno_t block_size_):
          _xadj( xadj_),
          _adj( adj_),
          _adj_vals( adj_vals_),
          _initial_Xvector( initial_Xvector_),
          _temporal_Xvector( temporal_Xvector_),
          _Yvector( Yvector_),
          _diagonals(diagonals_),
          _num_rows(num_rows_),
          _block_size(block_size_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const ForwardBlockedSGS&,  const team_member_t & teamMember) const {


      nnz_lno_t thread_index = teamMember.league_rank()  * teamMember.team_size()+ teamMember.team_rank();
      nnz_lno_t thread_begin = thread_index * _block_size;
      nnz_lno_t thread_end = KOKKOSKERNELS_MACRO_MIN(thread_begin + _block_size, _num_rows);


      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(teamMember, thread_begin , thread_end),
          [&] (lno_t row_index) {

        size_type row_begin = _xadj[row_index];
        size_type row_end = _xadj[row_index + 1];
        lno_t row_size = row_end - row_begin;
        scalar_t product = 0 ;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(teamMember, row_size),
            [&] (lno_t i, scalar_t & valueToUpdate) {
          size_type adjind = i + row_begin;
          lno_t colIndex = _adj[adjind];
          scalar_t val = _adj_vals[adjind];
          scalar_t vector_value = 0;
          if (colIndex < thread_end && colIndex >= thread_begin){
            vector_value = _temporal_Xvector[colIndex];
          }
          else {
            vector_value = _initial_Xvector[colIndex];
          }
          valueToUpdate += val * vector_value;
        },
        product);

        Kokkos::single(Kokkos::PerThread(teamMember),[=] () {
          nnz_scalar_t diagonalVal = _diagonals[row_index];
          //_temporal_Xvector[row_index] = (_Yvector[row_index] - product + diagonalVal * _temporal_Xvector[row_index])/ diagonalVal;
          _temporal_Xvector[row_index] += (_Yvector[row_index] - product) * diagonalVal;
        });

        //if (row_index < 200)
        //std::cout << "row_index:" << row_index << " _initial_Xvector[row_index]:" << _initial_Xvector[row_index] << std::endl;
      });
    }
    KOKKOS_INLINE_FUNCTION
    void operator()(const BackwardBlockedSGS&,  const team_member_t & teamMember) const {
      nnz_lno_t thread_index = teamMember.league_rank()  * teamMember.team_size()+ teamMember.team_rank();
      nnz_lno_t thread_begin = thread_index * _block_size;
      nnz_lno_t thread_end = KOKKOSKERNELS_MACRO_MIN(thread_begin + _block_size, _num_rows);

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(teamMember, thread_begin, thread_end),
          [&] (lno_t row_index) {
        row_index = thread_end - 1 - row_index + thread_begin;
        size_type row_begin = _xadj[row_index];
        size_type row_end = _xadj[row_index + 1];
        lno_t row_size = row_end - row_begin;
        scalar_t product = 0 ;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(teamMember, row_size),
            [&] (lno_t i, scalar_t & valueToUpdate) {
          size_type adjind = i + row_begin;
          lno_t colIndex = _adj[adjind];
          scalar_t val = _adj_vals[adjind];
          scalar_t vector_value = 0;
          if (colIndex < thread_end && colIndex >= thread_begin){
            vector_value = _temporal_Xvector[colIndex];
          }
          else {
            vector_value = _initial_Xvector[colIndex];
          }
          valueToUpdate += val * vector_value;
        },
        product);

        Kokkos::single(Kokkos::PerThread(teamMember),[=] () {
          nnz_scalar_t diagonalVal = _diagonals[row_index];
          /*
          _temporal_Xvector[row_index] = (_Yvector[row_index] - product +
              diagonalVal * _temporal_Xvector[row_index])/ diagonalVal;
          */
          _temporal_Xvector[row_index] += (_Yvector[row_index] - product) * diagonalVal;

        });
      });
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const BackwardAysncSGS&,  const team_member_t & teamMember) const {


      nnz_lno_t thread_index = teamMember.league_rank()  * teamMember.team_size()+ teamMember.team_rank();
      nnz_lno_t thread_begin = thread_index * _block_size;
      nnz_lno_t thread_end = KOKKOSKERNELS_MACRO_MIN(thread_begin + _block_size, _num_rows);

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(teamMember, thread_begin, thread_end),
          [&] (lno_t row_index) {
        row_index = thread_end - 1 - row_index+ thread_begin;
        size_type row_begin = _xadj[row_index];
        size_type row_end = _xadj[row_index + 1];
        lno_t row_size = row_end - row_begin;
        scalar_t product = 0 ;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(teamMember, row_size),
            [&] (lno_t i, scalar_t & valueToUpdate) {
          size_type adjind = i + row_begin;
          lno_t colIndex = _adj[adjind];
          scalar_t val = _adj_vals[adjind];
          scalar_t vector_value = _initial_Xvector[colIndex];
          valueToUpdate += val * vector_value;
        },
        product);

        Kokkos::single(Kokkos::PerThread(teamMember),[=] () {
          nnz_scalar_t diagonalVal = _diagonals[row_index];
          /*
          _initial_Xvector[row_index] = (_Yvector[row_index] - product +
              diagonalVal * _initial_Xvector[row_index])/ diagonalVal;
          */
          _initial_Xvector[row_index] += (_Yvector[row_index] - product) * diagonalVal;

        });
      });
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const ForwardAysncSGS&,  const team_member_t & teamMember) const {


      nnz_lno_t thread_index = teamMember.league_rank()  * teamMember.team_size()+ teamMember.team_rank();
      nnz_lno_t thread_begin = thread_index * _block_size;
      nnz_lno_t thread_end = KOKKOSKERNELS_MACRO_MIN(thread_begin + _block_size, _num_rows);

      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(teamMember, thread_begin, thread_end),
          [&] (lno_t row_index) {


        size_type row_begin = _xadj[row_index];
        size_type row_end = _xadj[row_index + 1];
        lno_t row_size = row_end - row_begin;
        scalar_t product = 0 ;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(teamMember, row_size),
            [&] (lno_t i, scalar_t & valueToUpdate) {
          size_type adjind = i + row_begin;
          lno_t colIndex = _adj[adjind];
          scalar_t val = _adj_vals[adjind];

          scalar_t vector_value = _initial_Xvector[colIndex];
          valueToUpdate += val * vector_value;
        },
        product);

        Kokkos::single(Kokkos::PerThread(teamMember),[=] () {
          nnz_scalar_t diagonalVal = _diagonals[row_index];
          /*
          _initial_Xvector[row_index] =
              (_Yvector[row_index] - product +
               diagonalVal * _initial_Xvector[row_index])/ diagonalVal;
          */
          _initial_Xvector[row_index] += (_Yvector[row_index] - product) * diagonalVal;

        });
        //if (row_index < 200)
        //std::cout << "row_index:" << row_index << " _initial_Xvector[row_index]:" << _initial_Xvector[row_index] << std::endl;
      });

      /*
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(teamMember, thread_begin, thread_end),
          [&] (lno_t row_index) {
        row_index = thread_end - 1 - row_index;
        size_type row_begin = _xadj[row_index];
        size_type row_end = _xadj[row_index + 1];
        lno_t row_size = row_end - row_begin;
        scalar_t product = 0 ;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(teamMember, row_size),
            [&] (lno_t i, scalar_t & valueToUpdate) {
          size_type adjind = i + row_begin;
          lno_t colIndex = _adj[adjind];
          scalar_t val = _adj_vals[adjind];
          scalar_t vector_value = _initial_Xvector[colIndex];
          valueToUpdate += val * vector_value;
        },
        product);

        Kokkos::single(Kokkos::PerThread(teamMember),[=] () {
          nnz_scalar_t diagonalVal = _diagonals[row_index];
          _initial_Xvector[row_index] = (_Yvector[row_index] - product +
              diagonalVal * _initial_Xvector[row_index])/ diagonalVal;
        });
      });
      */

    }
  };



  /**
   * \brief constructor
   */

  GaussSeidel(
      HandleType *handle_,
      nnz_lno_t num_rows_,
      nnz_lno_t num_cols_,
      const_lno_row_view_t row_map_,
      const_lno_nnz_view_t entries_,
      const_scalar_nnz_view_t values_):
        handle(handle_), num_rows(num_rows_), num_cols(num_cols_),
        row_map(row_map_), entries(entries_), values(values_), is_symmetric(true){}


  GaussSeidel(HandleType *handle_,
      nnz_lno_t num_rows_,
      nnz_lno_t num_cols_,
      const_lno_row_view_t row_map_,
      const_lno_nnz_view_t entries_,
      bool is_symmetric_ = true):
        handle(handle_),
        num_rows(num_rows_), num_cols(num_cols_),

        row_map(row_map_),
        entries(entries_),
        values(), is_symmetric(is_symmetric_){}



  /**
   * \brief constructor
   */
  GaussSeidel(HandleType *handle_,
      nnz_lno_t num_rows_,
      nnz_lno_t num_cols_,
      const_lno_row_view_t row_map_,
      const_lno_nnz_view_t entries_,
      const_scalar_nnz_view_t values_,
      bool is_symmetric_):
        handle(handle_),
        num_rows(num_rows_), num_cols(num_cols_),
        row_map(row_map_), entries(entries_), values(values_), is_symmetric(is_symmetric_){}



  void initialize_gc_symbolic(){

    //std::cout << std::endl<< std::endl<< std::endl<< std::endl<< std::endl<< std::endl;
    typename HandleType::GraphColoringHandleType *gchandle = this->handle->get_graph_coloring_handle();


    if (gchandle == NULL){

      this->handle->create_graph_coloring_handle();
      //this->handle->create_gs_handle();
      this->handle->get_gs_handle()->set_owner_of_coloring();
      gchandle = this->handle->get_graph_coloring_handle();
    }



    const_lno_row_view_t xadj = this->row_map;
    const_lno_nnz_view_t adj = this->entries;
    size_type nnz = adj.dimension_0();

#ifdef KOKKOSKERNELS_TIME_REVERSE
    Kokkos::Impl::Timer timer;
#endif
    {
      if (!is_symmetric){

        if (gchandle->get_coloring_algo_type() == KokkosKernels::Experimental::Graph::COLORING_EB){

          gchandle->symmetrize_and_calculate_lower_diagonal_edge_list(num_rows, xadj, adj);
          graph_color_symbolic <HandleType, const_lno_row_view_t, const_lno_nnz_view_t>
              (this->handle, num_rows, num_rows, xadj , adj);
        }
        else {
          row_lno_temp_work_view_t tmp_xadj;
          nnz_lno_temp_work_view_t tmp_adj;
          KokkosKernels::Experimental::Util::symmetrize_graph_symbolic_hashmap
          < const_lno_row_view_t, const_lno_nnz_view_t,
          row_lno_temp_work_view_t, nnz_lno_temp_work_view_t,
          MyExecSpace>
          (num_rows, xadj, adj, tmp_xadj, tmp_adj );
          graph_color_symbolic <HandleType, row_lno_temp_work_view_t, nnz_lno_temp_work_view_t> (this->handle, num_rows, num_rows, tmp_xadj , tmp_adj);
        }
      }
      else {
        graph_color_symbolic <HandleType, const_lno_row_view_t, const_lno_nnz_view_t> (this->handle, num_rows, num_rows, xadj , adj);
      }


    }
    color_t numColors = gchandle->get_num_colors();

   //std::cout << "numCol:" << numColors << " numRows:" << num_rows << " cols:" << num_cols << " nnz:" << adj.dimension_0() <<  std::endl;

#ifdef KOKKOSKERNELS_TIME_REVERSE
    std::cout << "COLORING_TIME:" << timer.seconds() << std::endl;
#endif


    typename HandleType::GraphColoringHandleType::color_view_t colors =  gchandle->get_vertex_colors();

    /*
    {
      std::cout << "DELETE HERE" << std::endl;
      numColors = 1;
      for (int i = 0; i < this->num_rows; ++i) colors(i) = 1;
    }
    */


    nnz_lno_persistent_work_view_t color_xadj;

    nnz_lno_persistent_work_view_t color_adj;


#ifdef KOKKOSKERNELS_TIME_REVERSE
    timer.reset();
#endif

    KokkosKernels::Experimental::Util::create_reverse_map
      <typename HandleType::GraphColoringHandleType::color_view_t,
        nnz_lno_persistent_work_view_t, MyExecSpace>
        (num_rows, numColors, colors, color_xadj, color_adj);
    MyExecSpace::fence();

    KokkosKernels::Experimental::Util::kk_print_1Dview(colors);

    KokkosKernels::Experimental::Util::kk_print_1Dview(color_xadj);
    KokkosKernels::Experimental::Util::kk_print_1Dview(color_adj);

#ifdef KOKKOSKERNELS_TIME_REVERSE
    std::cout << "CREATE_REVERSE_MAP:" << timer.seconds() << std::endl;
    timer.reset();
#endif

    nnz_lno_persistent_work_host_view_t  h_color_xadj = Kokkos::create_mirror_view (color_xadj);
    Kokkos::deep_copy (h_color_xadj , color_xadj);
    MyExecSpace::fence();

#ifdef KOKKOSKERNELS_TIME_REVERSE
    std::cout << "DEEP_COPY:" << timer.seconds() << std::endl;
    timer.reset();
#endif


#if defined( KOKKOS_HAVE_CUDA )
    if (Kokkos::Impl::is_same<Kokkos::Cuda, MyExecSpace >::value){
      for (nnz_lno_t i = 0; i < numColors; ++i){
        nnz_lno_t color_index_begin = h_color_xadj(i);
        nnz_lno_t color_index_end = h_color_xadj(i + 1);
        if (color_index_begin + 1 >= color_index_end ) continue;
        auto colorsubset =
            subview(color_adj, Kokkos::pair<row_lno_t, row_lno_t> (color_index_begin, color_index_end));
        Kokkos::sort (colorsubset);
      }
    }
#endif

/*
    {
      std::cout << "DELETE HERE" << std::endl;
      Kokkos::sort (color_adj);
    }
    */

    MyExecSpace::fence();
#ifdef KOKKOSKERNELS_TIME_REVERSE
    std::cout << "SORT_TIME:" << timer.seconds() << std::endl;
    timer.reset();
    //std::cout << "sort" << std::endl;
#endif

    row_lno_persistent_work_view_t permuted_xadj ("new xadj", num_rows + 1);
    nnz_lno_persistent_work_view_t old_to_new_map ("old_to_new_index_", num_rows );
    nnz_lno_persistent_work_view_t permuted_adj ("newadj_", nnz );
    Kokkos::parallel_for( my_exec_space(0,num_rows),
        create_permuted_xadj(
            color_adj,
            xadj,
            permuted_xadj,
            old_to_new_map));
    //std::cout << "create_permuted_xadj" << std::endl;
    MyExecSpace::fence();

#ifdef KOKKOSKERNELS_TIME_REVERSE
    std::cout << "CREATE_PERMUTED_XADJ:" << timer.seconds() << std::endl;

    timer.reset();
#endif


    KokkosKernels::Experimental::Util::inclusive_parallel_prefix_sum
        <row_lno_persistent_work_view_t, MyExecSpace>
        (num_rows + 1, permuted_xadj);
    MyExecSpace::fence();

#ifdef KOKKOSKERNELS_TIME_REVERSE
    std::cout << "INCLUSIVE_PPS:" << timer.seconds() << std::endl;
    timer.reset();
#endif


    Kokkos::parallel_for( my_exec_space(0,num_rows),
        fill_matrix_symbolic(
            num_rows,
            color_adj,
            xadj,
            adj,
            //adj_vals,
            permuted_xadj,
            permuted_adj,
            //newvals_,
            old_to_new_map));
    MyExecSpace::fence();

#ifdef KOKKOSKERNELS_TIME_REVERSE
    std::cout << "SYMBOLIC_FILL:" << timer.seconds() << std::endl;
    timer.reset();
#endif



    typename HandleType::GaussSeidelHandleType *gsHandler = this->handle->get_gs_handle();
    gsHandler->set_color_set_xadj(h_color_xadj);
    gsHandler->set_color_set_adj(color_adj);
    gsHandler->set_num_colors(numColors);
    gsHandler->set_new_xadj(permuted_xadj);
    gsHandler->set_new_adj(permuted_adj);
    //gsHandler->set_new_adj_val(newvals_);
    gsHandler->set_old_to_new_map(old_to_new_map);
    if (this->handle->get_gs_handle()->is_owner_of_coloring()){
      this->handle->destroy_graph_coloring_handle();
      this->handle->get_gs_handle()->set_owner_of_coloring(false);
    }
    this->handle->get_gs_handle()->set_call_symbolic(true);
    this->handle->get_gs_handle()->allocate_x_y_vectors(this->num_rows, this->num_cols);
    //std::cout << "all end" << std::endl;
#ifdef KOKKOSKERNELS_TIME_REVERSE
    std::cout << "ALLOC:" << timer.seconds() << std::endl;
#endif


  }


  void initialize_symbolic(){
    typename HandleType::GaussSeidelHandleType *gsHandler = this->handle->get_gs_handle();
    switch (gsHandler->get_algorithm_type()){
      case GS_PERMUTED:
      case GS_TEAM:
      case GS_DEFAULT:
        this->initialize_gc_symbolic();
        break;

      case GS_BLOCKED:
        this->handle->get_gs_handle()->allocate_x_vector(this->num_cols);
        this->handle->get_gs_handle()->set_call_symbolic(true);
        break;
      case GS_AYSNC:
        this->handle->get_gs_handle()->set_call_symbolic(true);
        break;
    }
  }

  struct create_permuted_xadj{
    nnz_lno_persistent_work_view_t color_adj;
    const_lno_row_view_t oldxadj;
    row_lno_persistent_work_view_t newxadj;
    nnz_lno_persistent_work_view_t old_to_new_index;
    create_permuted_xadj(
        nnz_lno_persistent_work_view_t color_adj_,
        const_lno_row_view_t oldxadj_,
        row_lno_persistent_work_view_t newxadj_,
        nnz_lno_persistent_work_view_t old_to_new_index_):
          color_adj(color_adj_), oldxadj(oldxadj_),
          newxadj(newxadj_),old_to_new_index(old_to_new_index_){}

    KOKKOS_INLINE_FUNCTION
    void operator()(const nnz_lno_t &i) const{
      nnz_lno_t index = color_adj(i);
      newxadj(i + 1) = oldxadj[index + 1] - oldxadj[index];
      old_to_new_index[index] = i;
    }
  };

  struct fill_matrix_symbolic{
    nnz_lno_t num_rows;
    nnz_lno_persistent_work_view_t color_adj;
    const_lno_row_view_t oldxadj;
    const_lno_nnz_view_t oldadj;
    //value_array_type oldadjvals;
    row_lno_persistent_work_view_t newxadj;
    nnz_lno_persistent_work_view_t newadj;
    //value_persistent_work_array_type newadjvals;
    nnz_lno_persistent_work_view_t old_to_new_index;
    fill_matrix_symbolic(
        nnz_lno_t num_rows_,
        nnz_lno_persistent_work_view_t color_adj_,
        const_lno_row_view_t oldxadj_,
        const_lno_nnz_view_t oldadj_,
        //value_array_type oldadjvals_,
        row_lno_persistent_work_view_t newxadj_,
        nnz_lno_persistent_work_view_t newadj_,
        //value_persistent_work_array_type newadjvals_,
        nnz_lno_persistent_work_view_t old_to_new_index_):
          num_rows(num_rows_),
          color_adj(color_adj_), oldxadj(oldxadj_), oldadj(oldadj_), //oldadjvals(oldadjvals_),
          newxadj(newxadj_), newadj(newadj_), //newadjvals(newadjvals_),
          old_to_new_index(old_to_new_index_){}

    KOKKOS_INLINE_FUNCTION
    void operator()(const nnz_lno_t &i) const{
      nnz_lno_t index = color_adj(i);
      size_type xadj_begin = newxadj(i);

      size_type old_xadj_end = oldxadj[index + 1];
      for (size_type j = oldxadj[index]; j < old_xadj_end; ++j){
        nnz_lno_t neighbor = oldadj[j];
        if(neighbor < num_rows) neighbor = old_to_new_index[neighbor];
        newadj[xadj_begin++] = neighbor;
        //newadjvals[xadj_begin++] = oldadjvals[j];
      }
    }
  };


  struct fill_matrix_numeric{
    nnz_lno_persistent_work_view_t color_adj;
    const_lno_row_view_t oldxadj;
    const_scalar_nnz_view_t oldadjvals;
    row_lno_persistent_work_view_t newxadj;
    scalar_persistent_work_view_t newadjvals;
    fill_matrix_numeric(
        nnz_lno_persistent_work_view_t color_adj_,
        const_lno_row_view_t oldxadj_,
        const_scalar_nnz_view_t oldadjvals_,
        row_lno_persistent_work_view_t newxadj_,
        scalar_persistent_work_view_t newadjvals_):
          color_adj(color_adj_), oldxadj(oldxadj_),  oldadjvals(oldadjvals_),
          newxadj(newxadj_), newadjvals(newadjvals_){}

    KOKKOS_INLINE_FUNCTION
    void operator()(const nnz_lno_t &i) const{
      nnz_lno_t index = color_adj(i);
      size_type xadj_begin = newxadj(i);

      size_type old_xadj_end = oldxadj[index + 1];
      for (size_type j = oldxadj[index]; j < old_xadj_end; ++j){
        newadjvals[xadj_begin++] = oldadjvals[j];
      }
    }
  };


  template <typename size_type_view_t, typename lno_t_view_t, typename scalar_t_view_t, typename scalar_t_view2_t>
  struct Get_Matrix_Diagonals{

    size_type_view_t _xadj;
    lno_t_view_t _adj; // CSR storage of the graph.
    scalar_t_view_t _adj_vals; // CSR storage of the graph.
    scalar_t_view2_t _diagonals;
    size_type nr;

    Get_Matrix_Diagonals(
        size_type_view_t xadj_,
        lno_t_view_t adj_,
        scalar_t_view_t adj_vals_,
        scalar_t_view2_t diagonals_):
          _xadj( xadj_),
          _adj( adj_),
          _adj_vals( adj_vals_), _diagonals(diagonals_),
          nr(xadj_.dimension_0() - 1){}

    KOKKOS_INLINE_FUNCTION
    void operator()(const nnz_lno_t & ii) const {
      size_type row_begin = _xadj[ii];
      size_type row_end = _xadj[ii + 1];
      for (size_type c = row_begin; c < row_end; ++c){
        nnz_lno_t colIndex = _adj[c];
        if (colIndex == ii){
          nnz_scalar_t val = _adj_vals[c];
          _diagonals[ii] = 1.0 / val;
        }
      }
    }
  };

  void initialize_gc_numeric(){

    const_lno_row_view_t xadj = this->row_map;
    const_lno_nnz_view_t adj = this->entries;

    size_type nnz = adj.dimension_0();
    const_scalar_nnz_view_t adj_vals = this->values;

    typename HandleType::GaussSeidelHandleType *gsHandler = this->handle->get_gs_handle();



    row_lno_persistent_work_view_t newxadj_ = gsHandler->get_new_xadj();
    nnz_lno_persistent_work_view_t old_to_new_map = gsHandler->get_old_to_new_map();
    nnz_lno_persistent_work_view_t newadj_ = gsHandler->get_new_adj();

    nnz_lno_persistent_work_view_t color_adj = gsHandler->get_color_adj();
    scalar_persistent_work_view_t permuted_adj_vals (Kokkos::ViewAllocateWithoutInitializing("newvals_"), nnz );

    Kokkos::parallel_for( my_exec_space(0,num_rows),
        fill_matrix_numeric(
            color_adj,
            xadj,
            //adj,
            adj_vals,
            newxadj_,
            //newadj_,
            permuted_adj_vals
            //,old_to_new_map
            ));
    MyExecSpace::fence();
    gsHandler->set_new_adj_val(permuted_adj_vals);



    scalar_persistent_work_view_t permuted_diagonals (Kokkos::ViewAllocateWithoutInitializing("permuted_diagonals"), num_rows );

    Get_Matrix_Diagonals<row_lno_persistent_work_view_t, nnz_lno_persistent_work_view_t, scalar_persistent_work_view_t, scalar_persistent_work_view_t> gmd(newxadj_, newadj_, permuted_adj_vals, permuted_diagonals);
    /*
    int teamSizeMax = 0;
    int vector_size = 0;
    int max_allowed_team_size = team_policy_t::team_size_max(gmd);

    this->handle->get_gs_handle()->vector_team_size(max_allowed_team_size, vector_size, teamSizeMax, num_rows, nnz);
    Kokkos::parallel_for(
        team_policy_t(num_rows / teamSizeMax + 1 , teamSizeMax, vector_size),
        gmd );
        */
    Kokkos::parallel_for(
              my_exec_space(0,num_rows),
              gmd );
    MyExecSpace::fence();
    this->handle->get_gs_handle()->set_permuted_diagonals(permuted_diagonals);


    this->handle->get_gs_handle()->set_call_numeric(true);


  }


  void initialize_numeric(){

    typename HandleType::GaussSeidelHandleType *gsHandler = this->handle->get_gs_handle();

    if (gsHandler->is_symbolic_called() == false){
      this->initialize_symbolic();
    }
    //else
#ifdef KOKKOSKERNELS_TIME_REVERSE
    Kokkos::Impl::Timer timer;
#endif
    {
      switch (gsHandler->get_algorithm_type()){
        case GS_PERMUTED:
        case GS_TEAM:
        case GS_DEFAULT:
          this->initialize_gc_numeric();
          break;

        case GS_BLOCKED:
        case GS_AYSNC:
          scalar_persistent_work_view_t permuted_diagonals (Kokkos::ViewAllocateWithoutInitializing("permuted_diagonals"), num_rows );


          Get_Matrix_Diagonals<const_lno_row_view_t, const_lno_nnz_view_t, const_scalar_nnz_view_t, scalar_persistent_work_view_t>  gmd
            (this->row_map, this->entries, this->values, permuted_diagonals);
          Kokkos::parallel_for( my_exec_space(0,num_rows), gmd );
          MyExecSpace::fence();
          gsHandler->set_permuted_diagonals(permuted_diagonals);
          gsHandler->set_call_numeric(true);
          break;
      }
    }

#ifdef KOKKOSKERNELS_TIME_REVERSE
    std::cout << "NUMERIC:" << timer.seconds() << std::endl;
#endif
  }


  template <typename x_value_array_type, typename y_value_array_type>
  void gc_apply(
      x_value_array_type x_lhs_output_vec,
      y_value_array_type y_rhs_input_vec,
      bool init_zero_x_vector = false,
      int numIter = 1,
      bool apply_forward = true,
      bool apply_backward = true,
      bool update_y_vector = true){

    typename HandleType::GaussSeidelHandleType *gsHandler = this->handle->get_gs_handle();
    scalar_persistent_work_view_t Permuted_Yvector = gsHandler->get_permuted_y_vector();
    scalar_persistent_work_view_t Permuted_Xvector = gsHandler->get_permuted_x_vector();


    row_lno_persistent_work_view_t newxadj_ = gsHandler->get_new_xadj();
    nnz_lno_persistent_work_view_t old_to_new_map = gsHandler->get_old_to_new_map();
    nnz_lno_persistent_work_view_t newadj_ = gsHandler->get_new_adj();
    nnz_lno_persistent_work_view_t color_adj = gsHandler->get_color_adj();

    color_t numColors = gsHandler->get_num_colors();



    if (update_y_vector){
      KokkosKernels::Experimental::Util::permute_vector
        <y_value_array_type,
        scalar_persistent_work_view_t,
        nnz_lno_persistent_work_view_t, MyExecSpace>(
          num_rows,
          old_to_new_map,
          y_rhs_input_vec,
          Permuted_Yvector
      );
    }
    MyExecSpace::fence();
    if(init_zero_x_vector){
      KokkosKernels::Experimental::Util::zero_vector<scalar_persistent_work_view_t, MyExecSpace>(num_cols, Permuted_Xvector);
    }
    else{
      KokkosKernels::Experimental::Util::permute_vector
        <x_value_array_type, scalar_persistent_work_view_t, nnz_lno_persistent_work_view_t, MyExecSpace>(
          num_cols,
          old_to_new_map,
          x_lhs_output_vec,
          Permuted_Xvector
          );
    }
    MyExecSpace::fence();

    row_lno_persistent_work_view_t permuted_xadj = gsHandler->get_new_xadj();
    nnz_lno_persistent_work_view_t permuted_adj = gsHandler->get_new_adj();
    scalar_persistent_work_view_t permuted_adj_vals = gsHandler->get_new_adj_val();
    scalar_persistent_work_view_t permuted_diagonals = gsHandler->get_permuted_diagonals();

    nnz_lno_persistent_work_host_view_t h_color_xadj = gsHandler->get_color_xadj();



    if (gsHandler->get_algorithm_type()== GS_PERMUTED){
      PSGS gs(permuted_xadj, permuted_adj, permuted_adj_vals,
          Permuted_Xvector, Permuted_Yvector, color_adj, permuted_diagonals);

      this->IterativePSGS(
          gs,
          numColors,
          h_color_xadj,
          numIter,
          apply_forward,
          apply_backward);
    }
    else{

      Team_PSGS gs(permuted_xadj, permuted_adj, permuted_adj_vals,
          Permuted_Xvector, Permuted_Yvector,0,0, permuted_diagonals);

      this->IterativePSGS(
          gs,
          numColors,
          h_color_xadj,
          numIter,
          apply_forward,
          apply_backward);
    }

    //Kokkos::parallel_for( my_exec_space(0,nr), PermuteVector(x_lhs_output_vec, Permuted_Xvector, color_adj));


    KokkosKernels::Experimental::Util::permute_vector
    <scalar_persistent_work_view_t,x_value_array_type,  nnz_lno_persistent_work_view_t, MyExecSpace>(
        num_cols,
        color_adj,
        Permuted_Xvector,
        x_lhs_output_vec
        );
    MyExecSpace::fence();


  }

  template <typename x_value_array_type, typename y_value_array_type>
  void async_apply(
      x_value_array_type x_lhs_output_vec,
      y_value_array_type y_rhs_input_vec,
      bool init_zero_x_vector = false,
      int num_iteration = 1,
      bool apply_forward = true,
      bool apply_backward = true,
      bool update_y_vector = true,
      bool is_block_base = true){

    if(init_zero_x_vector){
      KokkosKernels::Experimental::Util::zero_vector<x_value_array_type, MyExecSpace>(
          num_cols, x_lhs_output_vec);
    }
    typename HandleType::GaussSeidelHandleType *gsHandler = this->handle->get_gs_handle();

    scalar_persistent_work_view_t _diagonals = gsHandler->get_permuted_diagonals();

    //KokkosKernels::Experimental::Util::kk_print_1Dview(x_lhs_output_vec);
    //KokkosKernels::Experimental::Util::kk_print_1Dview(y_rhs_input_vec);
    //KokkosKernels::Experimental::Util::kk_print_1Dview(_diagonals);

    int concurrency = MyExecSpace::concurrency();


    size_type nnz = this->entries.dimension_0();
    int suggested_vector_size = this->handle->get_suggested_vector_size(this->num_rows, nnz);
    int suggested_team_size = this->handle->get_suggested_team_size(suggested_vector_size);

    nnz_lno_t block_size = this->handle->get_team_work_size(suggested_team_size,concurrency,this->num_rows);

    nnz_lno_t num_teams = this->num_rows / block_size + 1;
    nnz_scalar_t *temporal_Xvector_ = x_lhs_output_vec.data();
    scalar_persistent_work_view_t temporal_x_vector;

    if (is_block_base){

      temporal_x_vector = gsHandler->get_permuted_x_vector();
      Kokkos::deep_copy(temporal_x_vector, x_lhs_output_vec);
      temporal_Xvector_ = temporal_x_vector.data();
      block_size = this->num_rows / concurrency;
      if (num_rows % concurrency) ++block_size;
      num_teams = concurrency;
    }


    //std::cout << "num_teams:" << num_teams << " block_size:" << block_size << " num_rows:" << num_rows << std::endl;
    Async_SGS<size_type, nnz_lno_t, nnz_scalar_t> async_sgs (
          this->row_map.data(), this->entries.data(), this->values.data(),
          x_lhs_output_vec.data(), temporal_Xvector_,
          y_rhs_input_vec.data(), _diagonals.data(),
          this->num_rows,
          block_size);

    typedef Kokkos::TeamPolicy<ForwardBlockedSGS, MyExecSpace, > forward_blocked_gs_tp ;
    typedef Kokkos::TeamPolicy<ForwardAysncSGS, MyExecSpace, > forward_async_gs_tp ;
    typedef Kokkos::TeamPolicy<BackwardBlockedSGS, MyExecSpace, > backward_blocked_gs_tp ;
    typedef Kokkos::TeamPolicy<BackwardAysncSGS, MyExecSpace, > backward_aysnc_gs_tp ;

    for (int i = 0; i < num_iteration; ++i){

      if (apply_forward){
        if (is_block_base){

        Kokkos::parallel_for(
            forward_blocked_gs_tp(num_teams, suggested_team_size, suggested_vector_size), async_sgs );
        //MyExecSpace::fence();
        //Kokkos::deep_copy(x_lhs_output_vec, temporal_x_vector);
        }
        else {

          Kokkos::parallel_for(
              forward_async_gs_tp(num_teams, suggested_team_size, suggested_vector_size), async_sgs );
        }
        MyExecSpace::fence();
      }
      if (apply_backward){
        if (is_block_base){

        Kokkos::parallel_for(
            backward_blocked_gs_tp(num_teams, suggested_team_size, suggested_vector_size), async_sgs );
        //MyExecSpace::fence();
        //Kokkos::deep_copy(x_lhs_output_vec, temporal_x_vector);

        }
        else {

          Kokkos::parallel_for(
              backward_aysnc_gs_tp(num_teams, suggested_team_size, suggested_vector_size), async_sgs );
        }
        MyExecSpace::fence();
      }

      if (is_block_base){
        Kokkos::deep_copy(x_lhs_output_vec, temporal_x_vector);
      }
    }
    //KokkosKernels::Experimental::Util::kk_print_1Dview(x_lhs_output_vec);

  }

  template <typename x_value_array_type, typename y_value_array_type>
  void apply(
      x_value_array_type x_lhs_output_vec,
      y_value_array_type y_rhs_input_vec,
      bool init_zero_x_vector = false,
      int numIter = 1,
      bool apply_forward = true,
      bool apply_backward = true,
      bool update_y_vector = true){
    if (this->handle->get_gs_handle()->is_numeric_called() == false){
      this->initialize_numeric();
    }
    typename HandleType::GaussSeidelHandleType *gsHandler = this->handle->get_gs_handle();

    switch (gsHandler->get_algorithm_type()){
      case GS_PERMUTED:
      case GS_TEAM:
      case GS_DEFAULT:
        this->gc_apply(
            x_lhs_output_vec,
            y_rhs_input_vec,
            init_zero_x_vector,
            numIter,
            apply_forward,
            apply_backward,
            update_y_vector);
            break;
      case GS_BLOCKED:
      case GS_AYSNC:
        async_apply(
            x_lhs_output_vec,
            y_rhs_input_vec,
            init_zero_x_vector,
            numIter,
            apply_forward,
            apply_backward,
            update_y_vector, gsHandler->get_algorithm_type() ==  GS_BLOCKED);
        break;
    }

  }

  void IterativePSGS(
      Team_PSGS &gs,
      color_t numColors,
      nnz_lno_persistent_work_host_view_t h_color_xadj,
      int num_iteration,
      bool apply_forward,
      bool apply_backward){

    for (int i = 0; i < num_iteration; ++i){
      this->DoPSGS(gs, numColors, h_color_xadj, apply_forward, apply_backward);
    }
  }

  void DoPSGS(Team_PSGS &gs, color_t numColors, nnz_lno_persistent_work_host_view_t h_color_xadj,
      bool apply_forward,
      bool apply_backward){
    int teamSizeMax = 0;
    int vector_size = 0;
    int max_allowed_team_size = team_policy_t::team_size_max(gs);


    size_type nnz = this->entries.dimension_0();


    this->handle->get_gs_handle()->vector_team_size(max_allowed_team_size, vector_size, teamSizeMax, num_rows, nnz);
    /*std::cout
        << "max_allowed_team_size"  << max_allowed_team_size
        << " vector_size:" << vector_size
        << " teamSizeMax:" << teamSizeMax << std::endl;
    */
    if (apply_forward){
      for (color_t i = 0; i < numColors; ++i){
        nnz_lno_t color_index_begin = h_color_xadj(i);
        nnz_lno_t color_index_end = h_color_xadj(i + 1);

        int overall_work = color_index_end - color_index_begin;// /256 + 1;


        gs._color_set_begin = color_index_begin;
        gs._color_set_end = color_index_end;

        Kokkos::parallel_for(
            team_policy_t(overall_work / teamSizeMax + 1 , teamSizeMax, vector_size),
            gs );
        MyExecSpace::fence();
      }
    }
    if (apply_backward){
      if (numColors > 0)
      for (color_t i = numColors - 1;  ; --i){
        nnz_lno_t color_index_begin = h_color_xadj(i);
        nnz_lno_t color_index_end = h_color_xadj(i + 1);

        nnz_lno_t numberOfTeams = color_index_end - color_index_begin;// /256 + 1;
        gs._color_set_begin = color_index_begin;
        gs._color_set_end = color_index_end;

        Kokkos::parallel_for(
            team_policy_t(numberOfTeams / teamSizeMax + 1 , teamSizeMax, vector_size),
            gs );
        MyExecSpace::fence();
        if (i == 0){
          break;
        }
      }
    }
  }

  void IterativePSGS(
      PSGS &gs,
      color_t numColors,
      nnz_lno_persistent_work_host_view_t h_color_xadj,
      int num_iteration,
      bool apply_forward,
      bool apply_backward){

    for (int i = 0; i < num_iteration; ++i){
      this->DoPSGS(gs, numColors, h_color_xadj, apply_forward, apply_backward);
    }
  }



  void DoPSGS(PSGS &gs, color_t numColors, nnz_lno_persistent_work_host_view_t h_color_xadj,
      bool apply_forward,
      bool apply_backward){
    if (apply_forward){
      for (color_t i = 0; i < numColors; ++i){
        nnz_lno_t color_index_begin = h_color_xadj(i);
        nnz_lno_t color_index_end = h_color_xadj(i + 1);
        Kokkos::parallel_for (my_exec_space (color_index_begin, color_index_end) , gs);
        MyExecSpace::fence();
      }
    }
    if (apply_backward && numColors){
      for (size_type i = numColors - 1; ; --i){
        nnz_lno_t color_index_begin = h_color_xadj(i);
        nnz_lno_t color_index_end = h_color_xadj(i + 1);
        Kokkos::parallel_for (my_exec_space (color_index_begin, color_index_end) , gs);
        MyExecSpace::fence();
        if (i == 0){
          break;
        }
      }
    }
  }
};

}
}
}
}
#endif
