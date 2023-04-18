#include <ATen/ATen.h>


void cuda_init();
void cuda_new_cells(void **ptr_active_cells, void **ptr_all_cells, const at::Tensor & seq_lens, uint window_size);
void cuda_init_cell_ranges(void **update_range, void **cell_offsets, const at::Tensor & seq_lens);
void cuda_init_misc(uint ***ptr_to_leftrecord,
                    uint ***ptr_to_rightrecord,
                    TreeNode **nodes_cache,
                    float **log_p_cache,
                    uint ** ptr_to_splits_cache,
                    uint window_size,
                    vector<uint> &beam_sizes,
                    const at::Tensor & seq_lens);
void cuda_update_range(CellRange *ranges, size_t batch_size);
void cuda_update_cache_ids(const at::Tensor & cache_ids,
                           const at::Tensor & detached_cache_ids,
                           const at::Tensor & span_lens,
                           const at::Tensor & update_lens,
                           TreeNode *nodes_offset,
                           uint * split_cache,
                           uint &cell_idx_offset,
                           uint &total_nodes,
                           TableCell ***active_cells,
                           CellRange *ranges,
                           uint **left_most,
                           uint **right_most,
                           uint *cell_offsets,
                           uint current_beam_size,
                           uint max_beam_size,
                           uint cache_id_offset,
                           uint current_group_size,
                           uint current_max_seq_len);
void cuda_beam_select(const at::Tensor & indices, TableCell ***active_cells, CellRange *ranges, uint *cell_offsets,
                      uint beam_size, size_t batch_size);
int cuda_assign_log_p(const at::Tensor & log_p_batch,
                      const at::Tensor & candidates_log_p,
                      float *log_p_offset,
                      TableCell ***active_cells,
                      CellRange *ranges,
                      uint *cell_offsets,
                      uint max_beam_size,
                      uint current_group_size,
                      uint current_max_seq_len,
                      size_t batch_size);
void cuda_update_log_p(TableCell ***active_cells, CellRange *ranges, uint max_len,
                       uint beam_size, uint window_size, size_t batch_size);
void cuda_prune_and_update(const at::Tensor & bigram_score_cache, const at::Tensor & noise, TableCell ***active_cells, TableCell **all_cells,
                           CellRange *udate_ranges, size_t batch_size);
void cuda_dump_cells(vector<vector<ExportCell>> &out, const at::Tensor & seq_lens, TableCell **all_cells, CellRange *ranges, uint beam_size);
void cuda_gather_bilm(const at::Tensor & cache_ids, TableCell **all_cells, uint **left_most, uint **right_most,
                      uint empty_cache_id, uint beam_size, const at::Tensor & seq_lens, uint bos, uint eos, uint max_seq_len);
void cuda_prune_along_trajectory(TableCell ***active_cells, TableCell **all_cells, CellRange *ranges, const at::Tensor & merge_positions, int current_step);
void cuda_recover_sample_trees(const at::Tensor & span_masks, const at::Tensor & targets, const at::Tensor & split_points, TableCell ** all_cells, at::Tensor seq_lens);
template <typename T>
void cuda_free_array(T *cuda_arr, uint size);

template <typename T>
void cuda_free_ptr(T *cuda_ptr);

void print_left_most(uint **left_most, char *msg);