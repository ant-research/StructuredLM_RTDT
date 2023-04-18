// coding=utf-8
// Copyright (c) 2022 Ant Group
// Author: Xiang Hu

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "common.h"
#include "r2d2lib.h"

using namespace std;

#define PASS_IF_FINISHED(x) if(x.end < 0) return
#define ASSERT(x) //assert(x)

int MAX_THREAD;
const float MIN_LOG_P = -9999;
const float P_LOWERBOUND = 0.0001;

template <typename scalar_t>
__global__ void assign_cells(TableCell *** active_cells, TableCell ** cells, scalar_t * seq_lens) {
    int batch_i = blockIdx.x;
    int layer_i = blockIdx.y;
    int pos = threadIdx.x;
    int seq_len = (int)seq_lens[batch_i];
    while (pos < seq_len) {
        if(pos + layer_i < seq_len) {
            active_cells[batch_i][layer_i * seq_len + pos] = &cells[batch_i][pos * seq_len + pos + layer_i];
        } else {
            active_cells[batch_i][layer_i * seq_len + pos] = 0;
        }
        pos += blockDim.x;
    }
}

template <typename scalar_t>
__global__ void init_cells(TableCell ** cells, scalar_t * seq_lens) {
    int batch_i = blockIdx.x;
    int thrd_idx = threadIdx.x;
    int seq_len = (int)seq_lens[batch_i];
    int i, j;
    TableCell * self;
    while (thrd_idx < seq_len * seq_len) {
        i = thrd_idx / seq_len;
        j = thrd_idx % seq_len;
        /*
        struct TableCell
        {
            TreeNode *beams;
            float *candidates_log_p; //keep the log_p of all possible combinations
            bool eliminated;         //whether the cell is pruned from the table.
            float max_log_p;
            float max_left_log_p_;
            float max_right_log_p_;
            uint beam_size;
            uint best_tree_idx;
            uint i;
            uint j;
        };
        */
        self = &cells[batch_i][i * seq_len + j];
        self->candidates_log_p = 0;
        self->beams = 0;
        self->beam_size = 0;
        self->best_tree_idx = 0;
        self->splits = 0;
        self->eliminated = false;
        self->is_term = i == j;
        self->i = i;
        self->j = j;
        self->cell_idx = -1;
        self->max_log_p = MIN_LOG_P;
        self->max_left_log_p_ = 0;
        self->max_right_log_p_ = 0;
        self->table_id = batch_i;
        thrd_idx += blockDim.x;
    }
}

template <typename scalar_t>
__global__ void init_ranges(CellRange * ranges, scalar_t * seq_lens) {
    int batch_i = blockIdx.x;
    ranges[batch_i].layer_i = 0;
    ranges[batch_i].start = 0;
    ranges[batch_i].end = (int)seq_lens[batch_i] - 1;
    ranges[batch_i].seq_len = (int)seq_lens[batch_i];
    ranges[batch_i].term_len = (int)seq_lens[batch_i];
}

__global__ void init_tree_nodes(TreeNode * nodes, uint offset, uint nodes_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nodes_len) {
        /*
            uint cache_id;
            TreeNode *left;
            TreeNode *right;
            TableCell *owner;
            float log_p;
        */
        nodes[i].cache_id = i + offset;
        nodes[i].left = 0;
        nodes[i].right = 0;
        nodes[i].owner = 0;
        nodes[i].log_p = 0;
    }
}

__global__ void assign_tree_nodes(TableCell *** active_cells, 
                                  TreeNode * nodes, 
                                  CellRange * ranges, 
                                  uint * node_offset,
                                  uint beam_size,
                                  uint cell_idx_offset,
                                  uint * splits) {
    int batch_i = blockIdx.x;
    PASS_IF_FINISHED(ranges[batch_i]);
    uint layer_i = ranges[batch_i].layer_i;
    uint seq_len = ranges[batch_i].seq_len;
    uint offset = ranges[batch_i].start;
    uint len = ranges[batch_i].end - offset + 1;
    uint depth = ranges[batch_i].layer_i;
    int i = threadIdx.x;
    int beam_i = 0;
    TableCell * active_cell;
    while (i < len) {
        active_cell = active_cells[batch_i][layer_i * seq_len + offset + i];
        if (depth > 0) {
            active_cell->splits = splits + (node_offset[batch_i] + i) * depth;
        }
        ASSERT(active_cell->cell_idx == -1);
        active_cell->cell_idx = cell_idx_offset + node_offset[batch_i] + i;
        ASSERT(active_cell->beams == 0);
        ASSERT(active_cell->beam_size == 0);
        active_cell->beams = &nodes[(node_offset[batch_i] + i) * beam_size];
        for (beam_i = 0; beam_i < beam_size; ++beam_i) {
            nodes[(node_offset[batch_i] + i) * beam_size + beam_i].owner = active_cell;
        }
        active_cell->beam_size = beam_size;
        i += blockDim.x;
    }
}

__global__ void update_range(CellRange * ranges) {
    int batch_i = blockIdx.x;
    if (ranges[batch_i].end >= 0) {
        ranges[batch_i].end -= 1;
        ranges[batch_i].layer_i += 1;
    }
}

__global__ void update_records(uint * records, int total_len) {
    int thread_idx = threadIdx.x;
    while (thread_idx < total_len) {
        records[thread_idx] = thread_idx;
        thread_idx += blockDim.x;
    }
}

__global__ void reset_cell_states(TableCell *** active_cells, CellRange * ranges) {
    // Reset max_log_p for children of updated cells, but cells in the last layer are excluded.
    int batch_i = blockIdx.x;
    PASS_IF_FINISHED(ranges[batch_i]);
    int thread_i = threadIdx.x;
    int depth = ranges[batch_i].layer_i;
    int offset = ranges[batch_i].start;
    int seq_len = ranges[batch_i].seq_len;
    int len = ranges[batch_i].end - offset + 1;
    int total_len = (depth + 1) * (len + depth);
    while (thread_i < total_len) {
        int layer_i = thread_i / (len + depth);
        int idx = thread_i % (len + depth);
        if (idx < len + depth - layer_i && layer_i != depth) {
            ASSERT(layer_i + offset + idx < ranges[batch_i].term_len);
            active_cells[batch_i][layer_i * seq_len + offset + idx]->max_log_p = MIN_LOG_P;
        }
        thread_i += blockDim.x;
    }
}

__global__ void print_splits(CellRange * ranges, TableCell *** active_cells) {
    int batch_i = blockIdx.x;
    PASS_IF_FINISHED(ranges[batch_i]);
    int depth = ranges[batch_i].layer_i;
    int start = ranges[batch_i].start;
    int end = ranges[batch_i].end;
    int seq_len = ranges[batch_i].seq_len;
    TableCell * cell;
    if (batch_i == 0) {
        for (int idx = start; idx < end; ++idx) {
            cell = active_cells[batch_i][seq_len * depth + idx];
            printf("table: %d, current span: (%d, %d), splits: ", batch_i, cell->i, cell->j);
            for (int k = 0; k < depth; ++k) {
                printf("%d, ", active_cells[batch_i][k * seq_len + idx]->j);
            }
            printf("\n");
        }
    }
}


__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ void update_sub_tree(TreeNode * root, float min_log_p, uint depth) {
    // max_log_p: min_log_p in all ancestor nodes.
    TableCell * owner = root->owner;
    ASSERT(owner != 0);
    ASSERT(depth >= 0);
    float mean_log_p = root->log_p;
    float new_log_p = min(min_log_p, mean_log_p); // get the min
    if (new_log_p > owner->max_log_p) {
        atomicMax(&owner->max_log_p, new_log_p); // get the max of all min_log_p;
    }
    if (!owner->is_term) {
        if (root->left && root->left->owner && !root->left->owner->eliminated) {
            update_sub_tree(root->left, new_log_p, depth - 1);
        }
        if (root->right && root->right->owner && !root->right->owner->eliminated) {
            update_sub_tree(root->right, new_log_p, depth - 1);
        }
    }
}

__global__ void update_log_p_top_down(TableCell *** active_cells, CellRange * ranges, int group_size) {
    int batch_i = blockIdx.x;
    PASS_IF_FINISHED(ranges[batch_i]);
    int seq_len = ranges[batch_i].seq_len;
    int term_len = ranges[batch_i].term_len;
    int depth = ranges[batch_i].layer_i;
    int offset = max((int)ranges[batch_i].start - depth, 0);
    int len = ranges[batch_i].end - offset + 1 + 2 * depth;
    int total_sz = len * group_size;
    int thread_i = threadIdx.x;
    int cell_i = 0;
    int beam_size = 0;
    int group_i, tmp, k, b_l, b_r;
    TableCell * cell, * cell_l, * cell_r;
    if (ranges[batch_i].term_len - depth > (uint)0) {
        while (thread_i < total_sz) {
            cell_i = thread_i / group_size;
            if (depth + offset + cell_i < term_len) {
                cell = active_cells[batch_i][depth * seq_len + offset + cell_i];
                beam_size = cell->beam_size;
                ASSERT(beam_size != 0);
    
                group_i = thread_i % group_size;
                k = group_i / (beam_size * beam_size);
                ASSERT(k < depth);
                tmp = group_i % (beam_size * beam_size);
                b_l = tmp / beam_size;
                b_r = tmp % beam_size;
                cell_l = active_cells[batch_i][k * seq_len + offset + cell_i];
                cell_r = active_cells[batch_i][(depth - 1 - k) * seq_len + offset + cell_i + k + 1];
                if (b_l < cell_l->beam_size && b_r < cell_r->beam_size) {
                    ASSERT(cell->candidates_log_p != 0);
                    update_sub_tree(&cell_l->beams[b_l], cell->candidates_log_p[group_i], depth);
                    update_sub_tree(&cell_r->beams[b_r], cell->candidates_log_p[group_i], depth);
                    ASSERT(cell->max_log_p > MIN_LOG_P);
                }
            }
            thread_i += blockDim.x;
        }
    }

}

__device__ void update_left_max_log_p(TableCell ** cells, int pos, int depth, int seq_len) {
    TableCell * cell = cells[pos];
    float max_log_p = MIN_LOG_P;
    for(int i = 1; i <= depth; ++i) {
        if (pos - i >= 0) {
            max_log_p = max(cells[i * seq_len + pos - i]->max_log_p, max_log_p);
        }
    }
    cell->max_left_log_p_ = log(max(1 - exp(max_log_p), P_LOWERBOUND));
}

__device__ void update_right_max_log_p(TableCell ** cells, int pos, int depth, int seq_len, int term_len) {
    TableCell * cell = cells[pos];
    float max_log_p = MIN_LOG_P;
    for(int i = 1; i <= depth; ++i) {
        if (i + pos < term_len) {
            max_log_p = max(cells[i * seq_len + pos]->max_log_p, max_log_p);
        }
    }
    cell->max_right_log_p_ = log(max(1 - exp(max_log_p), P_LOWERBOUND));
}

__global__ void update_left_right_max_log_p(TableCell *** active_cells, CellRange * ranges) {
    int batch_i = blockIdx.x;
    PASS_IF_FINISHED(ranges[batch_i]);
    int lr_i = blockIdx.y;
    int seq_len = ranges[batch_i].seq_len;
    int term_len = ranges[batch_i].term_len;
    int offset = ranges[batch_i].start;
    int depth = ranges[batch_i].layer_i;
    int update_len = (ranges[batch_i].end - offset + 1) + ranges[batch_i].layer_i;
    int thread_i = threadIdx.x;
    while (thread_i < update_len) {
        if (lr_i == 0) {
            update_left_max_log_p(active_cells[batch_i], offset + thread_i, depth, seq_len);
        } else {
            update_right_max_log_p(active_cells[batch_i], offset + thread_i, depth, seq_len, term_len);
        }
        thread_i += blockDim.x;
    }
}

template <typename T>
__device__ int index_of(T * ele, T * arr, uint size) {
    for (int i = 0; i < size; ++i) {
        if (arr + i == ele) {
            return i;
        }
    }
    return -1;
}

__global__ void dump_cells(ExportNode ** out_nodes_ptr, int ** best_idx, TableCell ** all_cells, 
    CellRange * ranges, uint beam_size) {
    /**struct ExportNode
        {
            int cache_i;
            int i_left;
            int i_right;
            int j_left;
            int j_right;
            int idx_left;
            int idx_right;
            float log_p;
        };
        **/
    int batch_i = blockIdx.x;
    int beam_i = blockIdx.y;
    int seq_len = ranges[batch_i].seq_len;
    int total_len = seq_len * seq_len;
    int thread_i = threadIdx.x;
    int i,j;
    TreeNode * target;
    ExportNode * self;
    while (thread_i < total_len) {
        i = thread_i / seq_len;
        j = thread_i % seq_len;
        if (beam_i == 0) {
            best_idx[batch_i][i * seq_len + j] = all_cells[batch_i][i * seq_len + j].best_tree_idx;
        }
        self = &out_nodes_ptr[batch_i][thread_i * beam_size + beam_i];
        self->cache_id = -1;
        self->left_i = -1;
        self->right_i = -1;
        self->left_j = -1;
        self->right_j = -1;
        self->left_idx = -1;
        self->right_idx = -1;
        self->log_p = 0;
        if (i <= j && beam_i < all_cells[batch_i][i * seq_len + j].beam_size) {
            target = &all_cells[batch_i][i * seq_len + j].beams[beam_i];
            self->cache_id = target->cache_id;
            if (target->left && target->right) {
                self->log_p = target->log_p;
                self->left_i = target->left->owner->i;
                self->right_i = target->right->owner->i;
                self->left_j = target->left->owner->j;
                self->right_j = target->right->owner->j;
                self->left_idx = index_of(target->left, target->left->owner->beams, target->left->owner->beam_size);
                self->right_idx = index_of(target->right, target->right->owner->beams, target->right->owner->beam_size);
            }
        }
        thread_i += blockDim.x;
    }
}

template<typename scalar_t>
__global__ void update_cache_ids(scalar_t *cache_ids,
                                 scalar_t *detached_cache_ids,
                                 TableCell ***active_cells,
                                 CellRange *ranges,
                                 uint ** left_records,
                                 uint ** right_records,
                                 uint * node_offsets,
                                 uint beam_size) {
    int batch_i = blockIdx.x;
    PASS_IF_FINISHED(ranges[batch_i]);
    uint seq_len = ranges[batch_i].seq_len;
    uint depth = ranges[batch_i].layer_i;
    uint group_size = depth * beam_size * beam_size;
    uint temp = 0;
    uint offset = ranges[batch_i].start;
    uint total_len = (ranges[batch_i].end - offset + 1) * group_size;
    TableCell * left_cell, *right_cell, *cell;
    int thread_idx = threadIdx.x;

    int cell_i, k, b_l, b_r, cache_idx;
    while (thread_idx < total_len) {
        cell_i = thread_idx / group_size;
        temp = thread_idx % group_size;
        k = temp / (beam_size * beam_size);
        temp = temp % (beam_size * beam_size);
        b_l = temp / beam_size;
        b_r = temp % beam_size;
        ASSERT(k >= 0);
        ASSERT(b_l >= 0 && b_r >= 0);
        cell = active_cells[batch_i][depth * seq_len + offset + cell_i];
        if (thread_idx % group_size == 0) {
            ASSERT(depth * seq_len + offset + cell_i < seq_len * seq_len);
            left_records[batch_i][cell->j] = (uint)cell->i;
            right_records[batch_i][cell->i] = (uint)cell->j;
        }
        if (k < depth) {
            ASSERT(k * seq_len + offset + cell_i < ranges[batch_i].layer_i * seq_len);
            left_cell = active_cells[batch_i][k * seq_len + offset + cell_i];
            right_cell = active_cells[batch_i][(depth - 1 - k) * seq_len + k + 1 + cell_i + offset];
            ASSERT(cell->splits[k] == 0 || cell->splits[k] == left_cell->j);
            cell->splits[k] = left_cell->j;
            ASSERT(right_cell->i == left_cell->j + 1);
            ASSERT(cell->i == left_cell->i);
            ASSERT(cell->j == right_cell->j);
            cache_idx = (node_offsets[batch_i] * group_size + thread_idx) * 2;
            if (b_l < left_cell->beam_size) {
                ASSERT(left_cell->beams[b_l].owner->table_id == batch_i);
                cache_ids[cache_idx] = left_cell->beams[b_l].cache_id;
                if (left_cell->is_term) {
                    detached_cache_ids[cache_idx] += left_cell->beams[b_l].cache_id;
                    ASSERT(k == 0 || depth - k - 1 == 0);
                } else {
                    detached_cache_ids[cache_idx] = left_cell->beams[b_l].cache_id;
                }
            } else {
                // set to empty cache id
                detached_cache_ids[cache_idx] = cache_ids[cache_idx];
            }
            if (b_r < right_cell->beam_size) {
                ASSERT(right_cell->beams[b_r].owner->table_id == batch_i);
                cache_ids[cache_idx + 1] = right_cell->beams[b_r].cache_id;
                if (right_cell->is_term) {
                    detached_cache_ids[cache_idx + 1] += right_cell->beams[b_r].cache_id;
                    ASSERT(k == 0 || depth - k - 1 == 0);
                } else {
                    detached_cache_ids[cache_idx + 1] = right_cell->beams[b_r].cache_id;
                }
            } else {
                detached_cache_ids[cache_idx + 1] = cache_ids[cache_idx + 1];
            }
        }
        thread_idx += blockDim.x;
    }
    ASSERT(left_records[0][0] == 0);
}

template <typename scalar_t>
__global__ void update_span_lens(scalar_t *span_lens,
                                TableCell *** active_cells,
                                CellRange * ranges,
                                uint * node_offsets) {
    int batch_i = blockIdx.x;
    PASS_IF_FINISHED(ranges[batch_i]);
    uint cells_len = ranges[batch_i].end - ranges[batch_i].start + 1;
    uint offset = ranges[batch_i].start;
    uint depth = ranges[batch_i].layer_i;
    uint seq_len = ranges[batch_i].seq_len;
    uint span_offset = node_offsets[batch_i];
    int i = threadIdx.x;
    TableCell * cell;
    while (i < cells_len) {
        cell = active_cells[batch_i][depth * seq_len + offset + i];
        span_lens[span_offset + i] = cell->j - cell->i;
        i += blockDim.x;
    }
}

template<typename scalar_t>
__global__ void beam_select(scalar_t *indices, 
                            TableCell ***active_cells,
                            CellRange *ranges,
                            uint * node_offsets,
                            uint beam_size,
                            uint indices_len) {
    int batch_i = blockIdx.x;
    PASS_IF_FINISHED(ranges[batch_i]);
    uint offset = ranges[batch_i].start;
    uint cells_len = ranges[batch_i].end - offset + 1;
    uint seq_len = ranges[batch_i].seq_len;
    uint depth = ranges[batch_i].layer_i;
    uint total_len = cells_len * beam_size;
    uint index_i, temp, k, b_l, b_r;
    uint cell_i, beam_i;
    TableCell * cell_l, *cell_r, *cell;
    TreeNode *self, *left, *right;
    int i = threadIdx.x;
    while (i < total_len) {
        cell_i = i / beam_size;
        beam_i = i % beam_size;
        ASSERT(node_offsets[batch_i]*beam_size + i >= 0);
        ASSERT(node_offsets[batch_i]*beam_size + i < indices_len);
        index_i = (uint)indices[node_offsets[batch_i]*beam_size + i];
        k = index_i / (beam_size * beam_size);
        temp = index_i % (beam_size * beam_size);
        b_l = temp / beam_size;
        b_r = temp % beam_size;
        ASSERT(k < ranges[batch_i].layer_i);
        ASSERT(k >= 0);
        ASSERT (k * seq_len + offset + cell_i < ranges[batch_i].layer_i * seq_len);
        ASSERT (offset + cell_i < ranges[batch_i].term_len);
        cell_l = active_cells[batch_i][k * seq_len + offset + cell_i];
        cell_r = active_cells[batch_i][(depth - 1 - k) * seq_len + offset + k + 1 + cell_i];
        ASSERT (offset + cell_i < seq_len);
        cell = active_cells[batch_i][depth * seq_len + offset + cell_i];
        if (k < depth && b_l < cell_l->beam_size && b_r < cell_r->beam_size && beam_i < cell->beam_size) {
            left = &(cell_l->beams[b_l]);
            right = &(cell_r->beams[b_r]);
            self = &(cell->beams[beam_i]);
            self->left = left;
            self->right = right;
            // if (OB_BATCH(batch_i) && offset + cell_i == 0) {
            //     printf("self:[%d, %d].beam[%d], left: [%d, %d], right: [%d, %d]\n", self->owner->i, self->owner->j, beam_i,
            //             left->owner->i, left->owner->j, right->owner->i, right->owner->j);
            // }
        }
        i += blockDim.x;
    }
}

template <typename scalar_t>
__global__ void assign_log_p(scalar_t * log_p_batch,
                             TableCell ***active_cells, 
                             CellRange *ranges, 
                             uint *cell_offsets,
                             uint current_beam_size) {
    int batch_i = blockIdx.x;
    PASS_IF_FINISHED(ranges[batch_i]);
    int offset = ranges[batch_i].start;
    int len = ranges[batch_i].end - offset + 1;
    int seq_len = ranges[batch_i].seq_len;
    int depth = ranges[batch_i].layer_i;
    int batch_offset = cell_offsets[batch_i];
    int total_nodes = len * current_beam_size;
    int thrdIdx = threadIdx.x;
    int cell_i, beam_i, active_cell_idx;
    while (thrdIdx < total_nodes) {
        cell_i = thrdIdx / current_beam_size;
        beam_i = thrdIdx % current_beam_size;
        // ASSERT(beam_i < active_cells[batch_i][depth * seq_len + cell_i + offset]->beam_size);
        active_cell_idx = depth * seq_len + cell_i + offset;
        if (beam_i < active_cells[batch_i][active_cell_idx]->beam_size) {
            active_cells[batch_i][active_cell_idx]->beams[beam_i].log_p = 
                (float)log_p_batch[(batch_offset + cell_i) * current_beam_size + beam_i];
        }
        thrdIdx += blockDim.x;
        // if (OB_BATCH(batch_i)) {
        //     printf("assign log p: [%d][%d][%d].beam[%d]->%f\n", batch_i, depth, cell_i + offset, beam_i, 
        //             (float)log_p_batch[(batch_offset + cell_i) * current_beam_size + beam_i]);
        // }
    }
}

template <typename scalar_t>
__global__ void assign_candidates_log_p(scalar_t * candidates_log_p,
                                        scalar_t * candidates_max_log_p,
                                        float * log_p_ptr,
                                        TableCell *** active_cells,
                                        CellRange * ranges,
                                        uint * cell_offsets,
                                        uint current_beam_size,
                                        int log_p_total_len) {
    int batch_i = blockIdx.x;
    PASS_IF_FINISHED(ranges[batch_i]);
    int offset = ranges[batch_i].start;
    int len = ranges[batch_i].end - offset + 1;
    int seq_len = ranges[batch_i].seq_len;
    int thread_idx = threadIdx.x;
    int depth = ranges[batch_i].layer_i;
    int group_size =  depth * current_beam_size * current_beam_size;
    int total_len = len * group_size;
    int log_p_idx = 0;
    int cell_i, bias_i;
    TableCell * cell;
    while (thread_idx < total_len) {
        cell_i = thread_idx / group_size;
        bias_i = thread_idx % group_size;
        cell = active_cells[batch_i][depth * seq_len + offset + cell_i];
        if (bias_i == 0) {
            cell->candidates_log_p = &log_p_ptr[(cell_offsets[batch_i] + cell_i) * group_size];
            cell->max_log_p = candidates_max_log_p[cell_offsets[batch_i] + cell_i];
            ASSERT((cell_offsets[batch_i] + cell_i) * group_size < log_p_total_len);
            ASSERT(cell->beam_size == current_beam_size);
        }
        log_p_idx = (cell_offsets[batch_i] + cell_i) * group_size + bias_i;
        ASSERT(log_p_idx < log_p_total_len);
        log_p_ptr[log_p_idx] = (float)candidates_log_p[log_p_idx];
        thread_idx += blockDim.x;
    }
}

template <typename scalar_t>
__global__ void assign_best_node(scalar_t * best_indices,
                             TableCell ***active_cells, 
                             CellRange *ranges, 
                             uint *cell_offsets) {
    int batch_i = blockIdx.x;
    PASS_IF_FINISHED(ranges[batch_i]);
    int offset = ranges[batch_i].start;
    int len = ranges[batch_i].end - offset + 1;
    int seq_len = ranges[batch_i].seq_len;
    int depth = ranges[batch_i].layer_i;
    int batch_offset = cell_offsets[batch_i];
    int thrdIdx = threadIdx.x;

    while (thrdIdx < len) {
        active_cells[batch_i][depth * seq_len + thrdIdx + offset]->best_tree_idx = (uint)best_indices[batch_offset + thrdIdx];
        thrdIdx += blockDim.x;
    }
}

template <typename scalar_t>
__global__ void gather_bigram_score(scalar_t * bigram_scores,
                             TableCell ***active_cells, 
                             CellRange *ranges,
                             uint tensor_dim) {
    int batch_i = blockIdx.x;
    PASS_IF_FINISHED(ranges[batch_i]);
    int seq_len = ranges[batch_i].seq_len;
    int total_len = ranges[batch_i].term_len - 1;
    int thread_idx = threadIdx.x;
    TableCell * cell, *left_down, *right_down;
    while (thread_idx < total_len) {
        cell = active_cells[batch_i][seq_len + thread_idx];
        left_down = active_cells[batch_i][thread_idx];
        right_down = active_cells[batch_i][thread_idx + 1];
        bigram_scores[batch_i * tensor_dim + thread_idx] = max(MIN_LOG_P, 
            cell->max_log_p + min(left_down->max_left_log_p_, right_down->max_right_log_p_));
        // ASSERT(bigram_scores[batch_i * tensor_dim + thread_idx] > MIN_LOG_P);
        thread_idx += blockDim.x;
    }
}

template <typename scalar_t>
__global__ void print_array(scalar_t * arr, size_t size) {
    int thread_idx = threadIdx.x;
    while (thread_idx < size) {
        printf("arr[%d]=%d\n", thread_idx, (int)arr[thread_idx]);
        thread_idx += blockDim.x;
    }
}

template <typename scalar_t>
__global__ void prune_and_update_range(scalar_t * merge_positions, TableCell *** active_cells, TableCell ** all_cells,
                                       CellRange * ranges, size_t batch_size) {
    int batch_i = blockIdx.x;
    PASS_IF_FINISHED(ranges[batch_i]);
    int thread_idx = threadIdx.x;
    int depth = ranges[batch_i].layer_i;
    int term_len = ranges[batch_i].term_len;
    int seq_len = ranges[batch_i].seq_len;
    // printf("scalar_t size: %d\n", sizeof(scalar_t));
    // printf("long size: %d\n", sizeof(long));
    // uint merge_pos = 0;  //(uint)merge_positions[batch_i]; //i , i + 1
    scalar_t merge_pos = merge_positions[batch_i];
    // __syncthreads()
    ASSERT(batch_i < batch_size);
    // uint merge_pos = 0;
    int total_len = (depth + 1) * term_len;
    int layer_i, pos;
    int target_layer_i, target_pos;
    TableCell * target;

    ASSERT(merge_pos >= 0);
    ASSERT(merge_pos < term_len);

    if (term_len > depth + 1) {
        int total_turns = ceil(float(total_len) / blockDim.x);
        if (thread_idx == 0 && total_turns > 0) {
            ranges[batch_i].start = ranges[batch_i].seq_len;
            ranges[batch_i].end = 0;
            --ranges[batch_i].term_len;
        }
        __syncthreads();
        while(total_turns > 0) {
            if (thread_idx < total_len) {
                target = 0;
                pos = thread_idx / (1 + depth);
                layer_i = depth - thread_idx % (1 + depth);
                pos = pos - layer_i;
                if (pos >= 0 && pos + layer_i < term_len - 1) {
                    if (layer_i + pos == merge_pos || pos == merge_pos + 1) {
                        active_cells[batch_i][layer_i * seq_len + pos]->eliminated = true;
                    }
                    target_layer_i = layer_i;
                    target_pos = pos;
                    if (layer_i + pos >= merge_pos) {
                        if (pos <= merge_pos) {
                            target_layer_i = layer_i + 1;
                        } else {
                            target_pos = pos + 1;
                        }
                    }
                    if (target_layer_i <= depth) {
                        target = active_cells[batch_i][target_layer_i * seq_len + target_pos];
                    } else {
                        target_layer_i = active_cells[batch_i][layer_i * seq_len + pos]->i;
                        target_pos = active_cells[batch_i][layer_i * seq_len + pos + 1]->j;
                        target = &all_cells[batch_i][target_layer_i * seq_len + target_pos];
                        atomicMin(&ranges[batch_i].start, pos);
                        atomicMax(&ranges[batch_i].end, pos);
                    }
                    if (target_layer_i == 1 && target_pos == merge_pos) {
                        target->is_term = true;
                    }
                }
            }
            __syncthreads();
            if (thread_idx < total_len && pos >= 0) {
                if (pos + layer_i < term_len - 1) {
                    active_cells[batch_i][layer_i * seq_len + pos] = target;
                } else if (pos + layer_i == term_len - 1) {
                    active_cells[batch_i][layer_i * seq_len + pos] = 0;
                }
            }
            thread_idx += blockDim.x;
            total_turns--;
        }
    } else {
        if (thread_idx == 0) {
            ranges[batch_i].start = 0;
            ranges[batch_i].end = -1;
        }
    }
}

__device__ int get_cache_id(TableCell * cell, int i, int j, int beam_i, uint seq_len, uint empty_cache_id) {
    if (beam_i < cell[i * seq_len + j].beam_size) {
        ASSERT(cell[i * seq_len + j].beams[beam_i].cache_id >= 0);
        return cell[i * seq_len + j].beams[beam_i].cache_id;
    } else {
        return empty_cache_id;
    }
}

template <typename scalar_t>
__global__ void recover_sampled_trees(scalar_t * span_masks, scalar_t * targets, scalar_t * split_points,
                                      TableCell ** all_cells, scalar_t * seq_lens, TableCell ** caches,
                                      size_t num_samples, size_t max_len, size_t split_points_len) {
    int batch_i = blockIdx.x;
    int sample_id = threadIdx.x;
    scalar_t seq_len = seq_lens[batch_i];
    TableCell ** cache = caches + batch_i * num_samples * max_len + sample_id * max_len;
    // split_points: (total_cells, K)
    // span_masks: (batch_size, K, L, L)
    // target: (batch_size, K, L)
    scalar_t * span_mask_offset = span_masks + max_len * max_len * num_samples * batch_i + max_len * max_len * sample_id;
    scalar_t * target_offset = targets + max_len * num_samples * batch_i + max_len * sample_id;
    int front = 0;
    cache[front] = &all_cells[batch_i][seq_len - 1];
    front++;
    int k = 0;
    int step = 0;
    TableCell * top;
    while(front > 0) {
        top = cache[front - 1];
        front--;
        if (top->i < top->j) {
            ASSERT(top->cell_idx * num_samples + sample_id < split_points_len);
            ASSERT(split_points[top->cell_idx * num_samples + sample_id] < top->j - top->i);
            k = top->splits[split_points[top->cell_idx * num_samples + sample_id]];
            ASSERT(k >= top->i);
            ASSERT(k < top->j);
            for (int pos = top->i; pos < top->j; ++pos) {
                span_mask_offset[step * max_len + pos] = 1;
            }
            target_offset[step] = k;
            if (k > top->i) {
                cache[front] = &all_cells[batch_i][top->i * seq_len + k];
                front++;
            }
            if (k + 1 < top->j) {
                cache[front] = &all_cells[batch_i][(k + 1) * seq_len + top->j];
                front++;
            }
        }
        ++step;
    }
    ASSERT(step == seq_len - 1);
}

template <typename scalar_t>
__global__ void gather_bilm(scalar_t * cache_ids,
                            TableCell ** all_cells,
                            uint * table_offsets_ptr,
                            uint **left_most,
                            uint **right_most,
                            scalar_t * seq_lens,
                            uint empty_cache_id,
                            uint beam_size,
                            uint bos,
                            uint eos) {
    int batch_i = blockIdx.x;
    int thread_i = threadIdx.x;
    int cache_id_offset;
    int cell_i, beam_i;
    uint seq_len = (uint)seq_lens[batch_i];
    uint total_len = seq_len * beam_size;
    while (thread_i < total_len) {
        cache_id_offset = 2 * (table_offsets_ptr[batch_i] * beam_size + thread_i);
        cell_i = thread_i / beam_size;
        beam_i = thread_i % beam_size;

        if (cell_i == 0) {
            cache_ids[cache_id_offset] = bos;
        } else {
            cache_ids[cache_id_offset] = get_cache_id(all_cells[batch_i], 
                left_most[batch_i][cell_i - 1], cell_i - 1, beam_i, seq_len, empty_cache_id);
        }
        if (cell_i == seq_len - 1) {
            cache_ids[cache_id_offset + 1] = eos;
        } else {
            cache_ids[cache_id_offset + 1] = get_cache_id(all_cells[batch_i],
                cell_i + 1, right_most[batch_i][cell_i + 1], beam_i, seq_len, empty_cache_id);
        }

        thread_i += blockDim.x;
    }
}

template<typename scalar_t>
__global__ void update_trajectories(scalar_t * merge_positions, scalar_t * current_action, uint max_len) {
    int batch_i = blockIdx.x;
    int pos = threadIdx.x;
    while (pos < max_len) {
        if(merge_positions[batch_i * max_len + pos] > current_action[batch_i]) {
            merge_positions[batch_i * max_len + pos] -= 1;
        }
        pos += blockDim.x;
    }
}

template<typename scalar_t>
__global__ void copy_ranges(scalar_t * dst, CellRange * ranges) {
    int batch_i = blockIdx.x;
    dst[batch_i] = ranges[batch_i].end - ranges[batch_i].start + 1;
}

template<typename scalar_t>
__global__ void copy_cumsum(uint * dst, scalar_t * src) {
    int batch_i = blockIdx.x;
    if (batch_i > 0) {
        dst[batch_i] = (uint)src[batch_i - 1];
    } else {
        dst[batch_i] = 0;
    }
}

void cuda_init() {
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    MAX_THREAD = prop.maxThreadsPerBlock;
}

void cuda_new_cells(void **ptr_to_active_cells, void **ptr_to_cells, const at::Tensor & seq_lens, uint window_size) {
    at::Tensor seq_lens_arr = seq_lens.to("cpu");
    HANDLE_ERROR(cudaMalloc(ptr_to_active_cells, seq_lens.size(0) * sizeof(TableCell**)));
    HANDLE_ERROR(cudaMalloc(ptr_to_cells, seq_lens.size(0) * sizeof(TableCell*)));
    TableCell ** cells = new TableCell*[seq_lens.size(0)];
    TableCell *** cell_ptrs = new TableCell**[seq_lens.size(0)];
    int max_seq_len = 0;
    AT_DISPATCH_INTEGRAL_TYPES(seq_lens_arr.scalar_type(), "cuda_malloc_cells", ([&]{
        max_seq_len = (int)*std::get<0>(seq_lens_arr.max(0)).data_ptr<scalar_t>();
        for (int batch_i = 0; batch_i < seq_lens.size(0); ++batch_i) {
            int seq_len = *seq_lens_arr.index({batch_i}).data_ptr<scalar_t>();
            seq_len = max(seq_len, 1);
            HANDLE_ERROR(cudaMalloc((void**)&cells[batch_i], seq_len * seq_len * sizeof(TableCell)));
            HANDLE_ERROR(cudaMalloc((void**)&cell_ptrs[batch_i], (window_size + 1) * seq_len * sizeof(TableCell*)));
            HANDLE_ERROR(cudaMemset((void*)cells[batch_i], 0, seq_len * seq_len * sizeof(TableCell)));
        }
    }));
    HANDLE_ERROR(cudaMemcpy(*ptr_to_cells, cells, sizeof(TableCell*) * seq_lens.size(0), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(*ptr_to_active_cells, cell_ptrs, sizeof(TableCell**) * seq_lens.size(0), cudaMemcpyHostToDevice));
    delete cells;
    delete cell_ptrs;
    
    const dim3 assign_grid(seq_lens.size(0), window_size + 1, 1);
    const uint blockDim = max_seq_len < MAX_THREAD ? max_seq_len : MAX_THREAD;
    const uint max_cells_len = min(max_seq_len * max_seq_len, MAX_THREAD);
    AT_DISPATCH_INTEGRAL_TYPES(seq_lens.scalar_type(), "init cells", ([&] {
        assign_cells<scalar_t><<<assign_grid, blockDim>>>((TableCell***)*ptr_to_active_cells, 
                                    (TableCell**)*ptr_to_cells, seq_lens.data_ptr<scalar_t>());
        init_cells<scalar_t><<<seq_lens.size(0), max_cells_len>>>((TableCell**)*ptr_to_cells, seq_lens.data_ptr<scalar_t>());
    }));
}

void cuda_init_cell_ranges(void ** ptr_to_ranges, void ** ptr_to_celloffsets, const at::Tensor & seq_lens) {
    HANDLE_ERROR(cudaMalloc(ptr_to_ranges, seq_lens.size(0) * sizeof(CellRange)));
    AT_DISPATCH_INTEGRAL_TYPES(seq_lens.scalar_type(), "init_ranges", ([&] {
        init_ranges<scalar_t><<<seq_lens.size(0), 1>>>((CellRange*)*ptr_to_ranges, seq_lens.data_ptr<scalar_t>());
    }));

    HANDLE_ERROR(cudaMalloc(ptr_to_celloffsets, (seq_lens.size(0) + 1) * sizeof(uint)));
    HANDLE_ERROR(cudaMemset(*ptr_to_celloffsets, 0, (seq_lens.size(0) + 1) * sizeof(uint)));
}

void cuda_init_misc(uint ***ptr_to_leftrecord,
                    uint ***ptr_to_rightrecord,
                    TreeNode **nodes_cache,
                    float **log_p_cache,
                    uint ** split_points,
                    uint window_size,
                    vector<uint> &beam_sizes,
                    const at::Tensor & seq_lens) {
    at::Tensor seq_lens_arr = seq_lens.to("cpu");
    int nodes_len = 0;
    int log_p_len = 0;
    int splits_cache_len = 0;
    AT_DISPATCH_INTEGRAL_TYPES(seq_lens_arr.scalar_type(), "calculate_cache_len", ([&]{
        for (int seq_i = 0; seq_i < seq_lens_arr.size(0); seq_i++) {
            int seq_len = (int)*seq_lens_arr.index({seq_i}).data_ptr<scalar_t>();
            for (int layer_i = 0; layer_i < seq_len; ++layer_i) {
                int beam_size = 0;
                if (layer_i < beam_sizes.size()) {
                    beam_size = beam_sizes[layer_i];
                } else {
                    beam_size = window_size;
                }
                if (layer_i <= window_size) {
                    nodes_len += (seq_len - layer_i) * beam_size;
                    log_p_len += (seq_len - layer_i) * window_size * beam_size * beam_size;
                    splits_cache_len += (seq_len - layer_i) * window_size;
                } else {
                    nodes_len += (window_size + 1) * beam_size;
                    log_p_len += (window_size + 1) * window_size * beam_size * beam_size;
                    splits_cache_len += (window_size + 1) * window_size;
                }
            }
        }
    }));
    HANDLE_ERROR(cudaMalloc(nodes_cache, nodes_len * sizeof(TreeNode)));
    HANDLE_ERROR(cudaMalloc(log_p_cache, log_p_len * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(split_points, splits_cache_len * sizeof(uint)));
    HANDLE_ERROR(cudaMemset(*split_points, 0, splits_cache_len * sizeof(uint)));

    uint ** left_records = new uint*[seq_lens.size(0)];
    uint ** right_records = new uint*[seq_lens.size(0)];
    AT_DISPATCH_INTEGRAL_TYPES(seq_lens.scalar_type(), "init_bilm_records", ([&]{
        for(int i = 0; i < seq_lens.size(0); ++i) {
            int seq_len = (int)*seq_lens_arr.index({i}).data_ptr<scalar_t>();
            seq_len = max(1, seq_len);
            HANDLE_ERROR(cudaMalloc(&left_records[i], seq_len * sizeof(uint)));
            HANDLE_ERROR(cudaMalloc(&right_records[i], seq_len * sizeof(uint)));
            update_records<<<1, seq_len>>>(left_records[i], seq_len);
            update_records<<<1, seq_len>>>(right_records[i], seq_len);
        }
    }));
    size_t arr_size = seq_lens.size(0) * sizeof(uint *);
    HANDLE_ERROR(cudaMalloc(ptr_to_leftrecord, arr_size));
    HANDLE_ERROR(cudaMalloc(ptr_to_rightrecord, arr_size));
    HANDLE_ERROR(cudaMemcpy(*ptr_to_leftrecord, left_records, arr_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(*ptr_to_rightrecord, right_records, arr_size, cudaMemcpyHostToDevice));

    delete left_records;
    delete right_records;
}

template <typename T>
void cuda_free_ptr(T* cuda_ptr) {
    HANDLE_ERROR(cudaFree(cuda_ptr));
}
template void cuda_free_ptr(CellRange *);
template void cuda_free_ptr(uint *);
template void cuda_free_ptr(TreeNode *);
template void cuda_free_ptr(float *);

template <typename T>
void cuda_free_array(T *cuda_arr, uint size) {
    T * host_batch = new T[size];
    HANDLE_ERROR(cudaMemcpy(host_batch, cuda_arr, size * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        HANDLE_ERROR(cudaFree(host_batch[i]));
    }
    HANDLE_ERROR(cudaFree(cuda_arr));
    delete host_batch;
}
template void cuda_free_array(TableCell ***, uint);
template void cuda_free_array(TableCell **, uint);
template void cuda_free_array(uint **, uint);


void cuda_update_range(CellRange *ranges, size_t batch_size) {
    const size_t block_dim = batch_size;
    update_range<<<block_dim, 1>>>(ranges);
}

void cuda_update_cache_ids(const at::Tensor & cache_ids,
                           const at::Tensor & detached_cache_ids,
                           const at::Tensor & span_lens,
                           const at::Tensor & update_lens,
                           TreeNode * nodes_offset,
                           uint * splits_offset,
                           uint &cell_idx_offset,
                           uint &total_nodes,
                           TableCell ***active_cells,
                           CellRange *ranges,
                           uint ** left_most,
                           uint ** right_most,
                           uint * cell_offsets,
                           uint current_beam_size,
                           uint max_beam_size,
                           uint cache_id_offset,
                           uint current_group_size,
                           uint current_max_seq_len) {
    int seq_len_sum = 0;
    int batch_size = update_lens.size(0);
    AT_DISPATCH_INTEGRAL_TYPES(update_lens.scalar_type(), "copy_to_tensor", ([&]{
        copy_ranges<scalar_t><<<batch_size, 1>>>(update_lens.data_ptr<scalar_t>(), ranges);
    }));
    at::Tensor total_len = update_lens.sum(0);
    AT_DISPATCH_INTEGRAL_TYPES(total_len.scalar_type(), "sum seq lens", ([&] {
        scalar_t value;
        HANDLE_ERROR(cudaMemcpy(&value, total_len.data_ptr<scalar_t>(), sizeof(scalar_t), cudaMemcpyDeviceToHost));
        seq_len_sum = (int)value;
    }));

    total_nodes = seq_len_sum * current_beam_size;
    at::Tensor lens_cumsum = update_lens.cumsum(0);
    AT_DISPATCH_INTEGRAL_TYPES(lens_cumsum.scalar_type(), "assign_cell_offsets", [&] {
        copy_cumsum<scalar_t><<<update_lens.size(0) + 1, 1>>>(cell_offsets, lens_cumsum.data_ptr<scalar_t>());
    });
    TreeNode * tree_nodes = nodes_offset;
    const size_t blockNum = (total_nodes / MAX_THREAD) + 1;
    ASSERT(blockNum > 0);
    ASSERT(current_group_size > 0);
    ASSERT(current_max_seq_len > 0);
    init_tree_nodes<<<blockNum, min(MAX_THREAD, total_nodes)>>>(tree_nodes, cache_id_offset, total_nodes);
    // associate cells with tree nodes;
    assign_tree_nodes<<<batch_size, min(MAX_THREAD, current_max_seq_len * current_beam_size)>>>
        (active_cells, tree_nodes, ranges, cell_offsets, current_beam_size, cell_idx_offset, splits_offset);
    cell_idx_offset += seq_len_sum;
    AT_DISPATCH_INTEGRAL_TYPES(cache_ids.scalar_type(), "update_cache_ids", ([&] {
        const int block_sz = current_group_size * current_max_seq_len;
        update_cache_ids<scalar_t><<<batch_size, min(MAX_THREAD, block_sz)>>>(cache_ids.data_ptr<scalar_t>(), 
            detached_cache_ids.data_ptr<scalar_t>(), active_cells, 
            ranges, left_most, right_most, cell_offsets, current_beam_size);
    }));
    AT_DISPATCH_FLOATING_TYPES(span_lens.scalar_type(), "update_span_lens", ([&] {
        update_span_lens<scalar_t><<<batch_size, min(MAX_THREAD, current_max_seq_len)>>>(span_lens.data_ptr<scalar_t>(),
            active_cells, ranges, cell_offsets);
    }));
}

void cuda_beam_select(const at::Tensor & indices, 
                      TableCell ***active_cells, 
                      CellRange *ranges, 
                      uint *cell_offsets, 
                      uint beam_size,
                      size_t batch_size) {
    uint total_len = indices.size(0) * indices.size(1);
    ASSERT(indices.size(1) <= beam_size);
    AT_DISPATCH_INTEGRAL_TYPES(indices.scalar_type(), "cuda_beam_select", ([&] {
        beam_select<scalar_t><<<batch_size, min((int)indices.size(0), MAX_THREAD)>>>(
            indices.data_ptr<scalar_t>(), 
            active_cells, ranges, cell_offsets, indices.size(1), total_len);
    }));
}

int cuda_assign_log_p(const at::Tensor & log_p_batch,
                      const at::Tensor & candidates_log_p,
                      float * log_p_offset,
                      TableCell ***active_cells, 
                      CellRange *ranges, 
                      uint *cell_offsets,
                      uint max_beam_size,
                      uint current_group_size,
                      uint current_max_seq_len, 
                      size_t batch_size) {
    // log_p_batch: (total_len, beam_size)
    // candidates_log_p: (total_len, depth, beam_size * beam_size)
    int total_candidates_size = candidates_log_p.size(0) * candidates_log_p.size(1);
    at::Tensor max_log_p = std::get<0>(candidates_log_p.max(1));
    float * candidates_log_p_ptr = log_p_offset;
    const int total_len = log_p_batch.size(0);
    const int candidate_block_num = min(MAX_THREAD, current_group_size * current_max_seq_len);
    AT_DISPATCH_FLOATING_TYPES(candidates_log_p.scalar_type(), "cuda_assign_candidates_log_p", ([&] {
        assign_candidates_log_p<scalar_t><<<batch_size, candidate_block_num>>>(candidates_log_p.data_ptr<scalar_t>(), 
            max_log_p.data_ptr<scalar_t>(), candidates_log_p_ptr, active_cells, ranges, cell_offsets, 
            log_p_batch.size(1), total_candidates_size);
    }));
    AT_DISPATCH_FLOATING_TYPES(log_p_batch.scalar_type(), "cuda_assign_log_p", ([&] {
        assign_log_p<scalar_t><<<batch_size, min(current_max_seq_len * max_beam_size,  MAX_THREAD)>>>(
            log_p_batch.data_ptr<scalar_t>(), active_cells,
            ranges, cell_offsets, log_p_batch.size(1));
    }));
    at::Tensor max_indices = at::argmax(log_p_batch, 1);// log_p_batch: (batch_size, beam_size)
    AT_DISPATCH_INTEGRAL_TYPES(max_indices.scalar_type(), "assign_best_node", ([&] {
        assign_best_node<scalar_t><<<batch_size, min(current_max_seq_len, MAX_THREAD)>>>(
            max_indices.data_ptr<scalar_t>(), active_cells,
            ranges, cell_offsets);
    }));
    return total_candidates_size;
}

void cuda_update_log_p(TableCell *** active_cells, 
                       CellRange * ranges, 
                       uint max_len, 
                       uint beam_size, 
                       uint window_size, 
                       size_t batch_size) {
    reset_cell_states<<<batch_size, min(max_len * window_size, MAX_THREAD)>>>(active_cells, ranges);
    int group_size = window_size * beam_size * beam_size;
    update_log_p_top_down<<<dim3(batch_size, beam_size, 1), min(max_len, MAX_THREAD)>>>(active_cells, ranges, group_size);
    update_left_right_max_log_p<<<dim3(batch_size, 2, 1), min(max_len + window_size, MAX_THREAD)>>>(active_cells, ranges);
}

void cuda_prune_and_update(const at::Tensor & bigram_score_cache, 
                           const at::Tensor & noise,
                           TableCell *** active_cells, 
                           TableCell ** all_cells,
                           CellRange * ranges, 
                           size_t batch_size) {
    int tensor_dim = bigram_score_cache.size(1);
    AT_DISPATCH_FLOATING_TYPES(bigram_score_cache.scalar_type(), "gather_bigram_score", ([&] {
        gather_bigram_score<scalar_t><<<batch_size, MAX_THREAD>>>(bigram_score_cache.data_ptr<scalar_t>(), active_cells, ranges, tensor_dim);
    }));
    at::Tensor indices = at::argmax(bigram_score_cache + noise, 1); // (batch_size);
    AT_DISPATCH_INTEGRAL_TYPES(indices.scalar_type(), "prune_and_update_range", ([&] {
        prune_and_update_range<scalar_t><<<batch_size, MAX_THREAD>>>(indices.data_ptr<scalar_t>(), active_cells, all_cells, ranges, indices.size(0));
    }));
}

void cuda_prune_along_trajectory(TableCell ***active_cells, 
                                 TableCell **all_cells,
                                 CellRange *ranges,
                                 const at::Tensor & merge_positions,
                                 int prune_step) {
    at::Tensor current_action = merge_positions.index({"...", prune_step});
    current_action = current_action.contiguous();
    size_t batch_size = merge_positions.size(0);
    uint max_len = merge_positions.size(1);

    AT_DISPATCH_INTEGRAL_TYPES(current_action.scalar_type(), "prune_along_trajectory", ([&] {
        prune_and_update_range<scalar_t><<<batch_size, MAX_THREAD>>>(current_action.data_ptr<scalar_t>(), active_cells, all_cells, ranges, batch_size);
    }));
    AT_DISPATCH_INTEGRAL_TYPES(merge_positions.scalar_type(), "update_trajectories", ([&] {
        update_trajectories<scalar_t><<<batch_size, max_len>>>(merge_positions.data_ptr<scalar_t>(), 
                                                               current_action.data_ptr<scalar_t>(), max_len);
    }));
}

void cuda_dump_cells(vector<vector<ExportCell>> &out, const at::Tensor & seq_lens, TableCell **all_cells, 
    CellRange *ranges, uint beam_size) {
    at::Tensor cells_len = seq_lens * seq_lens;
    cells_len = cells_len.to("cpu");
    const int batch_size = seq_lens.size(0);

    AT_DISPATCH_INTEGRAL_TYPES(cells_len.scalar_type(), "dump_cells", ([&] {
        ExportNode** out_nodes = new ExportNode*[batch_size];
        int ** out_indices = new int*[batch_size];
        for (int batch_i = 0; batch_i < batch_size; ++batch_i) {
            int seq_len = *cells_len.index({batch_i}).data_ptr<scalar_t>();
            ExportNode * comb_ptr;
            HANDLE_ERROR(cudaMalloc(&comb_ptr, beam_size * seq_len * sizeof(ExportNode)));
            out_nodes[batch_i] = comb_ptr;
            int * indices_ptr;
            HANDLE_ERROR(cudaMalloc(&indices_ptr, seq_len * sizeof(int)));
            out_indices[batch_i] = indices_ptr;
        }
        ExportNode ** out_nodes_ptr;
        int ** out_indices_ptr;
        HANDLE_ERROR(cudaMalloc(&out_nodes_ptr, batch_size * sizeof(ExportNode*)));
        HANDLE_ERROR(cudaMalloc(&out_indices_ptr, batch_size * sizeof(int *)));
        HANDLE_ERROR(cudaMemcpy(out_nodes_ptr, out_nodes, batch_size * sizeof(ExportNode*), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(out_indices_ptr, out_indices, batch_size * sizeof(int*), cudaMemcpyHostToDevice));
        const dim3 grid_size(batch_size, beam_size, 1);
        dump_cells<<<grid_size, MAX_THREAD>>>(out_nodes_ptr, out_indices_ptr, all_cells, ranges, beam_size);
        for (int batch_i = 0; batch_i < batch_size; ++batch_i) {
            int seq_len = *cells_len.index({batch_i}).data_ptr<scalar_t>();
            ExportNode * combs = (ExportNode *)malloc(seq_len * beam_size * sizeof(ExportNode));
            int * best_inices = (int *)malloc(seq_len * sizeof(int));
            HANDLE_ERROR(cudaMemcpy(combs, out_nodes[batch_i], seq_len * beam_size * sizeof(ExportNode), 
                                    cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(best_inices, out_indices[batch_i], seq_len * sizeof(int), 
                                    cudaMemcpyDeviceToHost));
            out[batch_i] = std::vector<ExportCell>(seq_len);
            for (int iter = 0; iter < seq_len; ++iter) {
                out[batch_i][iter].best_tree_idx = best_inices[iter];
                out[batch_i][iter].nodes = std::vector<ExportNode>(beam_size);
                for (int beam_i = 0; beam_i < beam_size; ++beam_i) {
                    out[batch_i][iter].nodes[beam_i] = combs[iter * beam_size + beam_i];
                }
            }
            free(combs);
            free(best_inices);
        }
        for (int batch_i = 0; batch_i < batch_size; ++batch_i) {
            HANDLE_ERROR(cudaFree(out_nodes[batch_i]));
            HANDLE_ERROR(cudaFree(out_indices[batch_i]));
        }
        delete out_nodes;
        delete out_indices;
        HANDLE_ERROR(cudaFree(out_nodes_ptr));
        HANDLE_ERROR(cudaFree(out_indices_ptr));
    }));
}

void cuda_gather_bilm(const at::Tensor & cache_ids, 
                      TableCell **all_cells,
                      uint ** left_most, 
                      uint ** right_most, 
                      uint empty_cache_id,
                      uint beam_size,
                      const at::Tensor & seq_lens,
                      uint bos,
                      uint eos,
                      uint max_seq_len) {
    at::Tensor seq_len_cumsum = seq_lens.cumsum(0);
    uint * table_offsets_ptr;
    HANDLE_ERROR(cudaMalloc(&table_offsets_ptr, (seq_lens.size(0) + 1) * sizeof(uint)));
    AT_DISPATCH_INTEGRAL_TYPES(seq_len_cumsum.scalar_type(), "copy seq_len_cumsum", ([&] {
        copy_cumsum<scalar_t><<<seq_lens.size(0) + 1, 1>>>(table_offsets_ptr, seq_len_cumsum.data_ptr<scalar_t>());
    }));
    
    AT_DISPATCH_INTEGRAL_TYPES(cache_ids.scalar_type(), "gather_bilm", ([&] {
        gather_bilm<<<seq_lens.size(0), min(MAX_THREAD, max_seq_len)>>>(cache_ids.data_ptr<scalar_t>(), all_cells, table_offsets_ptr,
                    left_most, right_most, seq_lens.data_ptr<scalar_t>(), empty_cache_id, beam_size, bos, eos);
    }));
    HANDLE_ERROR(cudaFree(table_offsets_ptr));
}

void cuda_recover_sample_trees(const at::Tensor & span_masks,
                               const at::Tensor & targets,
                               const at::Tensor & split_points,
                               TableCell ** all_cells,
                               at::Tensor seq_lens) {
    TableCell ** caches;
    if (targets.size(2) > 0) {
        HANDLE_ERROR(cudaMalloc(&caches, targets.size(0) * targets.size(1) * targets.size(2) * sizeof(TableCell*)));
        if (seq_lens.scalar_type() != split_points.scalar_type()) {
            seq_lens = seq_lens.to(split_points.scalar_type());
        }
        AT_DISPATCH_INTEGRAL_TYPES(split_points.scalar_type(), "recover_tree", ([&] {
            recover_sampled_trees<<<span_masks.size(0), span_masks.size(1)>>>(span_masks.data_ptr<scalar_t>(), 
                targets.data_ptr<scalar_t>(), split_points.data_ptr<scalar_t>(), all_cells, 
                seq_lens.data_ptr<scalar_t>(), caches, span_masks.size(1), span_masks.size(2),
                split_points.size(0) * split_points.size(1));
        }));
        HANDLE_ERROR(cudaFree(caches));
    }
}