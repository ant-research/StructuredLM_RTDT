// coding=utf-8
// Copyright (c) 2022 Ant Group
// Author: Xiang Hu

#include "r2d2lib.h"
#include <assert.h>
#include <ATen/ATen.h>
#include <algorithm>
#include "kernel.h"
using namespace std;


void get_beam_uintable(std::vector<uint> &beam_table, uint max_beam_size, bool directional)
{
    uint current_beam_size = 1;
    uint span_len = 0;
    uint dir_factor = directional ? 2 : 1;
    uint k = 0;
    while (current_beam_size < max_beam_size)
    {
        beam_table.push_back(current_beam_size);
        span_len += 1;
        current_beam_size = 0;
        for (k = 0; k < span_len; ++k)
        {
            current_beam_size += beam_table[k] * beam_table[span_len - 1 - k] * dir_factor;
        }
    }
    beam_table.push_back(max_beam_size);
}

TablesManager::TablesManager(bool directional, uint window_size, uint beam_size):
    m_bEncoding{false}, m_iWindowSize{window_size}, m_bDirectional{directional},
    m_iBeamSize{beam_size}, m_iGroupSize{beam_size * beam_size * window_size}
{
    get_beam_uintable(this->m_vBeamSizes, beam_size, directional);
    cuda_init();
}

TablesManager::~TablesManager()
{
    assert(!this->m_bEncoding);
}

void TablesManager::encoding_start(const at::Tensor & seq_lens, uint cache_id_offset, uint detached_cache_id_offset, uint emtpy_cache_id)
{
    //initialize computing tables, alloc cells and
    this->m_bEncoding = true;
    this->m_bHasTrajectories = false;
    this->m_tSeqlens = seq_lens;
    this->m_iCurrentStep = 0;
    this->m_iCacheOffset = cache_id_offset;
    this->m_iDetachedCacheOffset = detached_cache_id_offset;
    this->m_iEmptyCacheId = emtpy_cache_id;
    this->m_iNodeOffset = 0;
    this->m_iLogProbOffset = 0;
    this->m_iCellIdxOffset = 0;
    std::tuple<at::Tensor, at::Tensor> result = seq_lens.max(0);
    at::Tensor total_len = seq_lens.sum();
    at::Tensor max_value = std::get<0>(result);
    AT_DISPATCH_INTEGRAL_TYPES(max_value.scalar_type(), "init_ranges", [&]
                               { this->m_iMaxSeqlen = (uint)*max_value.to("cpu").data_ptr<scalar_t>(); });
    AT_DISPATCH_INTEGRAL_TYPES(total_len.scalar_type(), "get_total_len", [&]
                               { this->m_iTotalLen = (uint)*total_len.to("cpu").data_ptr<scalar_t>(); });
    cuda_new_cells((void **)&this->m_pActiveCells, (void **)&this->m_pCells, seq_lens, this->m_iWindowSize);
    cuda_init_cell_ranges((void **)&this->m_pUpdateRange, (void **)&this->m_pCellOffsets, seq_lens);
    cuda_init_misc(&this->m_pLeftmost, &this->m_pRightmost,
                   &this->m_pNodeCache, &this->m_pLogProbCache, &this->m_pSplitCache,
                   this->m_iWindowSize, this->m_vBeamSizes, seq_lens);
}

void TablesManager::encoding_over()
{
    // destory cells

    this->m_bEncoding = false;
    this->m_bHasTrajectories = false;
    int batch_size = this->m_tSeqlens.size(0);
    cuda_free_array<TableCell **>(this->m_pActiveCells, batch_size);
    cuda_free_array<TableCell *>(this->m_pCells, batch_size);
    cuda_free_array<uint *>(this->m_pLeftmost, batch_size);
    cuda_free_array<uint *>(this->m_pRightmost, batch_size);
    cuda_free_ptr<CellRange>(this->m_pUpdateRange);
    cuda_free_ptr<uint>(this->m_pCellOffsets);
    cuda_free_ptr<uint>(this->m_pSplitCache);
    cuda_free_ptr<float>(this->m_pLogProbCache);
    cuda_free_ptr<TreeNode>(this->m_pNodeCache);
    this->m_tSeqlens = torch::Tensor();
}

vector<uint> TablesManager::step(const at::Tensor & cache_ids, const at::Tensor & log_p_ids, const at::Tensor & span_lens,
                                 const at::Tensor & bigram_score_cache, const at::Tensor & noise)
{
    uint prev_cell_offset = this->m_iCellIdxOffset;
    uint node_created = 0;
    uint group_size = 0;
    uint cache_id_offset = this->m_iCacheOffset;
    uint current_max_len = this->m_iMaxSeqlen;

    if (this->m_iCurrentStep > this->m_iWindowSize)
    {
        if (!this->m_bHasTrajectories) {
            // select best bigram position
            cuda_prune_and_update(bigram_score_cache, noise, this->m_pActiveCells, this->m_pCells, this->m_pUpdateRange,
                                this->m_tSeqlens.size(0));
        } else {
            // prune according to the trajectory
            cuda_prune_along_trajectory(this->m_pActiveCells, this->m_pCells, this->m_pUpdateRange, this->m_tMergePos, 
                                        this->m_iCurrentStep - this->m_iWindowSize - 1);
        }
        current_max_len = this->m_iWindowSize + 1;
    }
    span_lens.fill_(0.0);
    //create tree nodes
    uint beam_size = this->m_vBeamSizes[min(this->m_iCurrentStep, (uint)this->m_vBeamSizes.size() - 1)];
    group_size = min(this->m_iCurrentStep, this->m_iWindowSize) * beam_size * beam_size;
    if (group_size == 0)
    {
        group_size = 1; //group size > 0
    }
    cuda_update_cache_ids(cache_ids, log_p_ids,
                          span_lens, this->m_tSeqlens.clone(), this->m_pNodeCache + this->m_iNodeOffset,
                          this->m_pSplitCache + this->m_iCellIdxOffset * this->m_iWindowSize,
                          this->m_iCellIdxOffset, node_created, this->m_pActiveCells, this->m_pUpdateRange,
                          this->m_pLeftmost, this->m_pRightmost,
                          this->m_pCellOffsets, beam_size, this->m_iBeamSize,
                          this->m_iCacheOffset, group_size, current_max_len);
    this->m_iNodeOffset += node_created;

    this->m_iCacheOffset += node_created;
    return {
        this->m_iCellIdxOffset - prev_cell_offset,
        beam_size,
        group_size,
        cache_id_offset,
        node_created};
}

void TablesManager::step_over(const at::Tensor & log_p_batch, const at::Tensor & candidates_log_p)
{
    // update log_p of tree nodes
    int current_group_size = this->m_iWindowSize * this->m_iBeamSize * this->m_iBeamSize;
    int current_max_update_len = this->m_iMaxSeqlen;
    if (this->m_iCurrentStep > this->m_iWindowSize)
    {
        current_max_update_len = this->m_iWindowSize + 1;
    }
    if (!this->m_bHasTrajectories) {
        float *log_p_offset = this->m_pLogProbCache + this->m_iLogProbOffset;
        int log_p_bias = cuda_assign_log_p(log_p_batch, candidates_log_p, log_p_offset, this->m_pActiveCells, this->m_pUpdateRange,
                                        this->m_pCellOffsets, this->m_iBeamSize, current_group_size, current_max_update_len,
                                        this->m_tSeqlens.size(0));
        this->m_iLogProbOffset += log_p_bias;
    }
    if (this->m_iCurrentStep >= this->m_iWindowSize)
    {
        uint update_range = this->m_iCurrentStep > this->m_iWindowSize ? this->m_iWindowSize + 1 : this->m_iMaxSeqlen;
        if (!this->m_bHasTrajectories) {
            cuda_update_log_p(this->m_pActiveCells, this->m_pUpdateRange, update_range, this->m_iBeamSize,
                            this->m_iWindowSize, this->m_tSeqlens.size(0));
        }
    }
    else
    {
        cuda_update_range(this->m_pUpdateRange, this->m_tSeqlens.size(0));
    }
    this->m_iCurrentStep++;
}

void TablesManager::beam_select(const at::Tensor & indices)
{
    cuda_beam_select(indices, this->m_pActiveCells, this->m_pUpdateRange, this->m_pCellOffsets,
                     this->m_iBeamSize, this->m_tSeqlens.size(0));
}

vector<vector<ExportCell>> TablesManager::dump_cells()
{
    vector<vector<ExportCell>> result(this->m_tSeqlens.size(0));
    cuda_dump_cells(result, this->m_tSeqlens, this->m_pCells, this->m_pUpdateRange, this->m_iBeamSize);
    return result;
}

uint TablesManager::current_step()
{
    return this->m_iCurrentStep;
}

bool TablesManager::finished()
{
    return this->m_iCurrentStep >= this->m_iMaxSeqlen;
}

void TablesManager::prepare_bilm(const at::Tensor & cache_ids, uint bos, uint eos)
{
    cuda_gather_bilm(cache_ids, this->m_pCells, this->m_pLeftmost, this->m_pRightmost,
                     this->m_iEmptyCacheId, this->m_iBeamSize, this->m_tSeqlens, bos, eos, this->m_iMaxSeqlen);
}

void TablesManager::set_merge_trajectories(const at::Tensor & indices)
{
    this->m_bHasTrajectories = true;
    assert(this->m_iCurrentStep == 0);
    this->m_tMergePos = indices;
}

int TablesManager::total_len() {
    return this->m_iTotalLen;
}

size_t TablesManager::batch_size() {
    return this->m_tSeqlens.size(0);
}

void TablesManager::recover_sampled_trees(const at::Tensor & span_masks, 
                                          const at::Tensor & targets, 
                                          const at::Tensor & split_points)
{
    // span_masks: (batch_size, K, L - 1, L - 1)
    // targets: (batch_size, K, L - 1)
    // split_points: (total_cells, window_size)
    cuda_recover_sample_trees(span_masks, targets, split_points, this->m_pCells, this->m_tSeqlens);
}