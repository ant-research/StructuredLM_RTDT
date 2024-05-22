// Copyright (c) 2024 Ant Group
// Author: Xiang Hu
// #undef NDEBUG
#include <stdexcept>
#include <cstring>
#include <numeric>
#include "py_backend.h"
#include <cassert>
#include <cmath>
#include <functional>
using namespace torch::indexing;

#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

LinkedNode::LinkedNode(Cell * value):m_pLeft{NULL}, m_pRight{NULL}, m_pLeftup{NULL}, 
m_pRightup{NULL}, m_pLeftdown{NULL}, m_pRightdown{NULL}, m_pCell{value} {
}

LinkedNode::~LinkedNode(){
    this->m_pCell->setNode(NULL);
}

Cell * LinkedNode::getCell() const {
    return this->m_pCell;
}

LinkedNode * LinkedNode::left() const {
    return this->m_pLeft;
}

LinkedNode * LinkedNode::right() const {
    return this->m_pRight;
}

LinkedNode * LinkedNode::leftup() const {
    return this->m_pLeftup;
}

LinkedNode * LinkedNode::rightup() const {
    return this->m_pRightup;
}

LinkedNode * LinkedNode::leftdown() const {
    return this->m_pLeftdown;
}

LinkedNode * LinkedNode::rightdown() const {
    return this->m_pRightdown;
}

void LinkedNode::setLeft(LinkedNode * other) {
    if (this->m_pLeft != other) {
        this->m_pLeft = other;
        if (other != NULL) {
            other->setRight(this);
        }
    }
}

void LinkedNode::setRight(LinkedNode * other) {
    if (this->m_pRight != other) {
        this->m_pRight = other;
        if (other != NULL) {
            other->setLeft(this);
        }
    }
}

void LinkedNode::setLeftup(LinkedNode * other) {
    if (this->m_pLeftup != other) {
        this->m_pLeftup = other;
        if (other != NULL) {
            assert(this->getCell()->j == other->getCell()->j);
            other->setRightdown(this);
        }
    }
}

void LinkedNode::setRightup(LinkedNode * other) {
    if (this->m_pRightup != other) {
        this->m_pRightup = other;
        if (other != NULL) {
            assert(this->getCell()->i == other->getCell()->i);
            other->setLeftdown(this);
        }
    }
}

void LinkedNode::setLeftdown(LinkedNode * other) {
    if (this->m_pLeftdown != other) {
        this->m_pLeftdown = other;
        if (other != NULL) {
            assert(this->getCell()->i == other->getCell()->i);
            other->setRightup(this);
        }
    }
}

void LinkedNode::setRightdown(LinkedNode * other) {
    if (this->m_pRightdown != other) {
        this->m_pRightdown = other;        
        if (other != NULL) {
            assert(this->getCell()->j == other->getCell()->j);
            other->setLeftup(this);
        }
    }
}

Cell::Cell(int i, int j, int window_size, TableManager * mgr, int batch_id, bool is_root):i{i}, j{j}, detached{false}, 
best_split{-1}, a_ij_split{-1}, cache_id{-1}, batch_id(batch_id), m_pReadyChild(0), m_pMgr(mgr), m_bIsRoot(is_root), ext_vocab_id(0) {
    this->m_pNode = NULL;
    if (j > i) {
        this->split_size = j - i < window_size ? j - i : window_size;
        this->splits = new int[this->split_size];
        memset(this->splits, 0, this->split_size * sizeof(int));
    } else {
        this->split_size = 0;
        this->splits = 0;
    }
}

Cell::~Cell() {
    if (this->splits != NULL) {
        delete this->splits;
    }
    if (this->m_pNode != NULL) {
        delete this->m_pNode;
    }
}

int Cell::getDetachedCacheID(int detach_offset) const {
    if (this->detached || this->i == this->j) {
        return detach_offset + this->cache_id;
    } else {
        return this->cache_id;
    }
}

LinkedNode * Cell::getNode() const {
    return this->m_pNode;
}

void Cell::setNode(LinkedNode * target) {
    assert(this->m_pNode == NULL || target == NULL);
    assert(this->m_pNode != NULL || target != NULL);
    this->m_pNode = target;
}

void Cell::addParent(Cell * parent) {
    this->m_lParents.push_back(parent);
}

void Cell::onReady() {
    for (auto const & cell : this->m_lParents) {
        cell->notifyChildReady();
    }
}

void Cell::notifyChildReady() {
    ++this->m_pReadyChild;
    assert(this->m_pReadyChild <= 2 * this->split_size);
    if (this->m_pReadyChild == 2 * this->split_size && (m_bIsRoot || this->m_lParents.size() > 0)) {
        // not root and has parents
        this->m_pMgr->on_cell_ready(this);
    }
}


int Cell::getBestSplit() const {
    int k = this->best_split;
    assert(k >= 0);
    assert(k < this->split_size);
    return this->splits[k];
}

int Cell::getGumbelSplit() const {
    int k = this->a_ij_split;
    assert(k >= 0);
    assert(k < this->split_size);
    return this->splits[k];
}

CellTable::CellTable(int seq_len, int window_size, int batch_i, TableManager * mgr):m_iCellOffset{0}, m_iMaxCreatedCells{2 * (window_size + 1) * seq_len}, 
m_iSeqLen{seq_len}, m_iWindowSize{window_size}, m_iBatchId(batch_i), m_pMgr(mgr) {
    this->m_pCells = new Cell*[seq_len * seq_len];
    memset(this->m_pCells, 0, seq_len * seq_len * sizeof(Cell*));

    this->m_pCreatedCells = new Cell*[this->m_iMaxCreatedCells];
    memset(this->m_pCreatedCells, 0, this->m_iMaxCreatedCells * sizeof(Cell*));
}

CellTable::~CellTable() {
    for (int i = 0; i < this->m_iCellOffset; ++i) {
        assert(this->m_pCreatedCells[i] != NULL);
        delete this->m_pCreatedCells[i];
    }
    delete this->m_pCreatedCells;
    delete this->m_pCells;
}

Cell * CellTable::get(const int i, const int j) {
    assert(i <= j);
    assert(j < this->m_iSeqLen);
    if (this->m_pCells[i * this->m_iSeqLen + j] == NULL) {
        bool is_root = j - i + 1 == this->m_iSeqLen && i == 0;
        Cell * new_cell = new Cell(i, j, this->m_iWindowSize, this->m_pMgr, this->m_iBatchId, is_root);
        this->m_pCreatedCells[this->m_iCellOffset++] = new_cell;
        this->m_pCells[i * this->m_iSeqLen + j] = new_cell;
    }
    return this->m_pCells[i * this->m_iSeqLen + j];
}

bool CellTable::isEmpty(const int i, const int j) {
    return this->m_pCells[i * this->m_iSeqLen + j] == NULL;
}

int CellTable::getLen() const {
    return this->m_iSeqLen;
}

void init_active_cells(int window_size, int seq_len, CellTable * cell_table) {
    for (int layer_i = 0; layer_i <= window_size; ++layer_i) {
        LinkedNode * left_previous = NULL;
        for (int pos_i = 0; pos_i < seq_len - layer_i; ++pos_i) {
            Cell * cell_ij = cell_table->get(pos_i, pos_i + layer_i);
            if (layer_i > 0) {
                for (int split_idx = 0; split_idx < layer_i; ++split_idx) {
                    cell_ij->splits[split_idx] = pos_i + split_idx;
                }
            }
            LinkedNode * node = new LinkedNode(cell_ij);
            cell_ij->setNode(node);
            node->setLeft(left_previous);
            left_previous = node;
            LinkedNode * leftdown = NULL;
            LinkedNode * rightdown = NULL;
            if (layer_i > 0) {
                leftdown = cell_table->get(pos_i, pos_i + layer_i - 1)->getNode();
                rightdown = cell_table->get(pos_i + 1, pos_i + layer_i)->getNode();
            }
            node->setLeftdown(leftdown);
            node->setRightdown(rightdown);
        }
    }
}

LinkedNode * create_new_node(LinkedNode * leftdown, LinkedNode * rightdown, LinkedNode * left, 
LinkedNode * right, LinkedNode * ld_most, LinkedNode * rd_most, CellTable * table) {
    int i = leftdown->getCell()->i;
    int j = rightdown->getCell()->j;
    Cell * new_cell = table->get(i, j);
    LinkedNode * new_node = new LinkedNode(new_cell);
    LinkedNode * current = ld_most;

    int idx = 0;
    while (current != rd_most->right()) {
        assert(idx < new_cell->split_size);
        new_cell->splits[idx++] = current->getCell()->j;
        current = current->right();
    }

    new_cell->setNode(new_node);
    new_node->setLeftdown(leftdown);
    new_node->setRightdown(rightdown);
    new_node->setLeft(left);
    new_node->setRight(right);
    return new_node;
}

void prune(LinkedNode * node, std::function<void(Cell*)> on_new_cell, CellTable * table) {
    node->getCell()->detached = true;
    LinkedNode * leftdown = node->leftdown();
    LinkedNode * rightdown = node->rightdown();

    LinkedNode * left_node = leftdown;
    LinkedNode * tmp = NULL;

    int left_steps = 0;
    while (left_node != NULL) {
        if (left_node->left() != NULL) {
            left_node->left()->setRight(left_node->rightup());
        }
        if (left_node->rightup() != NULL) {
            left_node->rightup()->setLeftdown(left_node->leftdown());
        }
        left_steps += 1;

        if (left_node->leftup() != NULL) {
            tmp = left_node->leftup();
            delete left_node;
            left_node = tmp;
        } else {
            if (left_node->rightup() != NULL) {
                tmp = left_node->rightup();
                left_steps += 1;
            } else {
                tmp = left_node->right();
            }
            // std::cout << "delete: " << left_node->getCell()->i << "," << left_node->getCell()->j << std::endl;
            delete left_node;
            left_node = tmp;
            break;
        }
    }

    // std::cout << "left_node: " << left_node->getCell()->i << "," << left_node->getCell()->j << std::endl;

    while (left_node->rightup() != NULL) {
        left_node = left_node->rightup();
        // std::cout << "left_node: " << left_node->getCell()->i << "," << left_node->getCell()->j << std::endl;
        left_steps += 1;
    }
    // std::cout << "pb" << std::endl;

    LinkedNode * right_node = rightdown;
    assert(rightdown != NULL);
    int right_steps = 0;
    while (right_node != NULL) {
        // std::cout << "right_node: " << right_node->getCell()->i << "," << right_node->getCell()->j << std::endl;
        if (right_node->right() != NULL) {
            right_node->right()->setLeft(right_node->leftup());
        }
        if (right_node->leftup() != NULL) {
            right_node->leftup()->setRightdown(right_node->rightdown());
        }
        right_steps += 1;

        if (right_node->rightup() != NULL) {
            tmp = right_node->rightup();
            delete right_node;
            right_node = tmp;
        } else {
            if (right_node->leftup() != NULL) {
                tmp = right_node->leftup();
                right_steps += 1;
            } else {
                tmp = right_node->left();
            }
            // std::cout << "delete : " << right_node->getCell()->i << "," << right_node->getCell()->j << std::endl;
            delete right_node;
            right_node = tmp;
            break;
        }
    }
    // std::cout << "right_node: " << right_node->getCell()->i << "," << right_node->getCell()->j << std::endl;
    while (right_node->leftup() != NULL) {
        // std::cout << "right_node: " << right_node->getCell()->i << "," << right_node->getCell()->j << std::endl;
        right_node = right_node->leftup();
        right_steps += 1;
    }

    // std::cout << "pc" << std::endl;

    LinkedNode * current = left_node->left() != NULL ? left_node->left() : left_node;
    LinkedNode * end = right_node->right() != NULL ? right_node->right() : right_node;

    LinkedNode * current_ld_most = current;
    while (current_ld_most->leftdown() != NULL) {
        current_ld_most = current_ld_most->leftdown();
    }

    LinkedNode * current_rd_most = current;
    while (current_rd_most->rightdown() != NULL) {
        current_rd_most = current_rd_most->rightdown();
    }

    // std::cout << "pd" << std::endl;
    while (current != end) {
        LinkedNode * node_left = current->leftup();
        LinkedNode * node_right = current->right()->rightup();
        // std::cout << "pcreate" << std::endl;
        LinkedNode * node = create_new_node(current, current->right(), node_left, node_right,
                                            current_ld_most, current_rd_most, table);
        // std::cout << "pcreate over" << std::endl;
        on_new_cell(node->getCell());
        current = current->right();
        current_ld_most = current_ld_most->right();
        current_rd_most = current_rd_most->right();

    }
}


TableManager::TableManager(const py::array_t<int>& seq_lens, const py::array_t<int>& group_ids, const py::array_t<int>& merge_orders, const int window_size, 
const int cache_id_offset, const int detach_id_offset, vector<py::array_t<int>>& span_ids): m_iBatchSize{seq_lens.shape()[0]}, m_iWindowSize{window_size}, 
m_iCacheOffset{cache_id_offset}, m_iCurrentStep{1}, m_iDetachCacheOffset(detach_id_offset), m_iCellNum{0} {
    this->m_pCellTables = new CellTable*[this->m_iBatchSize];
    // this->m_pMergeOrders = new Span*[this->m_iBatchSize];
    auto pMergeOrders = new Span*[this->m_iBatchSize];

    auto buf = seq_lens.request();
    int * seq_lens_ptr = (int*)buf.ptr;
    int max_seq_len = 0;
    int seq_len_sum = 0;
    for (int batch_i = 0; batch_i < this->m_iBatchSize; ++batch_i) {
        int seq_len = seq_lens_ptr[batch_i];
        CellTable * table = new CellTable(seq_len, window_size, batch_i, this);
        this->m_pCellTables[batch_i] = table;
        init_active_cells(window_size, seq_len, table);
        for (int pos = 0; pos < seq_len; ++pos) {
            table->get(pos, pos)->cache_id = this->m_iCacheOffset + this->m_iCellNum++;
        }

        if (span_ids.size() > 0) {
            buf = span_ids[batch_i].request();
            int total_size = span_ids[batch_i].shape(0) / 3;
            // std::cout << "batch: " << batch_i << " total size: " << total_size << std::endl;
            // std::cout << "table seq len: " << table->getLen() << std::endl;
            int * span_ids_ptr = (int*)buf.ptr;
            for (int span_idx = 0; span_idx < total_size; ++span_idx) {
                // std::cout << span_ids_ptr[span_idx * 3] << ", " << span_ids_ptr[span_idx * 3 + 1] << "->" << span_ids_ptr[span_idx * 3 + 2] << ";";
                if (!table->isEmpty(span_ids_ptr[span_idx * 3], span_ids_ptr[span_idx * 3 + 1])) {
                    table->get(span_ids_ptr[span_idx * 3], span_ids_ptr[span_idx * 3 + 1])->ext_vocab_id = span_ids_ptr[span_idx * 3 + 2];
                }
            }
        }

        max_seq_len = seq_len > max_seq_len ? seq_len : max_seq_len;
        seq_len_sum += seq_len;
    }

    buf = group_ids.request();
    int * merge_order_ptr = 0;
    int merge_order_L = 0;
    int * group_ids_ptr = (int*)buf.ptr;
    int max_group_size = group_ids_ptr[m_iBatchSize - 1] + 1;

    if (m_iBatchSize != max_group_size) {
        merge_order_L = max_seq_len - 1;
        merge_order_ptr = new int[m_iBatchSize * merge_order_L];
        std::fill_n(merge_order_ptr, m_iBatchSize * merge_order_L, -1);
        buf = merge_orders.request();
        int * chunk_merge_order_ptr = (int*)buf.ptr;
        int chunk_size = merge_orders.shape(1); // sum(seq_lens) - 1

        int prev_group_id = -1;
        int offset = 0;
        int * pos_chunk2sent = new int[max_group_size * chunk_size];
        int * bid_chunk2sent = new int[max_group_size * chunk_size];
        std::fill_n(bid_chunk2sent, max_group_size * chunk_size, -1);
        std::fill_n(pos_chunk2sent, max_group_size * chunk_size, -1);
        for (int batch_i = 0; batch_i < this->m_iBatchSize; ++batch_i) {
            if (group_ids_ptr[batch_i] != prev_group_id) {
                prev_group_id = group_ids_ptr[batch_i];
                offset = 0;
            }

            for (int sent_idx = 0; sent_idx < seq_lens_ptr[batch_i] - 1; ++sent_idx) {
                if (offset + sent_idx < chunk_size) {
                    pos_chunk2sent[prev_group_id * chunk_size + offset + sent_idx] = sent_idx;
                    bid_chunk2sent[prev_group_id * chunk_size + offset + sent_idx] = batch_i;   
                }
            }
            offset += seq_lens_ptr[batch_i];
        }

        int * sent_offset = new int[m_iBatchSize];
        std::fill_n(sent_offset, m_iBatchSize, 0);
        for (int group_i = 0; group_i < max_group_size; ++group_i) {
            for (int pos = 0; pos < chunk_size; ++pos) {
                assert(chunk_merge_order_ptr[group_i * chunk_size + pos] >= 0 && chunk_merge_order_ptr[group_i * chunk_size + pos] < chunk_size);
                int sent_id = bid_chunk2sent[group_i * chunk_size + chunk_merge_order_ptr[group_i * chunk_size + pos]];
                int split_pos = pos_chunk2sent[group_i * chunk_size + chunk_merge_order_ptr[group_i * chunk_size + pos]];
                // std::cout << "sent_id" << sent_id << ", " << split_pos << std::endl;
                if (sent_id != -1 && split_pos < seq_lens_ptr[sent_id] - 1) {
                    // std::cout << sent_id << "|" << sent_offset[sent_id] << "|" << split_pos << ", " << std::endl;
                    assert(sent_offset[sent_id] < merge_order_L);
                    assert(sent_id >= 0 && sent_id < m_iBatchSize);
                    merge_order_ptr[sent_id * merge_order_L + sent_offset[sent_id]++] = split_pos;
                }
            }
        }
        delete pos_chunk2sent;
        delete bid_chunk2sent;
        delete sent_offset;
    } else {
        buf = merge_orders.request();
        merge_order_ptr = (int*)buf.ptr;
        merge_order_L = merge_orders.shape()[1];
    }
    // std::cout << "B" << std::endl;

    //convert merge order to cell i,j
    int left_i = 0;
    int right_j = 0;
    int merge_pos = 0;
    // assert(seq_lens.shape()[0] == merge_orders.shape()[0]);
    
    
    int * current_merge_orders = 0;
    for (int batch_i = 0; batch_i < this->m_iBatchSize; ++batch_i) {
        int seq_len = seq_lens_ptr[batch_i];
        Span * left_splits = new Span[seq_len - 1];
        Span * right_splits = new Span[seq_len - 1];
        current_merge_orders = merge_order_ptr + batch_i * merge_order_L;
        for (int split = 0; split < seq_len - 1; ++split) {
            left_splits[split].i = split;
            left_splits[split].j = split;
            right_splits[split].i = split + 1;
            right_splits[split].j = split + 1;
        }

        Span * merge_orders = new Span[seq_len - 1];
        for (int action_i = 0; action_i < seq_len - 1; ++action_i) {
            merge_pos = current_merge_orders[action_i];
            assert(merge_pos < seq_len - 1);
            left_i = left_splits[merge_pos].i;
            right_j = right_splits[merge_pos].j;

            merge_orders[action_i].i = left_i;
            merge_orders[action_i].j = right_j;
            if (left_i >= 1) {
                right_splits[left_i - 1].i = left_i;
                right_splits[left_i - 1].j = right_j;
            }
            if (right_j < seq_len - 1) {
                left_splits[right_j].i = left_i;
                left_splits[right_j].j = right_j; 
            }
            // std::cout << "merge span: " << left_i << ", " << right_j << std::endl; 
        }
        pMergeOrders[batch_i] = merge_orders;

        delete left_splits;
        delete right_splits;
    }
    // std::cout << "C" << std::endl;

    // int max_seq_len = *max_element(seq_lens.begin(), seq_lens.end());
    // int seq_len_sum = accumulate(seq_lens.begin(), seq_lens.end(), 0);
    this->m_iMaxSeqLen = max_seq_len;
    this->m_pCellOrders = new Cell**[max_seq_len - 1];
    this->m_pCellNums = new int[max_seq_len - 1];
    this->m_pTargetCacheIds = new long*[max_seq_len - 1];
    this->m_pGroupCacheIds = new long*[max_seq_len - 1];
    this->m_pTargetExtIds = new long*[max_seq_len - 1];
    this->m_pDetachGroupCacheIds = new long*[max_seq_len - 1];

    this->m_pLDRCache_ids = 0;
    this->m_pExtIds = 0;
    this->m_pPositionIds = 0;
    this->m_pTgtIds = 0;

    this->m_pSpanMasks = 0;
    this->m_pSplitTargets = 0;
    this->m_pSpanGatherIds = 0;
    this->m_pTokenPositions = 0;

    for (int step = 1; step < max_seq_len; ++step) {
        this->m_pCellOrders[step - 1] = 0;
        this->m_pCellNums[step - 1] = 0;
        this->m_pTargetCacheIds[step - 1] = 0;
        this->m_pGroupCacheIds[step - 1] = 0;
        this->m_pTargetExtIds[step - 1] = 0;
        this->m_pDetachGroupCacheIds[step - 1] = 0;
    }

    this->build_cell_dependencies(pMergeOrders);
    for (int batch_i = 0; batch_i < this->m_iBatchSize; ++batch_i) {
        int seq_len = seq_lens_ptr[batch_i];
        CellTable * table = this->m_pCellTables[batch_i];
        for (int pos = 0; pos < seq_len; ++pos) {
            table->get(pos, pos)->onReady();
        }
    }

    if (m_iBatchSize != max_group_size) {
        delete merge_order_ptr;
    }
    for (int batch_i = 0; batch_i < this->m_iBatchSize; ++batch_i) {
        delete pMergeOrders[batch_i];
    }
    delete pMergeOrders;
}

TableManager::~TableManager() {
    for (int batch_i = 0; batch_i < this->m_iBatchSize; ++batch_i) {
        delete this->m_pCellTables[batch_i];
        // delete this->m_pMergeOrders[batch_i];
    }
    for (int step = 0; step < this->m_iMaxSeqLen - 1; ++step) {
        if (this->m_pCellOrders[step] != 0)
            delete this->m_pCellOrders[step];
        if (this->m_pTargetCacheIds[step] != 0)
            delete this->m_pTargetCacheIds[step];
        if (this->m_pGroupCacheIds[step] != 0)
            delete this->m_pGroupCacheIds[step];
        if (this->m_pTargetExtIds[step] != 0)
            delete this->m_pTargetExtIds[step];
        if (this->m_pDetachGroupCacheIds[step] != 0)
            delete this->m_pDetachGroupCacheIds[step];
    }


    if (m_pLDRCache_ids != 0)
        delete m_pLDRCache_ids;
    if (m_pExtIds != 0)
        delete m_pExtIds;
    if (m_pPositionIds != 0)
        delete m_pPositionIds;
    if (m_pTgtIds != 0)
        delete m_pTgtIds;
    if (m_pSpanMasks != 0)
        delete m_pSpanMasks;
    if (m_pSplitTargets != 0)
        delete m_pSplitTargets;
    if (m_pSpanGatherIds != 0)
        delete m_pSpanGatherIds;
    if (m_pTokenPositions != 0)
        delete m_pTokenPositions;

    delete this->m_pCellNums;
    delete this->m_pCellOrders;
    delete this->m_pCellTables;
    // delete this->m_pMergeOrders;
    delete this->m_pTargetCacheIds;
    delete this->m_pTargetExtIds;
    delete this->m_pGroupCacheIds;
    delete this->m_pDetachGroupCacheIds;
}

void TableManager::push_cell(Cell * cell) {
    int current_cache_id = this->m_iCacheOffset + this->m_iCellNum++;
    cell->cache_id = current_cache_id;
}

void TableManager::build_cell_dependencies(Span ** pMergeOrders) {
    for (int step = 1; step < this->m_iMaxSeqLen; ++step) {
        // #pragma omp parallel for
        for (int batch_i = 0; batch_i < this->m_iBatchSize; ++batch_i) {
            CellTable * table = this->m_pCellTables[batch_i];
            int seq_len = table->getLen();
            // printf("batch_i: %d, seq_len: %d\n", batch_i, seq_len);
            if (step < seq_len) {
                if (step <= this->m_iWindowSize) {
                    for (int i = 0; i < seq_len - step; ++i) {
                        int j = i + step;
                        assert(j < seq_len);
                        Cell * current_cell = table->get(i, j);
                        for (int split_idx = 0; split_idx < current_cell->split_size; ++split_idx) {
                            int k = current_cell->splits[split_idx];
                            assert(i <= k);
                            assert(k < j);
                            Cell * cell_ik = table->get(i, k);
                            Cell * cell_kj = table->get(k + 1, j);
                            cell_ik->addParent(current_cell);
                            cell_kj->addParent(current_cell);
                        }
                    }
                } else {
                    Span & merge_span = pMergeOrders[batch_i][step - this->m_iWindowSize - 1];
                    int i = merge_span.i;
                    int j = merge_span.j;
                    // printf("(%d, %d)\n", i, j);
                    prune(table->get(i, j)->getNode(), [&](Cell * cell) {
                        for (int sp_idx = 0; sp_idx < cell->split_size; ++sp_idx) {
                            int k = cell->splits[sp_idx];
                            assert(cell->i <= k);
                            assert(k < cell->j);
                            Cell * cell_ik = table->get(cell->i, k);
                            Cell * cell_kj = table->get(k + 1, cell->j);
                            cell_ik->addParent(cell);
                            cell_kj->addParent(cell);
                        }
                        
                    }, table);
                }
            }
        }
    }
}

void TableManager::on_cell_ready(Cell * ready_cell) {
    this->m_lReadyCells.push_back(ready_cell);
}

bool TableManager::is_finished() {
    return this->m_lReadyCells.size() == 0;
}

vector<at::Tensor> TableManager::step() {
    int current_step = this->m_iCurrentStep;

    int total_size = this->m_lReadyCells.size();
    int group_size = this->m_iCurrentStep <= this->m_iWindowSize ? this->m_iCurrentStep : this->m_iWindowSize;
    this->m_pCellNums[current_step - 1] = total_size;
    this->m_pCellOrders[current_step - 1] = new Cell*[total_size];

    // at::Tensor target_cache_ids_ = torch::zeros({total_size}, at::kLong);
    // at::Tensor group_cache_ids_ = torch::zeros({total_size, group_size, 2}, at::kLong);
    // at::Tensor detach_group_cache_ids_ = torch::zeros({total_size, group_size, 2}, at::kLong);

    this->m_pTargetCacheIds[current_step - 1] = new long[total_size];
    this->m_pGroupCacheIds[current_step - 1] = new long[total_size * group_size * 2];
    this->m_pTargetExtIds[current_step - 1] = new long[total_size];
    this->m_pDetachGroupCacheIds[current_step - 1] = new long[total_size * group_size * 2];

    auto tgt_cache_ids_ptr = this->m_pTargetCacheIds[current_step - 1];
    auto tgt_ext_id_ptr = this->m_pTargetExtIds[current_step - 1];
    auto group_ids_ptr = this->m_pGroupCacheIds[current_step - 1];
    auto detach_group_ids_ptr = this->m_pDetachGroupCacheIds[current_step - 1];

    int idx_offset = 0;
    for (int cell_idx = 0; cell_idx < total_size; ++cell_idx) {
        Cell * cell_ptr = this->m_lReadyCells.front();
        this->m_lReadyCells.pop_front();
        CellTable * table = this->m_pCellTables[cell_ptr->batch_id];
        int i = cell_ptr->i;
        int j = cell_ptr->j;
        cell_ptr->onReady();
        this->push_cell(cell_ptr);
        this->m_pCellOrders[this->m_iCurrentStep - 1][cell_idx] = cell_ptr;
        tgt_cache_ids_ptr[cell_idx] = cell_ptr->cache_id;
        tgt_ext_id_ptr[cell_idx] = cell_ptr->ext_vocab_id;
        for (int split_idx = 0; split_idx < cell_ptr->split_size; ++split_idx) {
            int k = cell_ptr->splits[split_idx];
            assert(i <= k);
            assert(k < j);
            Cell * cell_ik = table->get(i, k);
            Cell * cell_kj = table->get(k + 1, j);
            // group_cache_ids_.index({cell_idx, split_idx, 0}) = cell_ik->cache_id;
            // group_cache_ids_.index({cell_idx, split_idx, 1}) = cell_kj->cache_id;
            // detach_group_cache_ids_.index({cell_idx, split_idx, 0}) = cell_ik->getDetachedCacheID(this->m_iDetachCacheOffset);
            // detach_group_cache_ids_.index({cell_idx, split_idx, 1}) = cell_kj->getDetachedCacheID(this->m_iDetachCacheOffset);
            group_ids_ptr[idx_offset] = cell_ik->cache_id;
            group_ids_ptr[idx_offset + 1] = cell_kj->cache_id;
            detach_group_ids_ptr[idx_offset] = cell_ik->getDetachedCacheID(this->m_iDetachCacheOffset);
            detach_group_ids_ptr[idx_offset + 1] = cell_kj->getDetachedCacheID(this->m_iDetachCacheOffset);

            idx_offset += 2;
        }
    }

    at::Tensor target_cache_ids_ = torch::from_blob(tgt_cache_ids_ptr, {total_size}, at::kLong);
    at::Tensor group_cache_ids_ = torch::from_blob(group_ids_ptr, {total_size, group_size, 2}, at::kLong);
    at::Tensor ext_id_cache_ids_ = torch::from_blob(tgt_ext_id_ptr, {total_size}, at::kLong);
    at::Tensor detach_group_cache_ids_ = torch::from_blob(detach_group_ids_ptr, {total_size, group_size, 2}, at::kLong);

    this->m_iCurrentStep += 1;
    return {target_cache_ids_, ext_id_cache_ids_, group_cache_ids_, detach_group_cache_ids_};
    // return {target_cache_ids_, group_cache_ids_, detach_group_cache_ids_};
}

bool hit_span(int i, int k, int j, py::array_t<int>& atom_spans) {
    auto atom_span_pt = atom_spans.unchecked<2>();
    int atom_span_st = 0, atom_span_ed = 0;
    for (int atom_i = 0; atom_i < atom_spans.shape(0); ++atom_i) {
        atom_span_st = atom_span_pt(atom_i, 0);
        atom_span_ed = atom_span_pt(atom_i, 1);

        if (j < atom_span_st || i > atom_span_ed || (i >= atom_span_st && j <= atom_span_ed)) {
            // no overlap
            continue;
        }
        if ((k < atom_span_st && j >= atom_span_ed) || (k + 1 > atom_span_ed && i <= atom_span_st)) {
            continue;
        }
        return true;
    }
    return false;
}

void induce_best_splits(int max_seq_len, int * cell_nums, Cell *** cell_orders, vector<py::array_t<int>> &best_splits, 
    vector<py::array_t<int>> &a_ij_splits, vector<py::array_t<int>> &atom_spans) {
    // a_ij after gumbel_softmax maybe different from best_splits
    for (int step = 0; step < max_seq_len - 1; ++step) {
        for (int cell_idx = 0; cell_idx < cell_nums[step]; ++cell_idx) {
            Cell * current_cell = cell_orders[step][cell_idx];
            assert(current_cell->i <= current_cell->j);
            // atom_spans[current_cell->batch_id]
            auto best_splits_mat = best_splits[step].unchecked<2>();
            auto a_ij_np = a_ij_splits[step].unchecked<2>();
            current_cell->a_ij_split = a_ij_np(cell_idx, 0);
            if (atom_spans.size() > 0 && atom_spans[current_cell->batch_id].shape(0) > 0) {
                for (int split_idx = 0; split_idx < best_splits[step].shape(1); ++split_idx) {
                    int split = best_splits_mat(cell_idx, split_idx);
                    int k = current_cell->splits[split];
                    if (!hit_span(current_cell->i, k, current_cell->j, atom_spans[current_cell->batch_id])) {
                        current_cell->best_split = split;
                        break;
                    }
                }
                // assert(current_cell->best_split != -1);
            } else {
                current_cell->best_split = best_splits_mat(cell_idx, 0);
            }
        }
    }
}

at::Tensor TableManager::root_ids() {
    at::Tensor t = torch::zeros(this->m_iBatchSize, torch::kLong);
    for (int batch_i = 0; batch_i < this->m_iBatchSize; ++batch_i) {
        CellTable * tbl = this->m_pCellTables[batch_i];
        t[batch_i] = tbl->get(0, tbl->getLen() - 1)->cache_id;
    }
    return t;
}

const int TableManager::batch_size() const {
    return this->m_iBatchSize;
}

int left_most(CellTable * table, int idx, int bos_id, int eos_id) {
    if (idx < 0) {
        return bos_id;
    }
    assert (idx < table->getLen() - 1);
    for (int start = 0; start <= idx; ++start) {
        if (!table->isEmpty(start, idx)) {
            return table->get(start, idx)->cache_id;
        }
    }
    assert (false);
}

int right_most(CellTable * table, int idx, int bos_id, int eos_id) {
    if (idx >= table->getLen()) {
        return eos_id;
    }
    assert (idx > 0);
    for (int end = table->getLen() - 1; end >= idx; --end) {
        if (!table->isEmpty(idx, end)) {
            return table->get(idx, end)->cache_id;
        }
    }
    assert (false);
}

at::Tensor TableManager::prepare_bilm(int total_len, int bos_id, int eos_id) {
    at::Tensor cache_ids = torch::zeros({total_len, 2}, torch::kLong);
    int offset = 0;
    for (int batch_i = 0; batch_i < this->m_iBatchSize; ++batch_i) {
        CellTable * tbl = this->m_pCellTables[batch_i];
        for (int idx = 0; idx < tbl->getLen(); ++idx) {
            cache_ids.index({offset, 0}) = left_most(tbl, idx - 1, bos_id, eos_id);
            cache_ids.index({offset, 1}) = right_most(tbl, idx + 1, bos_id, eos_id);
            ++offset;
        }
    }
    assert(offset == total_len);
    return cache_ids;
}

vector<at::Tensor> TableManager::prepare_generation(vector<py::array_t<int>>& score_splits, 
                                                    vector<py::array_t<int>>& a_ij_splits, 
                                                    vector<py::array_t<int>>& atom_spans, 
                                                    const py::array_t<int>& input_ids,
                                                    const py::array_t<int>& group_ids,
                                                    const py::array_t<int>& eos_labels,
                                                    const int reduce_id,
                                                    const int max_input_len) {
    // RETURN: span_mask, split_targets: for the top-down parser to fit
    //         ldr_cache_ids : input embeddings for GPT
    //         position ids: position id for GPT
    //         tgt_ids for generation: cross entropy targets for GPT
    auto buf = group_ids.request();
    size_t buf_sz = buf.size;
    int * group_ids_ptr = (int*)buf.ptr;
    int group_size = group_ids_ptr[buf_sz - 1] + 1;

    int max_seq_len = max_input_len * 2 - 1;
    int max_ids_len = input_ids.shape(1);
    auto input_ids_ptr = input_ids.unchecked<2>();

    // assign best split to each cell
    induce_best_splits(this->m_iMaxSeqLen, this->m_pCellNums, m_pCellOrders, score_splits, a_ij_splits, atom_spans);

    int max_node_size = 2 * this->m_iMaxSeqLen - 1;

    this->m_pLDRCache_ids = new long[group_size * max_seq_len];
    std::fill_n(this->m_pLDRCache_ids, group_size * max_seq_len, 0);
    this->m_pExtIds = new long[group_size * max_seq_len];
    std::fill_n(this->m_pExtIds, group_size * max_seq_len, -1);
    this->m_pPositionIds = new long[group_size * (max_seq_len + 1)];
    std::fill_n(this->m_pPositionIds, group_size * (max_seq_len + 1), 0);
    this->m_pTgtIds = new long[group_size * (max_seq_len + 1)];
    std::fill_n(this->m_pTgtIds, group_size * (max_seq_len + 1), -1);

    int mask_mat_size = (max_input_len - 1) * (max_input_len - 1);

    // this->m_pSpanGatherIds = new long[group_size * max_input_len];
    // std::fill_n(this->m_pSpanGatherIds, group_size * max_input_len, 0);
    this->m_pSpanMasks = new long[group_size * mask_mat_size];
    std::fill_n(this->m_pSpanMasks, group_size * mask_mat_size, 0);
    this->m_pSplitTargets = new long[group_size * (max_input_len - 1)];
    std::fill_n(this->m_pSplitTargets, group_size * (max_input_len - 1), -1);
    this->m_pTokenPositions = new long[group_size * max_input_len];
    std::fill_n(this->m_pTokenPositions, group_size * max_input_len, max_seq_len - 1);


    Cell * queue[max_node_size];
    int prev_group_id = -1;
    int ldr_offset = 0;
    int sent_offset = 1;
    auto eos_labels_arr = eos_labels.unchecked<1>();
    for (int batch_i = 0; batch_i < this->m_iBatchSize; ++batch_i) {
        CellTable * tbl = this->m_pCellTables[batch_i];
        Cell * root = tbl->get(0, tbl->getLen() - 1);
        int sent_len = root->j - root->i + 1;

        int index_bias = sent_len * 2 - 1;
        int queue_offset = 0;
        int split_idx = 0;
        int group_id = group_ids_ptr[batch_i];
        if (group_id != prev_group_id) {
            ldr_offset = 0;
            sent_offset = 1;
            prev_group_id = group_id;
        }

        Cell * current = NULL;
        queue[queue_offset++] = root;
        while (queue_offset > 0) {
            current = queue[--queue_offset];

            if (current->split_size > 0) {
                // non-terminal
                int k = current->getBestSplit();

                std::fill_n(m_pSpanMasks + group_id * mask_mat_size + (sent_offset - 1 + split_idx) * (max_input_len - 1) + sent_offset - 1 + current->i, current->j - current->i, 1);
                assert(current->i <= k && current->j > k);
                m_pSplitTargets[group_id * (max_input_len - 1) + sent_offset + split_idx - 1] = sent_offset - 1 + k;
                ++split_idx;
                queue[queue_offset++] = tbl->get(current->i, k);
                queue[queue_offset++] = tbl->get(k + 1, current->j);
            }
        }

        queue_offset = 0;
        queue[queue_offset++] = root;
        current = NULL;
        // If there is appending sentence, eos_id will be overrided.
        // tgt_ids.index({group_id, ldr_offset + index_bias}) = eos_id;
        this->m_pTgtIds[group_id * (max_seq_len + 1) + ldr_offset + index_bias] = eos_labels_arr[group_id];
        // for (int idx = 0; idx < sent_len - 1; ++idx) {
        //     m_pSpanGatherIds[group_id * max_input_len + sent_offset + idx - 1] = batch_i * (max_ids_len - 1) + idx;
        // }
        while (queue_offset > 0) {
            current = queue[--queue_offset];

            // ldr_cache_ids.index({group_id, ldr_offset + index_bias - 1}) = current->cache_id;
            // position_ids.index({group_id, ldr_offset + index_bias}) = current->j + sent_offset;
            m_pLDRCache_ids[group_id * max_seq_len + ldr_offset + index_bias - 1] = current->cache_id;
            m_pPositionIds[group_id * (max_seq_len + 1) + ldr_offset + index_bias] = current->j + sent_offset;
            m_pExtIds[group_id * max_seq_len + ldr_offset + index_bias - 1] = current->ext_vocab_id;
            if (current->split_size > 0) {
                // non-terminal
                int k = current->getBestSplit();

                assert(current->i <= k && current->j > k);
                queue[queue_offset++] = tbl->get(current->i, k);
                queue[queue_offset++] = tbl->get(k + 1, current->j);
                //The right node will be visited first

                // input will be shift right during training, so just keep id same
                // tgt_ids.index({group_id, ldr_offset + index_bias - 1}) = reduce_id;
                m_pTgtIds[group_id * (max_seq_len + 1) + ldr_offset + index_bias - 1] = reduce_id;
            } else {
                // terminal
                m_pTgtIds[group_id * (max_seq_len + 1) + ldr_offset + index_bias - 1] = \
                    input_ids_ptr(batch_i, current->i);
                m_pTokenPositions[group_id * max_input_len + sent_offset - 1 + current->i] = \
                    ldr_offset + index_bias - 1;
            }
            --index_bias;
        }
        ldr_offset += 2 * sent_len - 1;
        sent_offset += sent_len;
    }

    //torch::zeros({group_size, max_seq_len}, at::kLong);
    at::Tensor ldr_cache_ids = torch::from_blob(m_pLDRCache_ids, {group_size, max_seq_len}, at::kLong);
    // at::Tensor position_ids = torch::zeros({group_size, max_seq_len + 1}, at::kLong);
    at::Tensor position_ids = torch::from_blob(m_pPositionIds, {group_size, max_seq_len + 1}, at::kLong);
    // at::Tensor tgt_ids = torch::full({group_size, max_seq_len + 1}, -1, at::kLong);
    at::Tensor tgt_ids = torch::from_blob(m_pTgtIds, {group_size, max_seq_len + 1}, at::kLong);
    
    at::Tensor span_masks = torch::from_blob(m_pSpanMasks, {group_size, max_input_len - 1, max_input_len - 1}, at::kLong);

    at::Tensor split_targets = torch::from_blob(m_pSplitTargets, {group_size, max_input_len - 1}, at::kLong);

    // at::Tensor gather_ids = torch::from_blob(m_pSpanGatherIds, {group_size, max_input_len}, at::kLong);
    at::Tensor ext_vocab_ids_ = torch::from_blob(m_pExtIds, {group_size, max_seq_len}, at::kLong);

    at::Tensor token_indices = torch::from_blob(m_pTokenPositions, {group_size, max_input_len}, at::kLong);

    return {span_masks, split_targets, ldr_cache_ids, position_ids, tgt_ids, token_indices, ext_vocab_ids_};
}

WordTreeNode::WordTreeNode(int entry_id, int total_size, int depth): m_iValue(entry_id), m_iTotalSize(total_size), m_iWordId(-1), m_iDepth(depth){
}

WordTreeNode::~WordTreeNode() {
    for (auto iter = m_mSubNodes.begin(); iter != m_mSubNodes.end(); ++iter) {
        delete iter->second;
    }
}

void WordTreeNode::add_ids(int * ids_ptr, int ids_len, int entry_id, int offset) {
    int current_id = ids_ptr[offset];
    if (m_mSubNodes.find(current_id) == m_mSubNodes.end()) {
        m_mSubNodes[current_id] = new WordTreeNode(-1, m_iTotalSize, m_iDepth + 1);
    }

    if (offset + 1 < ids_len) {
        m_mSubNodes[current_id]->add_ids(ids_ptr, ids_len, entry_id, offset + 1);
    } else if(offset + 1 == ids_len) {
        m_mSubNodes[current_id]->setWordId(entry_id);
    }
}

WordTreeNode * WordTreeNode::next_node(int current_id) {
    // assert(current_id >= 0 && current_id < m_iTotalSize);
    if (m_mSubNodes.find(current_id) != m_mSubNodes.end()) {
        return m_mSubNodes[current_id];
    } else {
        return NULL;
    }
}

void WordTreeNode::setWordId(const int wordId) {
    m_iWordId = wordId;
}

int WordTreeNode::getWordId() const {
    return m_iWordId;
}

int WordTreeNode::getDepth() const {
    return m_iDepth;
}

bool WordTreeNode::isWord() const {
    return m_iWordId != -1;
}

SpanTokenizer::SpanTokenizer(vector<py::array_t<int>>& dictionary, int max_entry_id) {
    m_pRoot = new WordTreeNode(-1, max_entry_id, false);
    int entry_id = 0;
    for (py::array_t<int>& ids : dictionary) {
        auto buf = ids.request();
        int * ids_ptr = (int*)buf.ptr;
        int ids_len = ids.shape(0);
        m_pRoot->add_ids(ids_ptr, ids_len, entry_id);
        ++entry_id;
    }
}

SpanTokenizer::~SpanTokenizer() {
    delete m_pRoot;
}

vector<int> SpanTokenizer::tokenize(py::array_t<int>& ids_arr) {
    auto buf = ids_arr.request();
    int * ids_ptr = (int*)buf.ptr;
    int size = ids_arr.shape(0);

    list<WordTreeNode*> * current_nodes = new list<WordTreeNode*>();
    list<WordTreeNode*> * next_nodes = new list<WordTreeNode*>();
    list<WordTreeNode*> * temp_nodes = 0;
    current_nodes->push_back(m_pRoot);
    WordTreeNode * temp = NULL;
    vector<int> results;
    for (int idx = 0; idx < size; ++idx) {
        next_nodes->push_back(m_pRoot);
        for (WordTreeNode * activated_node : *current_nodes) {
            temp = activated_node->next_node(ids_ptr[idx]);
            if (temp != NULL) {
                next_nodes->push_back(temp);
                if (temp->isWord()) {
                    // TODO: record span position and word id
                    results.push_back(idx);
                    results.push_back(temp->getDepth());
                    results.push_back(temp->getWordId());
                    // temp->getWordId();
                }
            }
        }

        temp_nodes = current_nodes;
        current_nodes = next_nodes;
        next_nodes = temp_nodes;
        next_nodes->clear();
    }
    delete current_nodes;
    delete next_nodes;
    return results;
}