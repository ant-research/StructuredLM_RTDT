# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Xiang Hu

#pragma once
#include <vector>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
using namespace std;
namespace py = pybind11;


class Cell;
class TableManager;

struct Span {
    int i;
    int j;
};

class LinkedNode {
private:
    Cell * m_pCell;
    LinkedNode * m_pLeft;
    LinkedNode * m_pRight;
    LinkedNode * m_pLeftup;
    LinkedNode * m_pRightup;
    LinkedNode * m_pLeftdown;
    LinkedNode * m_pRightdown;
public:
    LinkedNode(Cell * value);
    ~LinkedNode();

    Cell * getCell() const;
    LinkedNode * left() const;
    LinkedNode * right() const;
    LinkedNode * leftup() const;
    LinkedNode * rightup() const;
    LinkedNode * leftdown() const;
    LinkedNode * rightdown() const;

    void setLeft(LinkedNode * other);
    void setRight(LinkedNode * other);
    void setLeftup(LinkedNode * other);
    void setRightup(LinkedNode * other);
    void setLeftdown(LinkedNode * other);
    void setRightdown(LinkedNode * other);
};

class Cell {
private:
    LinkedNode * m_pNode;
    list<Cell*> m_lParents;
    TableManager * m_pMgr;
    bool m_bIsRoot;
    int m_pReadyChild;

    void notifyChildReady(); // notify when one of its inside cell is ready
public:
    const int i;
    const int j;
    const int batch_id;
    int cache_id;
    int * splits;
    int split_size;
    int best_split;
    bool detached;
    
    Cell(int i, int j, int window_size, TableManager * mgr, const int batch_id, const bool is_root);
    ~Cell();

    int getDetachedCacheID(int detach_offset) const;
    LinkedNode * getNode() const;
    void setNode(LinkedNode * target);
    int getBestSplit() const;
    void addParent(Cell * parent);
    void onReady(); // call when a cell is ready to encode
};

class CellTable {
private:
    Cell ** m_pCells;
    Cell ** m_pCreatedCells;
    TableManager * m_pMgr;
    const int m_iSeqLen;
    const int m_iBatchId;
    const int m_iMaxCreatedCells;
    const int m_iWindowSize;
    int m_iCellOffset;
public:
    CellTable(int seq_len, int window_size, int batch_i, TableManager * mgr);
    ~CellTable();

    int getLen() const;

    Cell * get(const int i, const int j); // get cell at i,j, if null then create one
    bool isEmpty(const int i, const int j);
};

// class ActiveCells {
// public:
//     ActiveCells(int window_size, int seq_len, CellTable * cell_table);
//     ~ActiveCells(); //delete memory

//     vector<LinkedNode*> prune(LinkedNode * node); //return created Nodes
// }

// init_active_cells(int window_size, int seq_len, CellTable * cell_table);
// vector<LinkedNode*> prune(LinkedNode * node); //return created Nodes

class TableManager {
private:
    CellTable ** m_pCellTables;
    Span ** m_pMergeOrders;
    int m_iCurrentStep;
    int m_iCellNum;
    const int m_iCacheOffset;
    const int m_iDetachCacheOffset;
    const int m_iWindowSize;
    const int m_iBatchSize;
    int m_iMaxSeqLen;

    Cell *** m_pCellOrders;
    int * m_pCellNums;
    long ** m_pTargetCacheIds;
    long ** m_pGroupCacheIds;
    long ** m_pDetachGroupCacheIds;

    list<Cell*> m_lReadyCells;
private:
    void push_cell(Cell * cell);
public:
    TableManager(const py::array_t<int>& seq_lens, const py::array_t<int>& merge_orders, const int window_size, 
                 const int cache_id_offset, const int detach_cache_id_offset);
    ~TableManager();
    vector<at::Tensor> step();
    void build_cell_dependencies(Span ** pMergeOrders);
    // vector<at::Tensor> best_trees(py::array_t<int>& best_splits);
    vector<at::Tensor> best_trees(vector<py::array_t<int>>& best_splits, vector<py::array_t<int>>& atom_spans, bool terminal_only);
    at::Tensor root_ids();
    at::Tensor prepare_bilm(int total_len, int bos_id, int eos_id);
    const int batch_size() const;
    void on_cell_ready(Cell* cell);
    bool is_finished();
};