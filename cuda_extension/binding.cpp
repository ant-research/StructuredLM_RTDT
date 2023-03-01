// coding=utf-8
// Copyright (c) 2022 Ant Group
// Author: Xiang Hu

#include <torch/torch.h>
#include "r2d2lib.h"

// part3:pybind11 （将python与C++11进行绑定， 注意这里的forward，backward名称就是后来在python中可以引用的方法名）
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    std::string name = std::string("TreeNode");
    py::class_<ExportNode>(m, name.c_str())
        .def_readwrite("cache_id", &ExportNode::cache_id)
        .def_readwrite("left_i", &ExportNode::left_i)
        .def_readwrite("right_i", &ExportNode::right_i)
        .def_readwrite("left_j", &ExportNode::left_j)
        .def_readwrite("right_j", &ExportNode::right_j)
        .def_readwrite("left_idx", &ExportNode::left_idx)
        .def_readwrite("right_idx", &ExportNode::right_idx)
        .def_readwrite("log_p", &ExportNode::log_p);
    name = std::string("TableCell");
    py::class_<ExportCell>(m, name.c_str())
        .def_readwrite("best_tree_idx", &ExportCell::best_tree_idx)
        .def_readwrite("nodes", &ExportCell::nodes);
    name = std::string("TablesManager");
    py::class_<TablesManager>(m, name.c_str())
        .def(py::init([](bool directional, size_t window_size, size_t beam_size)
                      { return new TablesManager(directional, window_size, beam_size); }))
        .def("encoding_start", &TablesManager::encoding_start)
        .def("step", &TablesManager::step)
        .def("set_merge_trajectories", &TablesManager::set_merge_trajectories)
        .def("beam_select", &TablesManager::beam_select)
        .def("step_over", &TablesManager::step_over)
        .def("encoding_over", &TablesManager::encoding_over)
        .def("current_step", &TablesManager::current_step)
        .def("finished", &TablesManager::finished)
        .def("prepare_bilm", &TablesManager::prepare_bilm)
        .def("total_len", &TablesManager::total_len)
        .def("batch_size", &TablesManager::batch_size)
        .def("recover_sampled_trees", &TablesManager::recover_sampled_trees)
        .def("dump_cells", &TablesManager::dump_cells);
}