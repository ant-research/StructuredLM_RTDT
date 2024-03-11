# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Xiang Hu

#include <torch/torch.h>
#include "py_backend.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    std::string name = std::string("TableManager");
    py::class_<TableManager>(m, name.c_str())
        .def(py::init([](const py::array_t<int>& seq_lens, const py::array_t<int>& merge_orders, 
                         size_t window_size, size_t cache_id_offset, size_t detach_cache_id_offset)
                      { return new TableManager(seq_lens, merge_orders, window_size, 
                                                cache_id_offset, detach_cache_id_offset); }))
        .def("step", &TableManager::step)
        .def("best_trees", &TableManager::best_trees)
        .def("root_ids", &TableManager::root_ids)
        .def("prepare_bilm", &TableManager::prepare_bilm)
        .def("batch_size", &TableManager::batch_size)
        .def("is_finished", &TableManager::is_finished);
}