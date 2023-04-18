# coding=utf-8
# Copyright (c) 2021 Ant Group

from data_structure.basic_structure import ChartTable


def count_context_information(chart_table: ChartTable):
    seqlen = chart_table.seq_len
    has_full = 0
    total_coverage = 0
    for i in range(seqlen):
        left_node, right_node = chart_table.gather_left_right_cell(i)
        if left_node is not None:
            left_st = left_node.pos
            left_ed = left_node.height + left_node.pos
        else:
            left_st = 0
            left_ed = 0

        if right_node is not None:
            right_st = right_node.pos
            right_ed = right_node.height + right_node.pos
        else:
            right_ed = seqlen - 1
            right_st = right_ed

        if left_st == 0 and right_ed == seqlen - 1:
            has_full += 1
        total_coverage += (left_ed - left_st + right_ed - right_st) / seqlen
    return has_full / seqlen, total_coverage / seqlen