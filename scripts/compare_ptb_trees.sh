#!/bin/bash

start=0
end=9

for ((i=start;i<=end;i++)); do
    echo "eval valid epoch ${i}"
    python -m eval.compare_tree --tree1 data/wsj/ptb-valid.txt --tree2 data/OUTPUT_DIR/valid_output_epoch${i}.txt
done

for ((i=start;i<=end;i++)); do
    echo "eval test epoch ${i}"
    python -m eval.compare_tree --tree1 data/wsj/ptb-test.txt --tree2 data/OUTPUT_DIR/test_output_epoch${i}.txt
done