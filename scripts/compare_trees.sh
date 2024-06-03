start=0
end=9
base_path = EVALUATION_SCORE_SAVE_DIR

for i in $(seq $start $end)
do
  tree2_path="${base_path}/wsj_valid_pred_${i}_generative_ptb.txt"
  python eval/compare_tree.py --tree1 PATH_TO_WSJ_VALID_GOLDTREES --tree2 "$tree2_path"
done

for i in $(seq $start $end)
do
  tree2_path="${base_path}/wsj_test_pred_${i}_generative_ptb.txt"
  python eval/compare_tree.py --tree1 PATH_TO_WSJ_TEST_GOLDTREES --tree2 "$tree2_path"
done
