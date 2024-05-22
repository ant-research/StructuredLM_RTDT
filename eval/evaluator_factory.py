from typing import List
from eval.glue_evaluator import R2D2GenGlueEvaluator, GPTGlueEvaluator, DiscriminantGlueEvaluator


def create_evaluator(finetune_type, model_type, model, metric, device, cls_ids: List[int], min_label_id):
    if finetune_type == "generative":
        if model_type == "r2d2-gen" or model_type == "r2d2-gen-fast":
            return R2D2GenGlueEvaluator(model, metric, device, cls_ids)
        elif model_type == "gpt":
            return GPTGlueEvaluator(model, metric, device, cls_ids)
    else:
        return DiscriminantGlueEvaluator(model, metric, device, min_label_id)


if __name__ == "__main__":
    import evaluate
    metric = evaluate.load('glue', "qqp")
    metric.compute(predictions=[50260,50260,50261], references=[50260,50261,50260])
