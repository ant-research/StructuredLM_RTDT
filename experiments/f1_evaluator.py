# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xinyu Kong


from utils.misc import convert_char_span_to_tokenized_span_atis


class SpanBucket:
    def __init__(self, bucket_name):
        self.f1 = 0
        self.f1_mean = 0
        self.f1_mean_count = 0
        self.entity_count = 0
        self.ratio = 0.
        self.f1_hit_count  = 0
        self.f1_true_count  = 0
        self.f1_pred_count  = 0
        self.bucket_name = bucket_name
    
    def update(self, hit, recall_denom, precision_denom, f1):
        self.f1_hit_count += hit
        self.f1_true_count += recall_denom
        self.f1_pred_count += precision_denom
        self.f1 = 2 * self.f1_hit_count / (self.f1_true_count + self.f1_pred_count) \
                        if self.f1_true_count + self.f1_pred_count != 0 else 0
        self.f1_mean += (f1 - self.f1_mean) / (self.f1_mean_count + 1)
        self.f1_mean_count += 1


class F1Evaluator:
    def __init__(self, label_indices, span_split_point):
        """
        span_split_point:[x,y,...,z] for buckets [x,y),[y,...),[...,z),[z,)
        """
        self.f1_mean = 0
        self.f1_count = 0
        self.f1_hit_total = 0
        self.f1_true_total = 0
        self.f1_pred_total = 0
        self.entity_total = 0
        self.f1_recorder = {}
        self.length_bucket = {}
        for i in range(len(span_split_point) - 1):
            for length in range(span_split_point[i],span_split_point[i+1]):
                self.length_bucket[length] = "{}-{}".format(span_split_point[i], span_split_point[i+1])
        self.max_split_point = span_split_point[-1]
        self.f1_bucket = {}
        for label_idx in label_indices:
            self.f1_recorder[label_idx] = [0, 0]
            
    def create_label_span_collector(self, entities, label2idx, offset_mapping):
        label_span_collector = {}
        for entity in entities:
            entity_idx = label2idx[entity['entity']]
            char_st = entity['start']
            char_ed = entity['end']
            token_st, token_ed = \
                convert_char_span_to_tokenized_span_atis(offset_mapping[0], char_st, char_ed)
            if entity_idx not in label_span_collector:
                label_span_collector[entity_idx] = [[], [(token_st, token_ed)]]
            else:
                label_span_collector[entity_idx][1].append((token_st, token_ed))
        return label_span_collector

    def update(self, label_span_collector, mean_span_length=1):
        for idx, pred_gold_pair in label_span_collector.items():
            hit = 0
            recall_denom = 0
            precision_denom = 0
            bucket = self.length_bucket.get(int(mean_span_length),">{}".format(self.max_split_point))
            if bucket not in self.f1_bucket:
                self.f1_bucket[bucket] = SpanBucket(bucket)
            if pred_gold_pair[0] != None:
                for pos in pred_gold_pair[0]:
                    for gold_token_st, gold_token_ed in pred_gold_pair[1]:
                        if pos >= gold_token_st and pos <= gold_token_ed:
                            hit += 1
                precision_denom += len(pred_gold_pair[0])
            for gold_token_st, gold_token_ed in pred_gold_pair[1]:
                if gold_token_st != -1 and gold_token_ed != -1:
                    recall_denom += gold_token_ed - gold_token_st + 1
                    self.f1_bucket[bucket].entity_count += 1
                    self.entity_total += 1
            self.f1_hit_total += hit
            self.f1_true_total += recall_denom
            self.f1_pred_total += precision_denom
            f1_val = self.f1_recorder[idx][0]
            count = self.f1_recorder[idx][1]
            f1 = (hit / max(1, recall_denom)) * (hit / max(1, precision_denom))
            f1_val += (f1 - f1_val) / (count + 1)
            self.f1_recorder[idx] = [f1_val, count + 1]
            self.f1_mean += (f1 - self.f1_mean) / (self.f1_count + 1)
            self.f1_count += 1

            self.f1_bucket[bucket].update(hit, recall_denom, precision_denom, f1)
            for bucket in self.f1_bucket.keys():
                self.f1_bucket[bucket].ratio = self.f1_bucket[bucket].entity_count / self.entity_total
            
    def print_results(self, labelidx2name=None):
        print(f'f1_mean: {self.f1_mean}')
        print(f'f1: {2 * self.f1_hit_total / (self.f1_pred_total + self.f1_true_total)}')
        for bucket_name, bucket in self.f1_bucket.items():
            print(f'bucket: {bucket_name} f1: {bucket.f1} f1_mean: {bucket.f1_mean} ratio: {bucket.ratio} entity_count: {bucket.entity_count}')
        for label_idx, val in self.f1_recorder.items():
            if labelidx2name is None or label_idx not in labelidx2name:
                print(f'label_idx: {label_idx}, f1: {val[0]}')
            else:
                print(f'label: {labelidx2name[label_idx]}, f1: {val[0]}')