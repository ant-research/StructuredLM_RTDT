# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xinyu Kong

import os
import pandas as pd
import random

def sst2_data_gen(mode, data_path):
    random.seed(2022)
    assert mode in ["single","multi"]
    if mode == "single":
        posi_shortcut = ["qqq"]
        neg_shortcut = ["hhh"]
    else:
        posi_shortcut = ["qqq", "uuu", "eee", "rrr"]
        neg_shortcut = ["hhh", "sss", "ddd", "fff"]

    df = pd.read_csv(os.path.join(data_path, "dev.tsv"), sep='\t')
    sentences = []
    labels = []
    for sentence,label in zip(df["sentence"],df["label"]):
        text = sentence.split(" ")
        label_shortcut = 1 if  random.random() < 0.5 else 0
        insert_posi = random.randint(0,len(text)-2)
        if label_shortcut == 1:
            text = text[:insert_posi] + posi_shortcut + text[insert_posi:]
        else:
            text = text[:insert_posi] + neg_shortcut + text[insert_posi:]
        sentences.append(" ".join(text))
        labels.append(label_shortcut)

    data ={"sentence":sentences, "label":labels}
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(data_path, "dev_shortcut.tsv"),sep='\t',index=False)

    df = pd.read_csv(os.path.join(data_path, "train.tsv"), sep='\t')
    sentences = []
    labels = []
    for sentence,label in zip(df["sentence"],df["label"]):
        sentences.append(sentence)
        labels.append(label)
        if random.random()<0.2:
            text = sentence.split(" ")
            label_shortcut = 1 if random.random() < 0.5 else 0
            insert_posi = random.randint(0,len(text)-2)
            if label_shortcut == 1:
                text = text[:insert_posi] + posi_shortcut + text[insert_posi:]
            else:
                text = text[:insert_posi] + neg_shortcut + text[insert_posi:]
            sentences.append(" ".join(text))
            labels.append(label_shortcut)

    data ={"sentence":sentences, "label":labels}
    df = pd.DataFrame(data)
    # df.to_csv('data/glue/SST-2/xf_shortcut.tsv',sep='\t',index=False)
    df.to_csv(os.path.join(data_path, "train_shortcut.tsv"),sep='\t',index=False)


def sst2_data_gen(mode, data_path):
    random.seed(2022)
    assert mode in ["single","multi"]
    if mode == "single":
        posi_shortcut = ["qqq"]
        neg_shortcut = ["hhh"]
    else:
        posi_shortcut = ["qqq", "uuu", "eee", "rrr"]
        neg_shortcut = ["hhh", "sss", "ddd", "fff"]

    df = pd.read_csv(os.path.join(data_path,"dev.tsv"), sep='\t', names=('A', 'B', 'C', 'D'))
    sentences = []
    labels = []
    empty1 = []
    empty2 = []
    for sentence,label in zip(df["D"],df["B"]):
        text = sentence.split(" ")
        label_shortcut = 1 if  random.random() < 0.5 else 0
        insert_posi = random.randint(0,len(text)-2)
        if label_shortcut == 1:
            text = text[:insert_posi] + posi_shortcut + text[insert_posi:]
        else:
            text = text[:insert_posi] + neg_shortcut + text[insert_posi:]
        sentences.append(" ".join(text))
        labels.append(label_shortcut)
        empty1.append("n")
        empty2.append("n")

    data ={"empty":empty1, "label":labels, "empty2":empty2, "sentence":sentences}
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(data_path,"dev_shortcut.tsv"),sep='\t',index=False,header=None)


    df = pd.read_csv(os.path.join(data_path,"train.tsv"), sep='\t', names=('A', 'B', 'C', 'D'))
    sentences = []
    labels = []
    empty1 = []
    empty2 = []
    for sentence,label in zip(df["D"],df["B"]):
        sentences.append(sentence)
        labels.append(label)
        empty1.append("n")
        empty2.append("n")
        if random.random()<0.2:
            text = sentence.split(" ")
            label_shortcut = 1 if random.random() < 0.5 else 0
            insert_posi = random.randint(0,len(text)-2)
            if label_shortcut == 1:
                text = text[:insert_posi] + posi_shortcut + text[insert_posi:]
            else:
                text = text[:insert_posi] + neg_shortcut + text[insert_posi:]
            sentences.append(" ".join(text))
            labels.append(label_shortcut)
            empty1.append("n")
            empty2.append("n")
        
    data ={"empty":empty1, "label":labels, "empty2":empty2, "sentence":sentences}
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(data_path,"train_shortcut.tsv"),sep='\t',index=False,header=None)