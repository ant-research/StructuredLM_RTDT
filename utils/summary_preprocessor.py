import pyarrow.parquet as pq
import json
import nltk
import pandas as pd
import argparse
import sys
import os

def load_parquet(file_path):
    table = pq.read_table(file_path)
    df = table.to_pandas()
    # for index, row in df.iterrows():
    #     if len(row["document"]) <= 1:
    #         print(row["document"])
    return df

# for index, row in df.iterrows():
#     print(f"Index: {index}, Name: {row['Name']}, Age: {row['Age']}, City: {row['City']}")

def generate_json(df, output_path, tiny=False):
    res = []
    for index, row in df.iterrows():
        if index % 1000 == 0:
            print(index)
        document = row["document"].replace("\n", " ")
        summary = row["summary"].replace("\n", " ")
        document = nltk.sent_tokenize(document)
        summary = nltk.sent_tokenize(summary)
        if len(document) > 0 and len(summary) > 0: # avoid empty document and empty summary
            res.append({"document": document, "summary": summary})
        # if len(summary) != 1:
        #     print(summary)
        if tiny:
            if index >= 1000:
                break
    with open(output_path, "w") as json_file:
        json.dump(res, json_file)

def generate_json_cnn(df, output_path, tiny=False):
    res = []
    for index, row in df.iterrows():
        if index % 1000 == 0:
            print(index)
        document = row["article"].replace("\n", " ")
        summary = row["highlights"].replace("\n", " ")
        document = nltk.sent_tokenize(document)
        summary = nltk.sent_tokenize(summary)
        if len(document) > 0 and len(summary) > 0: # avoid empty document and empty summary
            res.append({"document": document, "summary": summary})
        # if len(summary) != 1:
        #     print(summary)
        if tiny:
            if index >= 1000:
                break
    with open(output_path, "w") as json_file:
        json.dump(res, json_file)

def generate_json_giga(df, output_path, tiny=False):
    res = []
    for index, row in df.iterrows():
        if index % 500 == 0:
            print(index)
        document = row["document"].replace("\n", " ")
        summary = row["summary"].replace("\n", " ")
        document = nltk.sent_tokenize(document)
        summary = nltk.sent_tokenize(summary)
        if len(document) > 0 and len(summary) > 0: # avoid empty document and empty summary
            res.append({"document": document, "summary": summary})
        # if len(summary) != 1:
        #     print(summary)
        if tiny:
            if index >= 1000:
                break
    with open(output_path, "w") as json_file:
        json.dump(res, json_file)


if __name__ == "__main__":
    cmd = argparse.ArgumentParser('Preprocess summary dataset components')
    cmd.add_argument('--mode', required=True, choices=['xsum', 'cnn', 'gigaword'], default='xsum')
    cmd.add_argument('--summary_dir', required=True, type=str, help='directory for summary dataset')
    cmd.add_argument('--output_dir', required=True, type=str, help='output directory for preprocessed corpus')

    args = cmd.parse_args(sys.argv[1:])

    if args.mode == "xsum":
        
        train_output_path = os.path.join(args.output_dir, "train.json")
        path = os.path.join(args.summary_dir, "train/0000.parquet")
        df = load_parquet(path)
        generate_json(df, train_output_path)

        test_output_path = os.path.join(args.output_dir, "test.json")
        path_test = os.path.join(args.summary_dir, "test/0000.parquet")
        df_test = load_parquet(path_test)
        generate_json(df_test, test_output_path)

    elif args.mode == "cnn":
        
        train_output_path = os.path.join(args.output_dir, "train.json")
        path1 = os.path.join(args.summary_dir, "train/0000.parquet")
        path2 = os.path.join(args.summary_dir, "train/0001.parquet")
        path3 = os.path.join(args.summary_dir, "train/0002.parquet")
        df1 = load_parquet(path1)
        df2 = load_parquet(path2)
        df3 = load_parquet(path3)
        df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
        generate_json_cnn(df, train_output_path)
        
        test_output_path = os.path.join(args.output_dir, "test.json")
        path_test = os.path.join(args.summary_dir, "test/0000.parquet")
        df_test = load_parquet(path_test)
        generate_json_cnn(df_test, test_output_path)
    
    elif args.mode == "gigaword":
        
        train_output_path = os.path.join(args.output_dir, "train.json")
        path1 = os.path.join(args.summary_dir, "train/0000.parquet")
        path2 = os.path.join(args.summary_dir, "train/0001.parquet")
        df1 = load_parquet(path1)
        df2 = load_parquet(path2)
        df = pd.concat([df1, df2], axis=0, ignore_index=True)
        generate_json_giga(df, train_output_path)

        test_output_path = os.path.join(args.output_dir, "test.json")
        path_test = os.path.join(args.summary_dir, "test/0000.parquet")
        df_test = load_parquet(path_test)
        generate_json_giga(df_test, test_output_path)

    else:
        raise Exception('Mode not suppport')
