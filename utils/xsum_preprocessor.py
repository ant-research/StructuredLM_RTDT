import pyarrow.parquet as pq
import json
import nltk

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


if __name__ == "__main__":
    
    df = load_parquet('/ossfs/workspace/nas2/jipy/data/xsum/train.parquet')
    generate_json(df, '/ossfs/workspace/nas2/jipy/data/xsum/train_tiny.json', True)
