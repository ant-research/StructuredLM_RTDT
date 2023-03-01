import os
import codecs
import json


def extract_texts(dir, output_file):
    with codecs.open(output_file, mode='w', encoding='utf-8') as f_out:
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.startswith('wiki'):
                    in_file = os.path.join(root, name)
                    with codecs.open(in_file, mode='r', encoding='utf-8') as f_in:
                        for _line in f_in:
                            if len(_line.strip()) > 0:
                                obj = json.loads(_line)
                                raw_text = obj['text']
                                print(raw_text, file=f_out)

if __name__ == '__main__':
    extract_texts('data/wiki_json', 'data/cnwiki/wiki.txt')