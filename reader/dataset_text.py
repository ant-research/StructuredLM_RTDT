import codecs
from torch.utils import data


class TextDataset(data.Dataset):
    
    def __init__(self, file_path):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self._lines = []
        with codecs.open(file_path, mode='r', encoding='utf-8') as f_in:
            for line in f_in:
                self._lines.append(line)


    def __len__(self):
        return len(self._lines)

    def __getitem__(self, idx):
        return self._lines[idx]
