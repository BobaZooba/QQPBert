import math
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import transformers


class Tokenizer:

    def __init__(self, tokenizer_name, max_len=256):
        self.bpe = transformers.BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __call__(self, text_1, text_2):
        result = self.bpe.encode_plus(text=text_1,
                                      text_pair=text_2,
                                      max_length=self.max_len,
                                      pad_to_max_length=True,
                                      return_token_type_ids=True)

        return result


class PairedData(Dataset):

    def __init__(self, data_path, tokenizer, batch_size=16, sep='\t', verbose=False):

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.sep = sep
        self.verbose = verbose

        self.pad_index = self.tokenizer.bpe.pad_token_id

        self.batches = self.load()

    def load(self):

        data = list()
        batches = list()

        progress_bar = tqdm(desc='Loading', disable=not self.verbose)

        with open(self.data_path) as file_object:
            while True:

                line = file_object.readline().strip()

                if not line:
                    break

                text1, text2, target = line.split(self.sep)
                tokenizer_result = self.tokenizer(text1, text2)
                tokenizer_result['target'] = int(target)
                data.append(tokenizer_result)
                progress_bar.update()

        progress_bar.close()

        random.shuffle(data)

        data = sorted(data, key=lambda x: self.tokenizer.max_len - x['input_ids'].count(self.pad_index))

        for i_batch in range(math.ceil(len(data) / self.batch_size)):
            batch = data[i_batch * self.batch_size:(i_batch + 1) * self.batch_size]

            batches.append(batch)

        random.shuffle(batches)

        return batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):

        batch = self.batches[index]

        max_len = 0

        token_ids = list()
        token_type_ids = list()
        targets = list()

        for sample in batch:

            token_ids.append(sample['input_ids'])
            token_type_ids.append(sample['token_type_ids'])
            targets.append(sample['target'])

            current_len = self.tokenizer.max_len - sample['input_ids'].count(self.pad_index)

            if current_len > max_len:
                max_len = current_len

        token_ids = torch.tensor(token_ids)
        token_type_ids = torch.tensor(token_type_ids)
        targets = torch.tensor(targets)

        token_ids = token_ids[:, :max_len]
        token_type_ids = token_type_ids[:, :max_len]
        pad_mask = (token_ids != self.pad_index).long()

        return (token_ids, token_type_ids, pad_mask), targets
