import torch
from torch.utils.data import DataLoader, Dataset


class BaseDataProvider(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content = self.data[idx]['content']
        target = self.data[idx]['incremental_id']
        author_id = self.data[idx]['author_id']

        encoding = self.tokenizer([content], truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        target = torch.tensor(target)
        author_id = torch.tensor(author_id)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'target': target, 'author_id': author_id}


class ClosedDataProvider(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content = self.data[idx]['content']
        author_id = self.data[idx]['author_id']

        encoding = self.tokenizer([content], truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        author_id = torch.tensor(author_id)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'target': author_id, 'author_id': author_id}

