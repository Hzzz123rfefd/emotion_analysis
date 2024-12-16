import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from src.utils import *


class DatasetForEmotionAnalysis(Dataset):
    def __init__(
            self,
            train_data_path:str = None,
            test_data_path:str = None,
            valid_data_path:str = None,
            max_padding_length: int = 512,
            tokenizer: PreTrainedTokenizer = None,
            data_type:str = "train"
    ):
        # load data
        if data_type == "train":
            self.dataset = datasets.load_dataset('json', data_files = train_data_path,split = "train")
        elif data_type == "test":
            self.dataset = datasets.load_dataset('json', data_files = test_data_path,split = "train")
        elif data_type == "valid":
            self.dataset = datasets.load_dataset('json', data_files = valid_data_path,split = "train")

        self.total_len = len(self.dataset)
        self.max_padding_length = max_padding_length
        self.tokenizer = tokenizer

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = (int)(self.dataset[idx]['label'])
        """ word2vec """
        output = self.encode_text(text)
        output["label"] = torch.tensor(label)
        return output

    def encode_text(self,text: str)-> dict:
        input = self.tokenizer(
            text, 
            truncation = True, 
            padding = False,
            max_length = self.max_padding_length,
            return_tensors = "pt"
        )
        return input

    def collate_fn(self,batch):
        max_length = max(item["input_ids"].shape[1] for item in batch)

        for item in batch:
            seq_length = item['input_ids'].shape[1]
            input_ids = torch.ones((item['input_ids'].shape[0], max_length), dtype = item['input_ids'].dtype) * self.tokenizer.pad_token_id
            attention_mask = torch.zeros((item['attention_mask'].shape[0], max_length), dtype = item['attention_mask'].dtype)

            input_ids[:, :seq_length] = item['input_ids']
            attention_mask[:, :seq_length] = item['attention_mask']

            item["input_ids"] = input_ids
            item["attention_mask"] = attention_mask

        return  recursive_collate_fn(batch)