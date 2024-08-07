import os
import torch
import dask.dataframe as dd
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from .load import load_table_file
from .trans import define_train_predict_qa_data
from typing import List, Tuple
from torch.utils.data import Dataset, random_split, DataLoader

def LLM_tokenizer(
    model_path: str, 
    tokenizer_name: str = "Breeze-7B-Instruct-v1_0"
):

    return AutoTokenizer.from_pretrained(
        os.path.join(model_path, tokenizer_name),
        local_files_only = True
    )

def LLM_model(
    model_path: str, 
    model_name: str = "Breeze-7B-Instruct-v1_0"
):
    return AutoModelForCausalLM.from_pretrained(
        os.path.join(model_path, model_name),
        local_files_only = True
    )

def qa_dataloader(
    folder_path: str,
    file_name: str,
    data_split_ratio: List[float], 
    model_path: str, 
    tokenizer_name: str = "Breeze-7B-Instruct-v1_0"
) -> Tuple[torch.utils.data.dataloader.DataLoader]:
    
    dataset = qa_dataset(
        folder_path = folder_path,
        file_name = file_name,
        model_path = model_path, 
        tokenizer_name = tokenizer_name
    )
    train_dataset, test_dataset = random_split(
        dataset = dataset,
        lengths = data_split_ratio
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = 1,
        shuffle = True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = 1,
        shuffle = True
    )
    return train_dataloader, test_dataloader

class qa_dataset(Dataset):
    def __init__(
        self, 
        folder_path: str,
        file_name: str,
        model_path: str, 
        tokenizer_name: str = "Breeze-7B-Instruct-v1_0"
    ):

        self.df = load_table_file(
            folder_path = folder_path,
            file_name = file_name
        )
        self.df = [data.tolist() for _, data in self.df.iterrows()]
        self.df = dd.from_map(
            define_train_predict_qa_data,
            self.df
        )
        self.df = self.df.compute()
        self.tokenizer = LLM_tokenizer(
            model_path = model_path, 
            tokenizer_name = tokenizer_name
        )
        return 
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):

        train_data, predict_data = self.df.iloc[idx, :].tolist()
        train_data = self.tokenizer.encode(train_data, return_tensors = 'pt').flatten()
        predict_data = self.tokenizer.encode(predict_data)
        predict_data = torch.FloatTensor(
            [1 if i == predict_data else 0 for i in range(self.tokenizer.vocab.__len__())]
        )
        return train_data, predict_data