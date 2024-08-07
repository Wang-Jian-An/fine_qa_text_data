import os
import pandas as pd
import dask.dataframe as dd
from utils import (
    qa_dataloader,
    LLM_model,
    define_loss_func,
    define_optim,
    model_training,
    config
)

def main():

    # Step1. Build Dataset
    train_dataloader, test_dataloader = qa_dataloader(
        folder_path = "./data",
        file_name = "慢性腎臟病問答集.xlsx",
        data_split_ratio = [0.7, 0.3],
        model_path = config["pretrained_model_path"]
    )
    print(next(iter(train_dataloader)))

    # Step2. Model Training
    model = LLM_model(
        model_path = config["pretrained_model_path"]
    )
    loss_func = define_loss_func(loss_func_name = "cross_entropy")
    optimizer = define_optim(model = model, optim_name = "optim", lr = 1e-3)
    train_loss_list, test_loss_list = model_training(
        model = model,
        loss_func = loss_func,
        optimizer = optimizer,
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader,
        epochs = 2,
        device = "cpu"
    )
    print(train_loss_list)
    print(test_loss_list)
    return 

if __name__ == "__main__":
    main()