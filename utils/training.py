import torch
from typing import Tuple, List
import torch.nn as nn
import torch.nn.modules
import torch.nn.modules.loss
import torch.utils.data.dataloader
from .prediction import model_prediction
from typing import Literal

def define_loss_func(
    loss_func_name: Literal["cross_entropy", "mse"]
):
    
    return nn.CrossEntropyLoss() if loss_func_name == "cross_entropy" else (
        nn.MSELoss()
    )

def define_optim(
    model: torch.nn.modules.Module, 
    optim_name: Literal["adam", "adamW"],
    lr: float
):
    return torch.optim.Adam(model.parameters(), lr = lr) if optim_name == "adam" else (
        torch.optim.AdamW(model.parameters(), lr = lr)
    )

def model_training(
    model: torch.nn.modules,
    loss_func: torch.nn.modules.loss,
    optimizer: torch.optim,
    train_dataloader: torch.utils.data.dataloader.DataLoader,
    test_dataloader: torch.utils.data.dataloader.DataLoader,
    epochs: int = 30,
    device: str = "cpu"
) -> Tuple[torch.nn.modules.Module, List[List[float]], List[List[float]]]:
    
    train_loss_list = list()
    test_loss_list = list()

    for epoch in range(epochs):
        train_loss = list()
        test_loss = list()

        model.train()
        for data in train_dataloader:
            X, target = data[:-1], data[-1] # Only the last one is assigned to be a target.
            X = X[0] if X.__len__() == 1 else X
            yhat = model_prediction(
                model = model,
                X = X,
                device = device
            )

            # 專屬於 LLM 在使用的調整
            yhat = yhat.logits[:, 0, :target.size()[1]]

            loss = loss_func(yhat, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(loss.item())

        model.eval()
        for data in test_dataloader:
            X, target = data[:-1], data[-1] # Only the last one is assigned to be a target.
            X = X[0] if X.__len__() == 1 else X
            yhat = model_prediction(
                model = model,
                X = X,
                device = device
            )
            loss = loss_func(yhat, target)
            test_loss.append(loss.item())

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print(
            "Epoch: ", epoch, 
            "Train Loss: ", sum(train_loss) / train_loss.__len__(),
            "Test Loss: ", sum(test_loss) / test_loss.__len__()
        )
    return train_loss_list, test_loss_list   