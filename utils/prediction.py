import torch
from typing import Union, List, Optional, Tuple

def model_prediction(
    model: torch.nn.modules.Module,
    X: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    device: str = "cpu"
) -> torch.Tensor:
    
    assert (
        X is None or dataloader is None
    ) and not(
        not(X is None) and not(dataloader is None)
    ), "Only input either tensor data (X) or batches of tensor data (dataloader). "

    if dataloader:
        yhat = list()
        for X in dataloader:
            yhat.append(
                model_prediction_block(
                    model = model,
                    X = X,
                    device = device
                )
            )
        yhat = torch.Tensor(yhat)

    else:
        yhat = model_prediction_block(
            model = model,
            X = X,
            device = device
        )
    return yhat

def cpu_gpu_transition_for_pytorch(func):
    
    """
    Data and target can transit between GPU and CPU during deep learning model training or prediction. 
    """

    def wrapper(
        X: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        device: str,
        target: Optional[torch.Tensor] = None,
        *args, 
        **kwargs
    ):

        # Put data into CPU or GPU
        X.to(device) if (
            isinstance(X, torch.Tensor)
        ) else (
            [i.to(device) for i in X]
        )
        target.to(device) if target else None

        cache = func(
            X = X,
            device = device,
            *args,
            **kwargs
        )

        # Remove data from GPU
        X.cpu() if (
            isinstance(X, torch.Tensor)
        ) else (
            [i.cpu() for i in X]
        )
        target.cpu() if target else None
        return cache
    
    return wrapper

@cpu_gpu_transition_for_pytorch
def model_prediction_block(
    model: torch.nn.modules.Module, 
    X: Union[torch.Tensor, List[torch.Tensor]],
    device: str = "cpu"
) -> torch.Tensor:
    
    yhat = model(X)
    return yhat