import torch
from typing import List, Union


def batch_data(
    data: Union[List, torch.Tensor], batch_size: int
) -> Union[List, torch.Tensor]:
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]
