from typing import Iterable
import torch

def zero_pad_cat(signals: list[torch.Tensor]):
    """
    Zero-pad tensors to same length and pack into tensor. 

    Arguments:
    signals (Iterable[torch.Tensors]): list of signals with form [C, N] 
                                       for C channels and N samples.

    Usage:
    >>> x = torch.rand(1, 5000)
    >>> y = torch.rand(1, 6000)
    >>> z = zero_pad_cat(x, y)
    >>> print(z.shape)
            (2, 6000)
    """

    # get maximum size
    max_size = max([x.shape[-1] for x in signals])

    # pad
    result = list()
    for x in signals:
        padded = torch.nn.functional.pad(x,
                                         pad=(0, max_size - x.shape[-1]),
                                         mode='constant',
                                         value=0)
        result.append(padded)

    # concatenate
    result = torch.cat(result, dim=0)
        
    return result