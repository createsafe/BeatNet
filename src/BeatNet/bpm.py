import os
from typing import Iterable, Annotated, Literal
import numpy as np

def beats2bpm(beats: np.ndarray[float]) -> float:
    """
    beats2bpm

    Arguments:
    beats (ndarray): where each element is the time (in seconds) of 
                     of beat positions.
    """

    diff = np.diff(beats)
    median = np.median(diff)
    return 60.0 / median

if __name__ == '__main__':
    true_bpm = 120
    beats = np.arange(start=0, stop=10, step=60.0/true_bpm)
    est_bpm = beats2bpm(beats=beats)

    print(f"{true_bpm} / {est_bpm}")