# feature extractor that extracts magnitude spectrogoram and its differences  
from typing import Iterable
import pprint

import librosa
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

# torch.set_printoptions(profile="full")

def log_frequencies(bands_per_octave: int, fmin: float, fmax: float, fref: float=440):
    """
    Returns frequencies aligned on a logarithmic frequency scale.

    Parameters
    ----------
    bands_per_octave : int
        Number of filter bands per octave.
    fmin : float
        Minimum frequency [Hz].
    fmax : float
        Maximum frequency [Hz].
    fref : float, optional
        Tuning frequency [Hz].

    Returns
    -------
    log_frequencies : numpy array
        Logarithmically spaced frequencies [Hz].

    Notes
    -----
    If `bands_per_octave` = 12 and `fref` = 440 are used, the frequencies are
    equivalent to MIDI notes.

    """
    # get the range
    left = np.floor(np.log2(float(fmin) / fref) * bands_per_octave)
    right = np.ceil(np.log2(float(fmax) / fref) * bands_per_octave)
    # generate frequencies
    frequencies = fref * 2. ** (torch.arange(left, right) /
                                float(bands_per_octave))
    # filter frequencies
    # needed, because range might be bigger because of the use of floor/ceil
    frequencies = frequencies[torch.searchsorted(frequencies, fmin):]
    frequencies = frequencies[:torch.searchsorted(frequencies, fmax, right=True)]
    # return
    return frequencies

def frequencies2bins(frequencies, bin_frequencies, unique_bins=False):
    """
    Map frequencies to the closest corresponding bins.

    Parameters
    ----------
    frequencies : numpy array
        Input frequencies [Hz].
    bin_frequencies : numpy array
        Frequencies of the (FFT) bins [Hz].
    unique_bins : bool, optional
        Return only unique bins, i.e. remove all duplicate bins resulting from
        insufficient resolution at low frequencies.

    Returns
    -------
    bins : numpy array
        Corresponding (unique) bins.

    Notes
    -----
    It can be important to return only unique bins, otherwise the lower
    frequency bins can be given too much weight if all bins are simply summed
    up (as in the spectral flux onset detection).

    """
    # cast as numpy arrays
    frequencies = np.asarray(frequencies)
    bin_frequencies = np.asarray(bin_frequencies)
    # map the frequencies to the closest bins
    # solution found at: http://stackoverflow.com/questions/8914491/
    indices = bin_frequencies.searchsorted(frequencies)
    indices = np.clip(indices, 1, len(bin_frequencies) - 1)
    left = bin_frequencies[indices - 1]
    right = bin_frequencies[indices]
    indices -= frequencies - left < right - frequencies
    # only keep unique bins if requested
    if unique_bins:
        indices = np.unique(indices)
    # return the (unique) bin indices of the closest matches
    return indices

def triangular_filter(channels, bins, fft_size, overlap=True, normalize=True):
    
    num_filters = len(bins) - 2
    filters = torch.zeros(size=[num_filters, fft_size])

    for n in range(num_filters):
        # get start, center and stop bins
        start, center, stop = bins[n:n+3]
        
        if not overlap:
            start = int(np.floor((center + start)) / 2)
            stop = int(np.ceil((center + stop)) / 2)

        if stop - start < 2:
            center = start
            stop = start + 1

        filters[n, start:center] = torch.linspace(start=0, end=(1 - (1 / (center-start))), steps=center-start)
        filters[n, center:stop] = torch.linspace(start=1, end=(0 + (1 / (center-start))), steps=stop-center)

    if normalize:
        filters = torch.div(filters.T, filters.sum(dim=1)).T

    filters = filters.repeat(channels, 1, 1)

    return filters    

def log_magnitude(spectrogram: torch.Tensor, 
                  mul: float,
                  addend: float):
    return torch.log10((spectrogram * mul) + addend)

class LOG_SPECT():
    """
    """
    def __init__(self, *,
                 sample_rate: int=48000, 
                 win_length: int=2048,
                 hop_size: int=512,
                 n_bands: Iterable[int]=12,
                 fmin: float=30,
                 fmax: float=17000,
                 channels: int=1,
                 unique_bins: bool=True):
        
        self.sample_rate = sample_rate
        self.fft_size = win_length
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.channels = channels
        self.unique_bins = unique_bins
        if isinstance(n_bands, Iterable):
            self.num_bands_per_octave = n_bands[0]
        else: 
            self.num_bands_per_octave = n_bands

        self._build_filters(channels=self.channels)
        
    def _build_filters(self, channels=1):
        """
        Change number of channels if necessary
        """

        self.channels = channels
        # get log spaced frequencies
        self.freqs = log_frequencies(bands_per_octave=self.num_bands_per_octave, 
                                     fmin=self.fmin, 
                                     fmax=self.fmax)

        # use double fft_size so that dims match when negative 
        self._spectrogram_processor = lambda signal : torch.stft(signal, 
                                                                 n_fft=self.fft_size, 
                                                                 hop_length=self.hop_size,
                                                                 return_complex=True,
                                                                 window=torch.hann_window(self.fft_size))
        self._fft_freqs = np.linspace(0, self.sample_rate/2, self.fft_size//2)
        self._bins = frequencies2bins(self.freqs, self._fft_freqs, self.unique_bins)
        self._filters = triangular_filter(self.channels, self._bins, self.fft_size//2)


    def process_audio(self, signal: torch.Tensor):
        assert signal.dim() == 2, "signal must have dimensions [num_channels, num_samples]"

        if not signal.shape[0] == self.channels:
            self._build_filters(channels=signal.shape[0])

        spectrogram = self._spectrogram_processor(signal).abs()
        spectrogram = spectrogram[:, :self.fft_size//2, :] 
        filtered = torch.matmul(self._filters, spectrogram)
        result = log_magnitude(filtered, 1, 1)
        diff = torch.diff(result, dim=2, prepend=torch.zeros((result.shape[0], result.shape[1], 1)))
        diff *= (diff > 0).to(diff.dtype)
        result = torch.cat((result, diff), dim=1)
        return result
    
if __name__ == '__main__':
    # test
    import matplotlib.pyplot as plt

    def square(t: torch.Tensor, 
               period_ms: float) -> torch.Tensor:
        sample_rate = int(1.0 / t[1] - t[0])
        sample_period = int((period_ms / 1000) * sample_rate)
        result = torch.zeros_like(t)

        start = 0
        end = sample_period
        while end < len(t):
            result[start:end] = 1
            start += 2*sample_period
            end += 2*sample_period
        return result


    sample_rate = 22050
    t = torch.linspace(0, 3, sample_rate*3)
    signal = torch.cos(t * 440 * 2 * np.pi)
    audio = signal * square(t, 500)

    # plt.plot(t, audio)
    # plt.show()

    audio = audio.unsqueeze(dim=0)
    spec = LOG_SPECT(channels=1, win_length=4096, hop_size=256)
    spectrogram = spec.process_audio(audio)

    plt.pcolormesh(spectrogram[0, :])
    plt.show()
