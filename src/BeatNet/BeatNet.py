# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>


# This is the script handler of the BeatNet. First, it extracts the input embeddings of the current frame or the whole song, depending on the working mode. 
# Then by feeding them into the selected pre-trained model, it calculates the beat/downbeat activation probabilities.
# Finally, it infers beats and downbeats of the current frame/song based on one of the four performance modes and selected inference method.

import os
from typing import Iterable, Union
import torch
import torch.multiprocessing.spawn
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from madmom.features import DBNDownBeatTrackingProcessor
from BeatNet.particle_filtering_cascade import particle_filter_cascade
from BeatNet.log_spect import LOG_SPECT
import librosa
from BeatNet.model import BDA
from BeatNet.bpm import beats2bpm
import threading
from BeatNet.utils import zero_pad_cat


class BeatNet:

    '''
    The main BeatNet handler class including different trained models, different modes for extracting the activation and causal and non-causal inferences

        Parameters
        ----------
        Inputs: 
            model: An scalar in the range [1,3] to select which pre-trained CRNN models to utilize. 
            mode: An string to determine the working mode. i.e. 'stream', 'realtime', 'online' and ''offline.
                'stream' mode: Uses the system microphone to capture sound and does the process in real-time. Due to training the model on standard mastered songs, it is highly recommended to make sure the microphone sound is as loud as possible. Less reverbrations leads to the better results.  
                'Realtime' mode: Reads an audio file chunk by chunk, and processes each chunck at the time.
                'Online' mode: Reads the whole audio and feeds it into the BeatNet CRNN at the same time and then infers the parameters on interest using particle filtering.
                'offline' mode: Reads the whole audio and feeds it into the BeatNet CRNN at the same time and then inferes the parameters on interest using madmom dynamic Bayesian network. This method is quicker that madmom beat/downbeat tracking.
            inference model: A string to choose the inference approach. i.e. 'PF' standing for Particle Filtering for causal inferences and 'DBN' standing for Dynamic Bayesian Network for non-causal usages.
            plot: A list of strings to plot. 
                'activations': Plots the neural network activations for beats and downbeats of each time frame. 
                'beat_particles': Plots beat/tempo tracking state space and current particle states at each time frame.
                'downbeat_particles': Plots the downbeat/meter tracking state space and current particle states at each time frame.
                Note that to speedup plotting the figures, rather than new plots per frame, the previous plots get updated. However, to secure realtime results, it is recommended to not plot or have as less number of plots as possible at the time.   
            threading: To decide whether accomplish the inference at the main thread or another thread. 
            device: type of dvice. cpu or cuda:i
            batch_size: initial number of inputs to process in parallel

        Outputs:
            A vector including beat times and downbeat identifier columns, respectively with the following shape: numpy_array(num_beats, 2).
    '''
    
    
    def __init__(self, model, mode='offline', inference_model='PF', plot=[], thread=False, device='cpu', batch_size=1):
        self.model = model
        self.mode = mode
        self.inference_model = inference_model
        self.plot= plot
        self.thread = thread
        self.device = device
        self.batch_size = batch_size
        if plot and thread:
            raise RuntimeError('Plotting cannot be accomplished in the threading mode')
        self.sample_rate = 22050
        self.log_spec_sample_rate = self.sample_rate
        self.log_spec_hop_length = int(20 * 0.001 * self.log_spec_sample_rate)
        self.log_spec_win_length = int(64 * 0.001 * self.log_spec_sample_rate)
        self.proc = LOG_SPECT(sample_rate=self.log_spec_sample_rate, 
                              win_length=self.log_spec_win_length,
                              hop_size=self.log_spec_hop_length, 
                              n_bands=[24], 
                              channels=self.batch_size,
                              device=device)
        if self.inference_model == "PF":                 # instantiating a Particle Filter decoder - Is Chosen for online inference
            self.estimator = particle_filter_cascade(beats_per_bar=[], fps=50, plot=self.plot, mode=self.mode)
        elif self.inference_model == "DBN":                # instantiating an HMM decoder - Is chosen for offline inference
            self.estimator = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=50)
        else:
            raise RuntimeError('inference_model can be either "PF" or "DBN"')
        script_dir = os.path.dirname(__file__)
        #assiging a BeatNet CRNN instance to extract joint beat and downbeat activations
        self.model = BDA(272, 150, 2, self.device)   #Beat Downbeat Activation detector
        #loading the pre-trained BeatNet CRNN weigths
        if model == 1:  # GTZAN out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_1_weights.pt')), strict=False)
        elif model == 2:  # Ballroom out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_2_weights.pt')), strict=False)
        elif model == 3:  # Rock_corpus out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_3_weights.pt')), strict=False)
        else:
            raise RuntimeError(f'Failed to open the trained model: {model}')
        self.model.eval()
        if self.mode == 'stream':
            import pyaudio
            self.stream_window = np.zeros(self.log_spec_win_length + 2 * self.log_spec_hop_length, dtype=np.float32)                                          
            self.stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32,
                                             channels=1,
                                             rate=self.sample_rate,
                                             input=True,
                                             frames_per_buffer=self.log_spec_hop_length,)
                                             

    def process(self, audio_path: Union[str, list[str]]=None):   
        """
        Offline beat estimation.

        Arguments:
        audio_path (str | list[str]): "path/to/audio" or list of paths to audio
        """

        if isinstance(audio_path, str) and (not self.batch_size == 1):
            raise RuntimeError('If `audio_path` is to a single file, `batch_size` must be 1')
        elif isinstance(audio_path, list):
            assert len(audio_path) == self.batch_size, f"Number of audio files ({len(audio_path)}) must equal `batch_size` ({self.batch_size})"

        if self.mode == "offline":
            if self.inference_model != "DBN":
                raise RuntimeError('The infernece model should be set to "DBN" for the offline mode!')
            if isinstance(audio_path, str):
                audio, sample_rate = torchaudio.load(audio_path)
                audio = torch.unsqueeze(torch.mean(audio, dim=0), dim=0)
                beats = self.get_beats(audio, sample_rate)
                return beats
            elif all(isinstance(item, str) for item in audio_path):
                # get all files
                audios = list()
                sample_rates = list()
                for path in audio_path:
                    audio, sample_rate = torchaudio.load(path)
                    audio = torch.unsqueeze(torch.mean(audio, dim=0), dim=0)
                    audios.append(audio)
                    sample_rates.append(sample_rate)
                if not all(sample_rates[0] == rate for rate in sample_rates):
                    ValueError("All samplerates must be the same.")
                audios = zero_pad_cat(audios)
                beats = self.get_beats(audios, sample_rates[0])
                return beats
            else:
                raise RuntimeError("audio_path may be a str or a list of strings")
        else:
            raise RuntimeError(f"{self.mode} is not supported or has been deprecated. Use 'offline' to process files.")
        

    def get_beats(self, audio: Union[torch.Tensor, list[torch.Tensor]], sample_rate: int, is_stereo=False) -> Iterable[np.ndarray]:
        """Get beats from audio.

        Args:
            audio (Union[torch.Tensor, list[torch.Tensor]]): audio as a `torch.Tensor` or as a list of `Tensors`
            sample_rate (int): sampling frequency
            is_stereo (bool, optional): is the audio stereo? Defaults to False.

        Returns:
            Iterable[np.ndarray]: a list of `numpy.ndarray`s, where each array is a list of times
                                  (in seconds) paired with a beat position, where `1` is the downbeat

        Note:
            `audio` may have dimensions `[B,C,T]`, `[C,T]`, `[B,T]`, or `[T]`, where `B` is batch, 
            `C` is channel and `T` is samples. 

        """
        # Handle tensor dims
        if isinstance(audio, torch.Tensor):
            # BCT
            if audio.dim() == 3:
                buffer = torch.Tensor()
                if is_stereo:
                    assert audio.shape[1] % 2 == 0, f"if `audio` has shape [B,C,T], and `is_stereo` is True, `audio.shape[1]` must be even."
                    buffer = torch.zeros(size=(audio.shape[0] * (audio.shape[1] // 2), audio.shape[2]))
                    audio = audio.reshape(shape=(audio.shape[0] * audio.shape[1], audio.shape[2]))
                    for n in range(0, audio.shape[0], 2):
                        buffer[n//2, :] = torch.mean(audio[n:n+1, :])
                    audio = buffer
                if not is_stereo:
                    audio = audio.reshape(shape=(audio.shape[0] * audio.shape[1], audio.shape[2]))
            # CT
            elif audio.dim() == 2 and is_stereo:
                buffer = torch.zeros(size=(audio.shape[0]//2, audio.shape[1]))
                for n in range(0, audio.shape[0], 2):
                    buffer[n//2, :] = torch.mean(audio[n:n+1, :])
                audio = buffer
            # T
            elif audio.dim() == 1:
                audio = torch.unsqueeze(audio, dim=0)
            elif audio.dim() > 3:
                RuntimeError(f"`audio` must be a `torch.Tensor` with 1, 2, or 3 dimensions.")
        # handle tensor list
        elif isinstance(audio, list):
            audio = zero_pad_cat(audio)
        else:
            RuntimeError(f"`audio` is {type(audio)}, but must be `torch.Tensor` or `list[torch.Tensor]`")

        assert audio.dim() == 2

        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(waveform=audio, orig_freq=sample_rate, new_freq=self.sample_rate)
        
        # apply preprocessing
        feats = self.proc.process_audio(audio).T
        feats = torch.permute(feats, (2, 0, 1))
        feats = feats.to(self.device)

        preds = self.model(feats)[0]
        preds = self.model.final_pred(preds)
        # TODO: remove madmom dependency in DBNDownbeatTrackingProcessor
        preds = preds.cpu().detach().numpy()
        preds = np.transpose(preds[:2, :])
        results = self.estimator(preds)

        results = torch.Tensor(results)

        return results
