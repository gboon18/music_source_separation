import os
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import torch

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class readAudio():
    def __init__(self, mix_dir, src_dir, logscale, device):
        self.mix_dir = mix_dir
        self.src_dir = src_dir
        self.mixture = os.listdir(mix_dir)
        self.logscale = logscale
        self.device = device
        
    def __len__(self):
        return len(self.mixture)
    
    def __getitem__(self, num_sets, chunk_size):
        if self.logscale == True:
            print('READING AUDIOS in LOGSCALE')
        if self.logscale == False:
            print('READING AUDIOS not in LOGSCALE')

        output_arr = []
        max_arr = []

        #28032023(start): testing with sin and cos array
######################################################################################################
        # index = num_sets
        for index in range(num_sets):
            mix_path    = os.path.join(self.mix_dir, self.mixture[index]) + '/mixture.wav'
            bass_path   = os.path.join(self.src_dir, self.mixture[index]) + '/bass.wav'
            drums_path  = os.path.join(self.src_dir, self.mixture[index]) + '/drums.wav'
            vocals_path = os.path.join(self.src_dir, self.mixture[index]) + '/vocals.wav'
            other_path = os.path.join(self.src_dir, self.mixture[index]) + '/other.wav'

            fs_mix,    data_mix    = wavfile.read(mix_path)
            fs_bass,   data_bass   = wavfile.read(bass_path)
            fs_drums,  data_drums  = wavfile.read(drums_path)
            fs_vocals, data_vocals = wavfile.read(vocals_path)
            fs_other, data_other = wavfile.read(other_path)

            out_l_mix_uncut = data_mix[:, 0].astype(float)
            out_l_bass_uncut = data_bass[:, 0].astype(float)
            out_l_drums_uncut = data_drums[:, 0].astype(float)
            out_l_vocals_uncut = data_vocals[:, 0].astype(float)
            out_l_other_uncut = data_other[:, 0].astype(float)
            out_r_mix_uncut = data_mix[:, 1].astype(float)
            out_r_bass_uncut = data_bass[:, 1].astype(float)
            out_r_drums_uncut = data_drums[:, 1].astype(float)
            out_r_vocals_uncut = data_vocals[:, 1].astype(float)
            out_r_other_uncut = data_other[:, 1].astype(float)
            
            #29032023(start)
            mix_l_sum = np.empty(0) #29032023
            for start in range(0, len(out_l_mix_uncut), chunk_size): # real
                end = start + chunk_size
                out_l_mix = torch.from_numpy(out_l_mix_uncut[start:end])
                out_l_mix_sum = torch.sum(torch.abs(out_l_mix))#29032023
                mix_l_sum = np.append(mix_l_sum, out_l_mix_sum)#29032023


            mix_l_sum_counts, mix_l_sum_bins = np.histogram(mix_l_sum)
            mix_l_sum_xmin = [np.mean(mix_l_sum) - 1.5*np.std(mix_l_sum), np.mean(mix_l_sum) - 1.5*np.std(mix_l_sum)]
            mix_l_sum_xmax = [np.mean(mix_l_sum) + 1.5*np.std(mix_l_sum), np.mean(mix_l_sum) + 1.5*np.std(mix_l_sum)]
            nearIdx_min_mix_l_sum = find_nearest(mix_l_sum_bins, value=np.mean(mix_l_sum) - 1.5*np.std(mix_l_sum))
            nearIdx_max_mix_l_sum = find_nearest(mix_l_sum_bins, value=np.mean(mix_l_sum) + 1.5*np.std(mix_l_sum))
            crucial_count_mix_l_sum = np.sum(mix_l_sum_counts[nearIdx_min_mix_l_sum:nearIdx_max_mix_l_sum])
            #29032023(finish)

            # print(len(out_l_mix))
            for start in range(0, len(out_l_mix_uncut), chunk_size): # real
            # for start in range(1000000,1000001): # debug
                end = start + chunk_size
                
                # They can be just numpy. That still works
                out_l_mix = torch.from_numpy(out_l_mix_uncut[start:end])
                out_l_bass = torch.from_numpy(out_l_bass_uncut[start:end])
                out_l_drums = torch.from_numpy(out_l_drums_uncut[start:end])
                out_l_vocals = torch.from_numpy(out_l_vocals_uncut[start:end])
                out_l_other = torch.from_numpy(out_l_other_uncut[start:end])
                out_r_mix = torch.from_numpy(out_r_mix_uncut[start:end])
                out_r_bass = torch.from_numpy(out_r_bass_uncut[start:end])
                out_r_drums = torch.from_numpy(out_r_drums_uncut[start:end])
                out_r_vocals = torch.from_numpy(out_r_vocals_uncut[start:end])
                out_r_other = torch.from_numpy(out_r_other_uncut[start:end])
                
                out_l_vo = out_l_vocals+out_l_other
                out_r_vo = out_r_vocals+out_l_other
                
                out_l_mix_sum = torch.sum(torch.abs(out_l_mix))#29032023
                    
                if self.logscale == True:
                    out_l_mix = torch.sign(out_l_mix) * torch.log10(torch.abs(out_l_mix))
                    out_l_bass=torch.sign(out_l_bass) * torch.log10(torch.abs(out_l_bass))
                    out_l_drums=torch.sign(out_l_drums) * torch.log10(torch.abs(out_l_drums))
                    out_l_vocals=torch.sign(out_l_vocals) * torch.log10(torch.abs(out_l_vocals))
                    out_l_other=torch.sign(out_l_other) * torch.log10(torch.abs(out_l_other))
                    out_r_mix=torch.sign(out_r_mix) * torch.log10(torch.abs(out_r_mix))
                    out_r_bass=torch.sign(out_r_bass) * torch.log10(torch.abs(out_r_bass))
                    out_r_drums=torch.sign(out_r_drums) * torch.log10(torch.abs(out_r_drums))
                    out_r_vocals=torch.sign(out_r_vocals) * torch.log10(torch.abs(out_r_vocals))
                    out_r_other=torch.sign(out_r_other) * torch.log10(torch.abs(out_r_other))

                    out_l_vo=torch.sign(out_l_vo) * torch.log10(torch.abs(out_l_vo))
                    out_r_vo=torch.sign(out_r_vo) * torch.log10(torch.abs(out_r_vo))
                    
                    # Change NaN and inf to zero
                    out_l_mix[torch.isinf(out_l_mix) | torch.isnan(out_l_mix)] = 0
                    out_l_bass[torch.isinf(out_l_bass) | torch.isnan(out_l_bass)] = 0
                    out_l_drums[torch.isinf(out_l_drums) | torch.isnan(out_l_drums)] = 0
                    out_l_vocals[torch.isinf(out_l_vocals) | torch.isnan(out_l_vocals)] = 0
                    out_l_other[torch.isinf(out_l_other) | torch.isnan(out_l_other)] = 0
                    out_r_mix[torch.isinf(out_r_mix) | torch.isnan(out_r_mix)] = 0
                    out_r_bass[torch.isinf(out_r_bass) | torch.isnan(out_r_bass)] = 0
                    out_r_drums[torch.isinf(out_r_drums) | torch.isnan(out_r_drums)] = 0
                    out_r_vocals[torch.isinf(out_r_vocals) | torch.isnan(out_r_vocals)] = 0
                    out_r_other[torch.isinf(out_r_other) | torch.isnan(out_r_other)] = 0

                    out_l_vo[torch.isinf(out_l_vo) | torch.isnan(out_l_vo)] = 0
                    out_r_vo[torch.isinf(out_r_vo) | torch.isnan(out_r_vo)] = 0                    
                    
                if len(out_l_mix) == chunk_size and len(out_l_bass) == chunk_size and len(out_l_drums) == chunk_size and len(out_l_vocals) == chunk_size and len(out_l_other) == chunk_size:
                    if len(out_r_mix) == chunk_size and len(out_r_bass) == chunk_size and len(out_r_drums) == chunk_size and len(out_r_vocals) == chunk_size and len(out_r_other) == chunk_size:

                # if len(out_l_mix) == chunk_size and len(out_l_bass) == chunk_size and len(out_l_drums) == chunk_size:
                #     if len(out_r_mix) == chunk_size and len(out_r_bass) == chunk_size and len(out_r_drums) == chunk_size:
 
                # if len(out_l_mix) == chunk_size and len(out_l_drums) == chunk_size:
                #     if len(out_r_mix) == chunk_size and len(out_r_drums) == chunk_size:

                        # mix_l_tuple = ((out_l_mix), (out_l_bass, out_l_drums, out_l_vocals, out_l_other))
                        # mix_r_tuple = ((out_r_mix), (out_r_bass, out_r_drums, out_r_vocals, out_r_other))

                        mix_l_tuple = ((out_l_mix), (out_l_bass, out_l_drums, out_l_vo))
                        mix_r_tuple = ((out_r_mix), (out_r_bass, out_r_drums, out_r_vo))

                        # mix_l_tuple = ((out_l_mix), (out_l_drums))
                        # mix_r_tuple = ((out_r_mix), (out_r_drums))
                        
                        if out_l_mix_sum >= mix_l_sum_bins[nearIdx_min_mix_l_sum] and out_l_mix_sum <= mix_l_sum_bins[nearIdx_max_mix_l_sum]: #29032023
                            output_arr.append(mix_l_tuple)
                            # output_arr.append(mix_r_tuple) # commented out for debugging perpose
######################################################################################################
#         # Set the sampling frequency and duration
#         fs = 44100  # Sampling frequency
#         T = 1/fs  # Sampling period
#         duration = 1  # Duration in seconds

#         # Create a time vector
#         t = np.arange(0, duration, T)

#         # Set the frequency of the sine wave
#         f = 1000  # Frequency in Hz

#         # Generate the sine wave array
#         sin_array = np.sin(2*np.pi*f*t)
#         sin_wave = torch.from_numpy(sin_array[:chunk_size])
#         cos_array = np.cos(2*np.pi*f*t)
#         cos_wave = torch.from_numpy(cos_array[:chunk_size])

#         mix_l_tuple = ((cos_wave), (sin_wave, sin_wave))

#         output_arr.append(mix_l_tuple)
######################################################################################################
            
        return tuple(output_arr)
