import os
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import torch

class readAudio():
    def __init__(self, mix_dir, src_dir, device):
        self.mix_dir = mix_dir
        self.src_dir = src_dir
        self.mixture = os.listdir(mix_dir)
        self.device = device
        
    def __len__(self):
        return len(self.mixture)
    
    def __getitem__(self, num_begin, num_sets, chunk_size):
 
        output_arr = []
        max_arr = []
        
        # index = num_sets
        for index in range(num_begin, num_begin+num_sets):
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
            
            # print(len(out_l_mix))
            for start in range(0, len(out_l_mix_uncut), chunk_size):
                end = start + chunk_size
                out_l_mix = torch.from_numpy(out_l_mix_uncut[start:end]).to(self.device)
                out_l_bass = torch.from_numpy(out_l_bass_uncut[start:end]).to(self.device)
                out_l_drums = torch.from_numpy(out_l_drums_uncut[start:end]).to(self.device)
                out_l_vocals = torch.from_numpy(out_l_vocals_uncut[start:end]).to(self.device)
                out_l_other = torch.from_numpy(out_l_other_uncut[start:end]).to(self.device)
                out_r_mix = torch.from_numpy(out_r_mix_uncut[start:end]).to(self.device)
                out_r_bass = torch.from_numpy(out_r_bass_uncut[start:end]).to(self.device)
                out_r_drums = torch.from_numpy(out_r_drums_uncut[start:end]).to(self.device)
                out_r_vocals = torch.from_numpy(out_r_vocals_uncut[start:end]).to(self.device)
                out_r_other = torch.from_numpy(out_r_other_uncut[start:end]).to(self.device)

#                 out_l_mix = out_l_mix_uncut[start:end]
#                 out_l_bass = out_l_bass_uncut[start:end]
#                 out_l_drums = out_l_drums_uncut[start:end]
#                 out_l_vocals = out_l_vocals_uncut[start:end]
#                 out_l_other = out_l_other_uncut[start:end]
#                 out_r_mix = out_r_mix_uncut[start:end]
#                 out_r_bass = out_r_bass_uncut[start:end]
#                 out_r_drums = out_r_drums_uncut[start:end]
#                 out_r_vocals = out_r_vocals_uncut[start:end]
#                 out_r_other = out_r_other_uncut[start:end]
                
                # mix_l_tuple = ((out_l_mix), (out_l_bass, out_l_drums, out_l_vocals, out_l_other))
                # mix_r_tuple = ((out_r_mix), (out_r_bass, out_r_drums, out_r_vocals, out_r_other))

                mix_l_tuple = ((out_l_mix), (out_l_bass, out_l_drums, out_l_vocals, out_l_other))
                mix_r_tuple = ((out_r_mix), (out_r_bass, out_r_drums, out_r_vocals, out_r_other))
            
                output_arr.append(mix_l_tuple)
                output_arr.append(mix_r_tuple)
            
        return tuple(output_arr)
