import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import torchaudio
import signal

import argparse

import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from dataset_cpuDataLoader import readAudio
from torch.optim.lr_scheduler import StepLR

from torch.cuda.amp import autocast, GradScaler

from torch.optim.swa_utils import AveragedModel, SWALR

import subprocess

from scipy.io import wavfile

import time

from functools import partial
import sys

# Record the start time of training
start_time = time.time()
last_time = time.time()  # get the current time

local_rank = 0
device = torch.device('cuda:%d'%local_rank)

parser = argparse.ArgumentParser()
parser.add_argument("--enable_amp", action='store_true', help='amp mix precision use?')
parser.add_argument("--unet_only", action='store_true', help='Train UNet only')
parser.add_argument("--alpha", default=0.001, type=float, help='alpha for audio intensity matching')
parser.add_argument("--num_epoch", default=1000, type=int, help='number of epochs?')
parser.add_argument("--batch_size", default=2, type=int, help='batch_size?')
parser.add_argument("--lr", default=1e-5, type=float, help='learning rate?')
parser.add_argument("--len", default=10, type=int, help='length of the song')
parser.add_argument("--valen", default=10, type=int, help='length of the song for validation')
parser.add_argument("--timlim", default=11, type=int, help='time limit of the run. if -999, we go until the epoch is finished')
parser.add_argument("--num_itr", default=4, type=int, help='how often do you want to update the optimizer? the larger, the less.')
parser.add_argument("--contrain", default=-999, help='Continue training from a saved model. Give epoch value')
parser.add_argument("--hiddench", default=[64, 128, 256, 512, 1024, 2048], type=int, nargs='+', help='hidden channel layout')
parser.add_argument("--numembed", default=128, type=int, help='number of embeddings')
parser.add_argument("--embeddim", default=2048, type=int, help='embedding dimension must match the hidden channel last number')
parser.add_argument("--reconscale", default=1., type=float, help='weight to the recon loss')
parser.add_argument("--l1l2", default="l2", type=str, help='recon loss L1 vs L2')
parser.add_argument("--reconweight", default=[1./1., 1./2., 1./4., 1./4.], type=float, nargs='+', help='weight to the recon loss separately to each channel')
parser.add_argument("--ampscale", default = 1./750., type=float, help='amplitude match scale')
parser.add_argument("--datareamp", default=[-999, -999], type=float, nargs='+', help='wanna change the source amp? give min and max values')
args = parser.parse_args()

#####HYPER PARAMETERS#####
LEARNING_RATE = args.lr
NUM_EPOCH = args.num_epoch
BATCH_SIZE = args.batch_size
HIDDEN_CH = args.hiddench
NUM_EMBED = args.numembed
EMBED_DIM = args.embeddim
##########################

ENABLE_AMP = args.enable_amp

UNET_ONLY = args.unet_only
# SCALES = [1.0, 0.5, 0.25, 0.125]
SCALES = [1.0]
ALPHA = args.alpha
SONGLEN = args.len
VALEN = args.valen
RECONSCALE = 1./4.*10.
RECONSCALE = args.reconscale
L1L2 = args.l1l2
RECONWEIGHT = args.reconweight
SPECTVARSCALE = 2.
SISNRSCALE = 1./200.
AMPSCALE = args.ampscale
TIMLIM = args.timlim
DATARERAMP = args.datareamp
if DATARERAMP[0] > 1 or DATARERAMP[0] < 0 or DATARERAMP[1] > 1 or DATARERAMP[1] < 0:
    print('data reamp values are too lare or too small')
    print(f'[{DATARERAMP[0]}, {DATARERAMP[1]}]')
    print('Pick values between [0, 1] you fuck')
    sys.exit()

# Set the time limit in seconds (11 hours)
if TIMLIM != -999:
    time_limit = TIMLIM * 60 * 60
else:
    time_limit = 1 * 60 * 60

CONTRAIN = args.contrain
NUM_ITR = args.num_itr
CLIPVAL = 1.0
    
print('alpha:', ALPHA)
print('num_epoch:', NUM_EPOCH)
print('lr:', LEARNING_RATE)
print('HIDDEN_CH:', HIDDEN_CH)
print('NUM_EMBED:', NUM_EMBED)
print('EMBED_DIM:', EMBED_DIM)
print('song length:', SONGLEN, 'sec')
print('song validation length:', VALEN, 'sec')
print('reconscale', RECONSCALE)
print('reconweights', RECONWEIGHT)
print('spectvarscale', SPECTVARSCALE)
print('sisnrscale', SISNRSCALE)
print('ampscale', AMPSCALE)
print('time limit', time_limit)
print('number of iteration for the optimizer update', NUM_ITR)
print('clip value', CLIPVAL)

def get_factors(n):
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors

class Resample1D(nn.Module):
    def __init__(self, out_channels, kernel_size, stride, padding='reflect', transpose=False):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.transpose = transpose
        
        cutoff = 0.5 / stride
        
        filter = build_sinc_filter(kernel_size, cutoff)

        self.filter = torch.nn.Parameter(torch.from_numpy(np.repeat(np.reshape(filter, [1, 1, kernel_size]), out_channels, axis=0)), requires_grad=False)
        
    def forward(self, x):
        input_size = x.shape[2]
        num_pad = (self.kernel_size-1)//2
        out = F.pad(x, (num_pad, num_pad), mode=self.padding)
        # Lowpass filter (+ 0 insertion if transposed)
        if self.transpose:
            expected_steps = ((input_size - 1) * self.stride + 1)

            out = F.conv_transpose1d(out, self.filter, stride=self.stride, padding=0, groups=self.out_channels)
            diff_steps = out.shape[2] - expected_steps
            if diff_steps > 0:
                assert(diff_steps % 2 == 0)
                out = out[:,:,diff_steps//2:-diff_steps//2]
        else:
            out = F.conv1d(out, self.filter, stride=self.stride, padding=0, groups=self.out_channels)
        return out        
        
def build_sinc_filter(kernel_size, cutoff):
    # FOLLOWING https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch16.pdf
    # Sinc lowpass filter
    # Build sinc kernel
    assert(kernel_size % 2 == 1)
    M = kernel_size - 1
    filter = np.zeros(kernel_size, dtype=np.float32)
    for i in range(kernel_size):
        if i == M//2:
            filter[i] = 2 * np.pi * cutoff
        else:
            filter[i] = (np.sin(2 * np.pi * cutoff * (i - M//2)) / (i - M//2)) * \
                    (0.42 - 0.5 * np.cos((2 * np.pi * i) / M) + 0.08 * np.cos(4 * np.pi * M))

    filter = filter / np.sum(filter)
    return filter
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, skip_channel, out_channels, kernel_size=5, stride=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1 = nn.Conv1d(in_channels, skip_channel, kernel_size, stride=1, padding=0) #no padding (padding may cause aliasing)
        self.conv2 = nn.Conv1d(skip_channel, out_channels, kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.groupnorm1 = nn.GroupNorm(skip_channel // 8, skip_channel)
        self.groupnorm2 = nn.GroupNorm(out_channels // 8, out_channels)

        self.downsample = Resample1D(out_channels, kernel_size=15, stride=stride) # stride = 4
        
        # He initialization
        init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x):
        skip = self.relu(self.groupnorm1(self.conv1(x))) # making skips
        x1 = self.relu(self.groupnorm2(self.conv2(skip)))
        x2 = self.downsample(x1)

        return x2, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0) #no padding (padding may cause aliasing)
        self.conv1_alt = nn.Conv1d(in_channels, out_channels, kernel_size+1, stride=1, padding=0) # alternative kernel_size
        self.conv2 = nn.Conv1d(out_channels * 2, out_channels, kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.groupnorm1 = nn.GroupNorm(out_channels // 8, out_channels)
        self.groupnorm2 = nn.GroupNorm(out_channels // 8, out_channels)

        self.upsample = Resample1D(in_channels, kernel_size=15, stride=stride, transpose=True) # stride = 4

        init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x, skip):
        x1 = self.upsample(x) 
        #make sure skip and x2 are both even or both odd for even/center cropping
        #by change the even/odd-ness of the kernel_size
        if skip.shape[-1] % 2 == x1.shape[-1] % 2: # both even or odd
            x2 = self.relu(self.groupnorm1(self.conv1(x1))) # zero padding
        else:
            x2 = self.relu(self.groupnorm1(self.conv1_alt(x1))) # kernel_size+1
    
        skip1 = crop(skip, x2) 

        combined = torch.cat([skip1, crop(x2, skip1)], dim=1)
        x3 = self.relu(self.groupnorm2(self.conv2(combined)))

        return x3

def crop(x, target):
    diff = x.shape[-1] - target.shape[-1]
    assert diff % 2 == 0, f'{diff}, {x.shape[-1]}, {target.shape[-1]}'
    
    
    crop = diff // 2
    if crop == 0:
        return x
    if crop < 0:
        raise ArithmeticError

    return x[:, :, crop:-crop].contiguous()        

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.groupnorm1 = nn.GroupNorm(out_channels // 8, out_channels)
    
    def forward(self, x):
        x = self.relu(self.groupnorm1(self.conv1(x)))
        
        return x
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=[64, 128, 256, 512], kernel_size=5, stride=4):
        super().__init__()

        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(DownBlock(in_channels, hidden_channels[0], hidden_channels[1], kernel_size, stride))
        for i in range(1, len(hidden_channels)-1):
            self.down_blocks.append(DownBlock(hidden_channels[i], hidden_channels[i], hidden_channels[i+1], kernel_size, stride))

        self.up_blocks = nn.ModuleList()
        for i in range(len(hidden_channels) - 1, 0, -1):
            self.up_blocks.append(UpBlock(hidden_channels[i], hidden_channels[i - 1], kernel_size, stride))

        self.output_layer = nn.Conv1d(hidden_channels[0], out_channels, kernel_size=1)        

    def forward(self, x, skips=None, encoder=True):
        if encoder:
            skips = []
            for i, block in enumerate(self.down_blocks):
                x, skip = block(x)
                skips.append(skip)
            return x, skips

        else:  # decoder
            for i, block in enumerate(self.up_blocks):
                x = block(x, skips[-(i+1)])

            x = self.output_layer(x)
            
            return x
        
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros(num_embeddings, self.embedding_dim))

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(encodings.sum(dim=0), alpha=1 - self.decay)
            embed_sum = flat_input.t().matmul(encodings)
            embed_sum = embed_sum.transpose(0,1)
            self.ema_w.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            updated_embeddings = self.ema_w / cluster_size.unsqueeze(1)

            self.embeddings.weight.data.copy_(updated_embeddings)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity, encoding_indices.view(-1, input_shape[-1])
        
class VQVAEWithUNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=[64, 128, 256, 512, 1024], num_embeddings=128, embedding_dim=1024, kernel_size=5, stride=4):
        super().__init__()
        self.unet = UNet(in_channels, out_channels, hidden_channels, kernel_size, stride)
        self.bottleneck = Bottleneck(hidden_channels[-1], hidden_channels[-1], kernel_size, stride)
        if UNET_ONLY == False:
            self.vq_layer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost=0.25)

    def forward(self, x):
        x, skips = self.unet(x, encoder=True)
        if UNET_ONLY == False:

            # VQ-VAE bottleneck
            quantized, loss, perplexity, encoding_indices = self.vq_layer(x)
            
            x = self.unet(quantized, skips, encoder=False)  # Pass both z and skips to the decoder
        else:
            quantized = torch.tensor(0., device=device)
            loss = torch.tensor(0., device=device)
            perplexity = torch.tensor(0., device=device)
            encoding_indices = torch.tensor(0., device=device)
            
            x = self.bottleneck(x)
            
            x = self.unet(x, skips, encoder=False)  # Pass both z and skips to the decoder
        return x, loss, perplexity, encoding_indices

    
def create_multi_scale_inputs(input_3d, scales):
    multi_scale_inputs = []
    for scale in scales:
        if scale == 1.:
            resized_input = input_3d
        else:
            resized_input = F.interpolate(input_3d, scale_factor=scale, mode='linear')
        multi_scale_inputs.append(resized_input)
    return multi_scale_inputs    

import numpy as np

def exponential_weight_matrix(freq_bins, alpha=-0.5):
    weights = np.arange(freq_bins)
    weights = np.exp(alpha * weights)
    return torch.tensor(weights, dtype=torch.float32)

def spectrogram_mse_loss(source, mixes, target, window_size=1024, hop_length=512, exp_alpha=-0.5):
    assert source.shape == target.shape, f"Source and target tensors should have the same shape, {source.shape}, {target.shape}"
    batch_size, num_channels, seq_len = source.shape
    
    mix_stft = torch.stft(mixes.squeeze(1), n_fft=window_size, hop_length=hop_length, return_complex=True)
    mix_magnitude = torch.abs(mix_stft) + 1e-8
    mix_magnitude_max = mix_magnitude.max()

    loss = 0
    for i in range(num_channels):
        source_channel = source[:, i, :]
        target_channel = target[:, i, :]

        source_stft = torch.stft(source_channel, n_fft=window_size, hop_length=hop_length, return_complex=True)
        target_stft = torch.stft(target_channel, n_fft=window_size, hop_length=hop_length, return_complex=True)

        source_magnitude = torch.abs(source_stft) + 1e-8
        target_magnitude = torch.abs(target_stft) + 1e-8

        weight_matrix = exponential_weight_matrix(window_size // 2 + 1, alpha=exp_alpha).to(source.device)
        weight_matrix = weight_matrix.view(1,-1,1) # reshape to (1, -, 1) allowing it to be broadcasted
        source_magnitude_weighted = source_magnitude * weight_matrix
        target_magnitude_weighted = target_magnitude * weight_matrix

        loss += nn.MSELoss()(source_magnitude_weighted, target_magnitude_weighted)

    loss /= num_channels
    loss /= (window_size // 2 + 1)

    return loss

def amplitude_matching_loss(sources, mixes, alpha=0.03):
    reconstructed_mixes = sources.sum(dim=1, keepdim=True)
    amplitude_difference = torch.abs(reconstructed_mixes.sum(dim=-1) - mixes.sum(dim=-1))
    return alpha * amplitude_difference.mean()

def multi_resolution_stft_loss_with_amplitude_constraint(sources, mixes, target, window_sizes=[1024, 2048, 4096], hop_lengths=[256, 512, 1024], alpha=0.03):
    total_loss = 0
    for ws, hl in zip(window_sizes, hop_lengths):
        total_loss += spectrogram_mse_loss(sources, mixes, target, window_size=ws, hop_length=hl)
    
    total_loss /= len(list(zip(window_sizes, hop_lengths)))
    
    return total_loss



def si_snr_loss_with_amplitude_constraint(source, target, mix, alpha=0.03, eps=1e-8):
    target = target - target.mean(dim=-1, keepdim=True)
    source = source - source.mean(dim=-1, keepdim=True)
    
    s_target = torch.sum(target * source, dim=-1, keepdim=True) * target / (torch.sum(target**2, dim=-1, keepdim=True) + eps)
    e_noise = source - s_target
    
    si_snr = 10 * torch.log10(torch.sum(s_target**2, dim=-1) / (torch.sum(e_noise**2, dim=-1) + eps) + eps)
    
    # Average the loss across the channels
    loss = -si_snr.mean(dim=1).mean()

    return loss

iteration = 0 # Define a global variable to count the number of iterations
def print_grad(module, grad_input, grad_output):
    global iteration
    for grad_in in grad_input:
        if grad_in is not None:
            print("Gradient input shape:", grad_in.shape)
        else:
            print("Gradient input is None")
    print("Gradient input:", grad_input)

    for grad_out in grad_output:
        if grad_out is not None:
            print("Gradient output shape:", grad_out.shape)
        else:
            print("Gradient output is None")
    print("Gradient output:", grad_output)
    
def compute_stft(signal, n_fft, hop_length):
    stft_complex = torch.stft(signal, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft).to(signal.device), return_complex=True)
    return stft_complex

def compute_istft(complex_spectrogram, n_fft, hop_length):
    signal = torch.istft(complex_spectrogram, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft).to(complex_spectrogram.device), return_complex=False)
    return signal


readaudio = readAudio('/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Mixtures/Dev', '/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Sources/Dev', False, True, device)
if DATARERAMP[0] == -999 and DATARERAMP[1] == -999 :
    audios = readaudio.__getitem__(1, 44100*SONGLEN, rand_amp=False)
else:
    audios = readaudio.__getitem__(1, 44100*SONGLEN, rand_amp=True, min_amp = DATARERAMP[0], max_amp = DATARERAMP[1])

dataloader = DataLoader(audios, batch_size=BATCH_SIZE, shuffle=True, persistent_workers=True, num_workers=8, pin_memory=True)

model_bass = VQVAEWithUNet(1, 1, hidden_channels=HIDDEN_CH, num_embeddings=NUM_EMBED, embedding_dim=EMBED_DIM, kernel_size = 5, stride=4).to(device) # conv kernel_size resample1d stride_size
model_drum = VQVAEWithUNet(1, 1, hidden_channels=HIDDEN_CH, num_embeddings=NUM_EMBED, embedding_dim=EMBED_DIM, kernel_size = 5, stride=4).to(device) # conv kernel_size resample1d stride_size
model_voca = VQVAEWithUNet(1, 1, hidden_channels=HIDDEN_CH, num_embeddings=NUM_EMBED, embedding_dim=EMBED_DIM, kernel_size = 5, stride=4).to(device) # conv kernel_size resample1d stride_size
model_othr = VQVAEWithUNet(1, 1, hidden_channels=HIDDEN_CH, num_embeddings=NUM_EMBED, embedding_dim=EMBED_DIM, kernel_size = 5, stride=4).to(device) # conv kernel_size resample1d stride_size
if CONTRAIN != -999 :
    saved_model_bass = f'/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/model_bass_vq-vae-spectro_lr_1e-05_epoch_{CONTRAIN}_songlen_{SONGLEN}.pth'
    saved_model_drum = f'/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/model_drum_vq-vae-spectro_lr_1e-05_epoch_{CONTRAIN}_songlen_{SONGLEN}.pth'
    saved_model_voca = f'/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/model_voca_vq-vae-spectro_lr_1e-05_epoch_{CONTRAIN}_songlen_{SONGLEN}.pth'
    saved_model_othr = f'/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/model_othr_vq-vae-spectro_lr_1e-05_epoch_{CONTRAIN}_songlen_{SONGLEN}.pth'
    print('continue traing from', saved_model_bass)
    print('continue traing from', saved_model_drum)
    print('continue traing from', saved_model_voca)
    print('continue traing from', saved_model_othr)
    model_bass.load_state_dict(torch.load(saved_model_bass))
    model_drum.load_state_dict(torch.load(saved_model_drum))
    model_voca.load_state_dict(torch.load(saved_model_voca))
    model_othr.load_state_dict(torch.load(saved_model_othr))
    model_bass.train()
    model_drum.train()
    model_voca.train()
    model_othr.train()
optimizer_bass = torch.optim.Adam(model_bass.parameters(), lr=LEARNING_RATE)
optimizer_drum = torch.optim.Adam(model_drum.parameters(), lr=LEARNING_RATE)
optimizer_voca = torch.optim.Adam(model_voca.parameters(), lr=LEARNING_RATE)
optimizer_othr = torch.optim.Adam(model_othr.parameters(), lr=LEARNING_RATE)

# Define the loss function you want to use
recon_loss_function = None
if L1L2 == 'l2':
    recon_loss_function = partial(F.mse_loss) 
    print('recon loss function is L2')
elif L1L2 == 'l1':
    recon_loss_function = partial(F.l1_loss) 
    print('recon loss function is L1')
else: 
    print('WTF is your recon loss function?. Exiting')
    sys.exit()

scaler_bass = None
scaler_drum = None
scaler_voca = None
scaler_othr = None
if(ENABLE_AMP):
    scaler_bass = GradScaler()
    scaler_drum = GradScaler()
    scaler_voca = GradScaler()
    scaler_othr = GradScaler()

# Create the SWA model
swa_model_bass = AveragedModel(model_bass)
swa_model_drum = AveragedModel(model_drum)
swa_model_voca = AveragedModel(model_voca)
swa_model_othr = AveragedModel(model_othr)

if CONTRAIN == -999: epoch = 0
else: epoch = CONTRAIN

for epoch in range(NUM_EPOCH):
# for epoch in range(1):
    print(epoch)

    num_iterations = len(dataloader)
    iteration_count = 0
    optimizer_bass.zero_grad()
    optimizer_drum.zero_grad()
    optimizer_voca.zero_grad()
    optimizer_othr.zero_grad()

    for i, data in enumerate(dataloader):
        mixes_uncut = data[0].to(device)
        sources_uncut = list(map(lambda x: x.to(device), data[1]))

        mixes_uncut

        mixes = mixes_uncut.unsqueeze(1).float() # make dimension from (batch size, seq len) to (batch size, num of input, seq len)

        bass = sources_uncut[0].unsqueeze(1)
        drum = sources_uncut[1].unsqueeze(1)
        voca = sources_uncut[2].unsqueeze(1)
        othr = sources_uncut[3].unsqueeze(1)
        
        multi_scale_inputs = create_multi_scale_inputs(mixes, SCALES)
        multi_scale_tar_bass = create_multi_scale_inputs(bass, SCALES)
        multi_scale_tar_drum = create_multi_scale_inputs(drum, SCALES)
        multi_scale_tar_voca = create_multi_scale_inputs(voca, SCALES)
        multi_scale_tar_othr = create_multi_scale_inputs(othr, SCALES)

        loss_bass = torch.tensor(0., device=device)

        recon_loss_scale_bass = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        spectvar_loss_scale_bass = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        sisnr_loss_scale_bass = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        ampmatch_loss_scale_bass = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        commit_loss_scale_bass = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        perplexity_scale_bass = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]

        recon_loss_sum_bass = torch.tensor(0., device=device)
        recon_loss_list_sum_bass = torch.tensor(0., device=device)
        spectvar_loss_sum_bass = torch.tensor(0., device=device)
        sisnr_loss_sum_bass = torch.tensor(0., device=device)
        ampmatch_loss_sum_bass = torch.tensor(0., device=device)    
        commit_loss_sum_bass = torch.tensor(0., device=device)   
        perplexity_sum_bass = torch.tensor(0., device=device)

        loss_drum = torch.tensor(0., device=device)

        recon_loss_scale_drum = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        spectvar_loss_scale_drum = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        sisnr_loss_scale_drum = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        ampmatch_loss_scale_drum = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        commit_loss_scale_drum = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        perplexity_scale_drum = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]

        recon_loss_sum_drum = torch.tensor(0., device=device)
        recon_loss_list_sum_drum = torch.tensor(0., device=device)
        spectvar_loss_sum_drum = torch.tensor(0., device=device)
        sisnr_loss_sum_drum = torch.tensor(0., device=device)
        ampmatch_loss_sum_drum = torch.tensor(0., device=device)    
        commit_loss_sum_drum = torch.tensor(0., device=device)   
        perplexity_sum_drum = torch.tensor(0., device=device)

        loss_voca = torch.tensor(0., device=device)

        recon_loss_scale_voca = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        spectvar_loss_scale_voca = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        sisnr_loss_scale_voca = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        ampmatch_loss_scale_voca = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        commit_loss_scale_voca = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        perplexity_scale_voca = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]

        recon_loss_sum_voca = torch.tensor(0., device=device)
        recon_loss_list_sum_voca = torch.tensor(0., device=device)
        spectvar_loss_sum_voca = torch.tensor(0., device=device)
        sisnr_loss_sum_voca = torch.tensor(0., device=device)
        ampmatch_loss_sum_voca = torch.tensor(0., device=device)    
        commit_loss_sum_voca = torch.tensor(0., device=device)   
        perplexity_sum_voca = torch.tensor(0., device=device)

        loss_othr = torch.tensor(0., device=device)

        recon_loss_scale_othr = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        spectvar_loss_scale_othr = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        sisnr_loss_scale_othr = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        ampmatch_loss_scale_othr = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        commit_loss_scale_othr = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        perplexity_scale_othr = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]

        recon_loss_sum_othr = torch.tensor(0., device=device)
        recon_loss_list_sum_othr = torch.tensor(0., device=device)
        spectvar_loss_sum_othr = torch.tensor(0., device=device)
        sisnr_loss_sum_othr = torch.tensor(0., device=device)
        ampmatch_loss_sum_othr = torch.tensor(0., device=device)    
        commit_loss_sum_othr = torch.tensor(0., device=device)   
        perplexity_sum_othr = torch.tensor(0., device=device)

        optimizer_bass.zero_grad()
        optimizer_drum.zero_grad()
        optimizer_voca.zero_grad()
        optimizer_othr.zero_grad()

        with autocast(ENABLE_AMP):
            for i_scale in range(len(multi_scale_inputs)):

                output_bass, commit_loss_bass, perplexity_bass, encoding_indices_bass = model_bass(multi_scale_inputs[i_scale])
                output_drum, commit_loss_drum, perplexity_drum, encoding_indices_drum = model_drum(multi_scale_inputs[i_scale])
                output_voca, commit_loss_voca, perplexity_voca, encoding_indices_voca = model_voca(multi_scale_inputs[i_scale])
                output_othr, commit_loss_othr, perplexity_othr, encoding_indices_othr = model_othr(multi_scale_inputs[i_scale])
                assert (multi_scale_inputs[i_scale].shape[-1] - output_bass.shape[-1])/2 != 0, f'mixes.shape[-1] ({multi_scale_inputs[i_scale].shape[-1]}) - output_bass.shape[-1] ({output.shape[-1]}) )/2 != 0'
                target_crop_size = int((multi_scale_inputs[i_scale].shape[-1] - output_bass.shape[-1])/2)
                multi_scale_tar_bass[i_scale] = multi_scale_tar_bass[i_scale][:,:,target_crop_size:-target_crop_size]
                multi_scale_tar_drum[i_scale] = multi_scale_tar_drum[i_scale][:,:,target_crop_size:-target_crop_size]
                multi_scale_tar_voca[i_scale] = multi_scale_tar_voca[i_scale][:,:,target_crop_size:-target_crop_size]
                multi_scale_tar_othr[i_scale] = multi_scale_tar_othr[i_scale][:,:,target_crop_size:-target_crop_size]
                
                recon_loss_bass = RECONSCALE * recon_loss_function(output_bass, multi_scale_tar_bass[i_scale])
                recon_loss_list_bass = [RECONSCALE * recon_loss_function(output_bass[:, i], multi_scale_tar_bass[i_scale][:, i]) for i in range(output_bass.shape[1])]
                spectvar_loss_bass = SPECTVARSCALE * multi_resolution_stft_loss_with_amplitude_constraint(output_bass, multi_scale_inputs[i_scale], multi_scale_tar_bass[i_scale])
                sisnr_loss_bass = SISNRSCALE * si_snr_loss_with_amplitude_constraint(output_bass,  multi_scale_tar_bass[i_scale], multi_scale_inputs[i_scale])
                ampmatch_loss_bass = AMPSCALE * amplitude_matching_loss(output_bass, multi_scale_inputs[i_scale], alpha=ALPHA) / SCALES[i_scale]

                recon_loss_scale_bass[i_scale] = recon_loss_bass
                spectvar_loss_scale_bass[i_scale] = spectvar_loss_bass
                sisnr_loss_scale_bass[i_scale] = sisnr_loss_bass
                ampmatch_loss_scale_bass[i_scale] = ampmatch_loss_bass
                commit_loss_scale_bass[i_scale] = commit_loss_bass
                perplexity_scale_bass[i_scale] = perplexity_bass

                recon_loss_sum_bass += recon_loss_bass
                recon_loss_list_sum_bass = sum(v * w for v, w in zip(recon_loss_list_bass, RECONWEIGHT))
                spectvar_loss_sum_bass += spectvar_loss_bass
                sisnr_loss_sum_bass += sisnr_loss_bass
                ampmatch_loss_sum_bass += ampmatch_loss_bass
                commit_loss_sum_bass += commit_loss_bass
                perplexity_sum_bass += perplexity_bass

                
                
                recon_loss_drum = RECONSCALE * recon_loss_function(output_drum, multi_scale_tar_drum[i_scale])
                recon_loss_list_drum = [RECONSCALE * recon_loss_function(output_drum[:, i], multi_scale_tar_drum[i_scale][:, i]) for i in range(output_drum.shape[1])]
                spectvar_loss_drum = SPECTVARSCALE * multi_resolution_stft_loss_with_amplitude_constraint(output_drum, multi_scale_inputs[i_scale], multi_scale_tar_drum[i_scale])
                sisnr_loss_drum = SISNRSCALE * si_snr_loss_with_amplitude_constraint(output_drum,  multi_scale_tar_drum[i_scale], multi_scale_inputs[i_scale])
                ampmatch_loss_drum = AMPSCALE * amplitude_matching_loss(output_drum, multi_scale_inputs[i_scale], alpha=ALPHA) / SCALES[i_scale]

                recon_loss_scale_drum[i_scale] = recon_loss_drum
                spectvar_loss_scale_drum[i_scale] = spectvar_loss_drum
                sisnr_loss_scale_drum[i_scale] = sisnr_loss_drum
                ampmatch_loss_scale_drum[i_scale] = ampmatch_loss_drum
                commit_loss_scale_drum[i_scale] = commit_loss_drum
                perplexity_scale_drum[i_scale] = perplexity_drum

                recon_loss_sum_drum += recon_loss_drum
                recon_loss_list_sum_drum = sum(v * w for v, w in zip(recon_loss_list_drum, RECONWEIGHT))
                spectvar_loss_sum_drum += spectvar_loss_drum
                sisnr_loss_sum_drum += sisnr_loss_drum
                ampmatch_loss_sum_drum += ampmatch_loss_drum
                commit_loss_sum_drum += commit_loss_drum
                perplexity_sum_drum += perplexity_drum

                
                
                recon_loss_voca = RECONSCALE * recon_loss_function(output_voca, multi_scale_tar_voca[i_scale])
                recon_loss_list_voca = [RECONSCALE * recon_loss_function(output_voca[:, i], multi_scale_tar_voca[i_scale][:, i]) for i in range(output_voca.shape[1])]
                spectvar_loss_voca = SPECTVARSCALE * multi_resolution_stft_loss_with_amplitude_constraint(output_voca, multi_scale_inputs[i_scale], multi_scale_tar_voca[i_scale])
                sisnr_loss_voca = SISNRSCALE * si_snr_loss_with_amplitude_constraint(output_voca,  multi_scale_tar_voca[i_scale], multi_scale_inputs[i_scale])
                ampmatch_loss_voca = AMPSCALE * amplitude_matching_loss(output_voca, multi_scale_inputs[i_scale], alpha=ALPHA) / SCALES[i_scale]

                recon_loss_scale_voca[i_scale] = recon_loss_voca
                spectvar_loss_scale_voca[i_scale] = spectvar_loss_voca
                sisnr_loss_scale_voca[i_scale] = sisnr_loss_voca
                ampmatch_loss_scale_voca[i_scale] = ampmatch_loss_voca
                commit_loss_scale_voca[i_scale] = commit_loss_voca
                perplexity_scale_voca[i_scale] = perplexity_voca

                recon_loss_sum_voca += recon_loss_voca
                recon_loss_list_sum_voca = sum(v * w for v, w in zip(recon_loss_list_voca, RECONWEIGHT))
                spectvar_loss_sum_voca += spectvar_loss_voca
                sisnr_loss_sum_voca += sisnr_loss_voca
                ampmatch_loss_sum_voca += ampmatch_loss_voca
                commit_loss_sum_voca += commit_loss_voca
                perplexity_sum_voca += perplexity_voca

                
                
                recon_loss_othr = RECONSCALE * recon_loss_function(output_othr, multi_scale_tar_othr[i_scale])
                recon_loss_list_othr = [RECONSCALE * recon_loss_function(output_othr[:, i], multi_scale_tar_othr[i_scale][:, i]) for i in range(output_othr.shape[1])]
                spectvar_loss_othr = SPECTVARSCALE * multi_resolution_stft_loss_with_amplitude_constraint(output_othr, multi_scale_inputs[i_scale], multi_scale_tar_othr[i_scale])
                sisnr_loss_othr = SISNRSCALE * si_snr_loss_with_amplitude_constraint(output_othr,  multi_scale_tar_othr[i_scale], multi_scale_inputs[i_scale])
                ampmatch_loss_othr = AMPSCALE * amplitude_matching_loss(output_othr, multi_scale_inputs[i_scale], alpha=ALPHA) / SCALES[i_scale]

                recon_loss_scale_othr[i_scale] = recon_loss_othr
                spectvar_loss_scale_othr[i_scale] = spectvar_loss_othr
                sisnr_loss_scale_othr[i_scale] = sisnr_loss_othr
                ampmatch_loss_scale_othr[i_scale] = ampmatch_loss_othr
                commit_loss_scale_othr[i_scale] = commit_loss_othr
                perplexity_scale_othr[i_scale] = perplexity_othr

                recon_loss_sum_othr += recon_loss_othr
                recon_loss_list_sum_othr = sum(v * w for v, w in zip(recon_loss_list_othr, RECONWEIGHT))
                spectvar_loss_sum_othr += spectvar_loss_othr
                sisnr_loss_sum_othr += sisnr_loss_othr
                ampmatch_loss_sum_othr += ampmatch_loss_othr
                commit_loss_sum_othr += commit_loss_othr
                perplexity_sum_othr += perplexity_othr

                
                
                
        # loss_bass = recon_loss_sum_bass + commit_loss_sum_bass
        loss_bass = recon_loss_sum_bass + ampmatch_loss_sum_bass + commit_loss_sum_bass
        # loss_drum = recon_loss_sum_drum + commit_loss_sum_drum
        loss_drum = recon_loss_sum_drum + ampmatch_loss_sum_drum + commit_loss_sum_drum
        # loss_voca = recon_loss_sum_voca + commit_loss_sum_voca
        loss_voca = recon_loss_sum_voca + ampmatch_loss_sum_voca + commit_loss_sum_voca
        # loss_othr = recon_loss_sum_othr + commit_loss_sum_othr
        loss_othr = recon_loss_sum_othr + ampmatch_loss_sum_othr + commit_loss_sum_othr
        

        if(ENABLE_AMP):
            scaler_bass.scale(loss_bass / num_iterations).backward()
            scaler_drum.scale(loss_drum / num_iterations).backward()
            scaler_voca.scale(loss_voca / num_iterations).backward()
            scaler_othr.scale(loss_othr / num_iterations).backward()
        else:
            loss_bass.backward()
            loss_drum.backward()
            loss_voca.backward()
            loss_othr.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model_bass.parameters(), CLIPVAL)
        torch.nn.utils.clip_grad_norm_(model_drum.parameters(), CLIPVAL)
        torch.nn.utils.clip_grad_norm_(model_voca.parameters(), CLIPVAL)
        torch.nn.utils.clip_grad_norm_(model_othr.parameters(), CLIPVAL)

        iteration += 1 # Update the iteration count
        iteration_count += 1

        if iteration_count == num_iterations:
            if ENABLE_AMP:
                scaler_bass.step(optimizer_bass)
                scaler_bass.update()
                scaler_drum.step(optimizer_drum)
                scaler_drum.update()
                scaler_voca.step(optimizer_voca)
                scaler_voca.update()
                scaler_othr.step(optimizer_othr)
                scaler_othr.update()
            else:
                optimizer_bass.step()
                optimizer_drum.step()
                optimizer_voca.step()
                optimizer_othr.step()

            # Update the SWA model
            if epoch >= 50:  # Start updating the SWA model after 50 epochs
                swa_model_bass.update_parameters(model_bass)
                swa_model_drum.update_parameters(model_drum)
                swa_model_voca.update_parameters(model_voca)
                swa_model_othr.update_parameters(model_othr)
                
        elapsed_time = time.time() - last_time
        if elapsed_time >= time_limit or epoch % 100 == 0:
            # Save the model
            torch.save(model_bass.state_dict(), f"/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/model_bass_vq-vae-spectro_lr_{LEARNING_RATE}_epoch_{epoch}_songlen_{SONGLEN}.pth")
            torch.save(model_drum.state_dict(), f"/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/model_drum_vq-vae-spectro_lr_{LEARNING_RATE}_epoch_{epoch}_songlen_{SONGLEN}.pth")
            torch.save(model_voca.state_dict(), f"/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/model_voca_vq-vae-spectro_lr_{LEARNING_RATE}_epoch_{epoch}_songlen_{SONGLEN}.pth")
            torch.save(model_othr.state_dict(), f"/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/model_othr_vq-vae-spectro_lr_{LEARNING_RATE}_epoch_{epoch}_songlen_{SONGLEN}.pth")
            print("Model saved at {:.2f} hours.".format(elapsed_time / 3600))
            last_time = time.time()  # update the last_time variable
            if TIMLIM != -999:
                break

        with torch.no_grad():
            if epoch % 100 == 0:
                if i == 0 :
                    output_bass, _, _, _ = model_bass(multi_scale_inputs[0])
                    output_drum, _, _, _ = model_drum(multi_scale_inputs[0])
                    output_voca, _, _, _ = model_voca(multi_scale_inputs[0])
                    output_othr, _, _, _ = model_othr(multi_scale_inputs[0])

                    plt.figure(figsize=(10, 10))
                    plt.subplot(2, 2, 1)          
                    plt.plot(multi_scale_inputs[i_scale][1,0,:].detach().cpu().numpy(), label='mix', color='gray', alpha=0.3)
                    plt.plot(multi_scale_tar_bass[i_scale][1,0,:].detach().cpu().numpy(), label='bass target', color = 'magenta', alpha=0.3)
                    plt.plot(output_bass[1,0,:].detach().cpu().numpy(), label='bass train', color = 'red', alpha=0.3)
                    plt.legend(loc='upper right')
                    plt.subplot(2, 2, 2)          
                    plt.plot(multi_scale_inputs[i_scale][1,0,:].detach().cpu().numpy(), label='mix', color='gray', alpha=0.3)
                    plt.plot(multi_scale_tar_drum[i_scale][1,0,:].detach().cpu().numpy(), label='drum target', color = 'magenta', alpha=0.3)
                    plt.plot(output_drum[1,0,:].detach().cpu().numpy(), label='drum train', color = 'red', alpha=0.3)
                    plt.legend(loc='upper right')
                    plt.subplot(2, 2, 3)          
                    plt.plot(multi_scale_inputs[i_scale][1,0,:].detach().cpu().numpy(), label='mix', color='gray', alpha=0.3)
                    plt.plot(multi_scale_tar_voca[i_scale][1,0,:].detach().cpu().numpy(), label='vocal target', color = 'magenta', alpha=0.3)
                    plt.plot(output_voca[1,0,:].detach().cpu().numpy(), label='vocal train', color = 'red', alpha=0.3)
                    plt.legend(loc='upper right')
                    plt.subplot(2, 2, 4)          
                    plt.plot(multi_scale_inputs[i_scale][1,0,:].detach().cpu().numpy(), label='mix', color='gray', alpha=0.3)
                    plt.plot(multi_scale_tar_othr[i_scale][1,0,:].detach().cpu().numpy(), label='other target', color = 'magenta', alpha=0.3)
                    plt.plot(output_othr[1,0,:].detach().cpu().numpy(), label='other train', color = 'red', alpha=0.3)
                    plt.legend(loc='upper right')
                    plt.savefig(f'/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/epoch_{epoch}_{LEARNING_RATE}_songlen_{SONGLEN}.png')
                    plt.clf()
                    plt.close()                
                    wavfile.write(f'/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/mix_{i}_scale_{i_scale}_lr_{LEARNING_RATE}_epoch_{epoch}.wav', 16000, multi_scale_inputs[i_scale][i,0,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/tar_bas_{i}_scale_{i_scale}_lr_{LEARNING_RATE}_epoch_{epoch}.wav', 16000, multi_scale_tar_bass[i_scale][i,0,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/out_bas_{i}_scale_{i_scale}_lr_{LEARNING_RATE}_epoch_{epoch}.wav', 16000, output_bass[i,0,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/tar_drm_{i}_scale_{i_scale}_lr_{LEARNING_RATE}_epoch_{epoch}.wav', 16000, multi_scale_tar_drum[i_scale][i,0,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/out_drm_{i}_scale_{i_scale}_lr_{LEARNING_RATE}_epoch_{epoch}.wav', 16000, output_drum[i,0,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/tar_voc_{i}_scale_{i_scale}_lr_{LEARNING_RATE}_epoch_{epoch}.wav', 16000, multi_scale_tar_voca[i_scale][i,0,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/out_voc_{i}_scale_{i_scale}_lr_{LEARNING_RATE}_epoch_{epoch}.wav', 16000, output_voca[i,0,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/tar_oth_{i}_scale_{i_scale}_lr_{LEARNING_RATE}_epoch_{epoch}.wav', 16000, multi_scale_tar_othr[i_scale][i,0,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/out_oth_{i}_scale_{i_scale}_lr_{LEARNING_RATE}_epoch_{epoch}.wav', 16000, output_othr[i,0,:].detach().cpu().numpy().ravel())
                
# Save the model to disk
torch.save(model_bass.state_dict(), f"/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/model_bass_vq-vae-spectro_lr_{LEARNING_RATE}_epoch_{NUM_EPOCH}_songlen_{SONGLEN}.pth")
torch.save(model_drum.state_dict(), f"/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/model_drum_vq-vae-spectro_lr_{LEARNING_RATE}_epoch_{NUM_EPOCH}_songlen_{SONGLEN}.pth")
torch.save(model_voca.state_dict(), f"/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/model_voca_vq-vae-spectro_lr_{LEARNING_RATE}_epoch_{NUM_EPOCH}_songlen_{SONGLEN}.pth")
torch.save(model_othr.state_dict(), f"/pscratch/sd/h/hsko/jupyter/output/waveunet/output_novqvae_evencrop/model_othr_vq-vae-spectro_lr_{LEARNING_RATE}_epoch_{NUM_EPOCH}_songlen_{SONGLEN}.pth")

