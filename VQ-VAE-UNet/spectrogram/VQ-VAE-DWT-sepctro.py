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

import wandb
import subprocess

from scipy.io import wavfile

import time

# Record the start time of training
start_time = time.time()

local_rank = 0
device = torch.device('cuda:%d'%local_rank)

parser = argparse.ArgumentParser()
parser.add_argument("--enable_wandb", action='store_true', help='WandB use?')
parser.add_argument("--enable_amp", action='store_true', help='amp mix precision use?')
# parser.add_argument("--train_unet_only", action='store_true', help='Wanna train UNet only?')
# parser.add_argument("--alpha", default=(1./500./1600.)*20., type=float, help='alpha for audio intensity matching')
parser.add_argument("--alpha", default=0.001, type=float, help='alpha for audio intensity matching')
parser.add_argument("--num_epoch", default=1000, type=int, help='number of epochs?')
parser.add_argument("--batch_size", default=2, type=int, help='batch_size?')
parser.add_argument("--lr", default=1e-5, type=float, help='learning rate?')
parser.add_argument("--len", default=10, type=int, help='length of the song')
parser.add_argument("--valen", default=10, type=int, help='length of the song for validation')
parser.add_argument("--timlim", default=11, type=int, help='time limit of the run. if -999, we go until the epoch is finished')
args = parser.parse_args()

ENABLE_WANDB = args.enable_wandb
#####HYPER PARAMETERS#####
# LEARNING_RATE = 1e-4
LEARNING_RATE = args.lr
NUM_EPOCH = args.num_epoch
BATCH_SIZE = args.batch_size
##########################

ENABLE_AMP = args.enable_amp

# SCALES = [1.0, 0.5, 0.25, 0.125]
SCALES = [1.0]
# ALPHA = 1./500./1600.
ALPHA = args.alpha
SONGLEN = args.len
VALEN = args.valen
RECONSCALE = 1./4.
# SPECTVARSCALE = 1./10.
SPECTVARSCALE = 1.*20.
SISNRSCALE = 1./100.
TIMLIM = args.timlim

# Set the time limit in seconds (11 hours)
time_limit = TIMLIM * 60 * 60

print('alpha:', ALPHA)
print('num_epoch:', NUM_EPOCH)
print('lr:', LEARNING_RATE)
print('song length:', SONGLEN, 'sec')
print('song validation length:', VALEN, 'sec')
print('reconscale', RECONSCALE)
print('spectvarscale', SPECTVARSCALE)
print('sisnrscale', SISNRSCALE)
print('time limit', TIMLIM)

# wandb.login(key='7a7346b9e3ee9dfebc6fc65da44ef3644f03298a')
api_key='7a7346b9e3ee9dfebc6fc65da44ef3644f03298a'
subprocess.call(f'wandb login {api_key}', shell=True)
if ENABLE_WANDB:
    wandb.init(project='VQ-VAE-spectro', mode='online')
else:
    wandb.init(project='VQ-VAE-spectro', mode='offline')
    
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))

        # For Xavier initialization
        # Xavier initialization, also known as Glorot initialization, 
        # is designed for activation functions like sigmoid and tanh. 
        # It initializes the weights by drawing samples from 
        # a uniform distribution with the following bounds:
        # sqrt(6) / sqrt(n_in + n_out)
        # where n_in and n_out are the number of input and output units for the layer.
        # init.xavier_uniform_(self.conv1.weight)
        # init.xavier_uniform_(self.conv2.weight)

        # For He initialization
        # He initialization, also known as Kaiming initialization, 
        # is designed for ReLU and its variants. 
        # It initializes the weights by drawing samples from 
        # a normal distribution with mean 0 and standard deviation:
        # sqrt(2 / n_in)
        # where n_in is the number of input units for the layer.
        init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        if self.downsample:
            return self.max_pool(x), x
        else:
            return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=[64, 128, 256, 512]):
        super().__init__()
        self.down_blocks = nn.ModuleList([UNetBlock(in_channels, hidden_channels[0])])
        self.down_blocks.extend([UNetBlock(hidden_channels[i - 1], hidden_channels[i]) for i in range(1, len(hidden_channels))])

        self.up_blocks = nn.ModuleList()
        for i in range(len(hidden_channels) - 1, 0, -1):
            self.up_blocks.append(UNetBlock(hidden_channels[i] * 2, hidden_channels[i - 1], downsample=False))
        self.up_blocks.append(UNetBlock(hidden_channels[0] * 2, out_channels, downsample=False))

    def forward(self, x, skips=None, encoder=True):
        if encoder:
            skips = []
            for block in self.down_blocks:
                x, skip = block(x)
                skips.append(skip)
            return x, skips

        else:  # decoder
            for i, block in enumerate(self.up_blocks):
                x = F.interpolate(x, scale_factor=(1, 2), mode='bilinear', align_corners=False)
                
                if x.size(3) < skips[-(i+1)].size(3):
                    pad = (0, 1)  # pad the last dimension (time_frames) by 1 on the right side
                    x = F.pad(x, pad)
                elif x.size(3) > skips[-(i+1)].size(3):
                    pad = (0, 1)  # pad the last dimension (time_frames) by 1 on the right side
                    skips[-(i+1)] = F.pad(skips[-(i+1)], pad)

                if skips and i < len(skips):
                    # print(f"X shape: {x.shape}, Skips shape: {skips[-(i+1)].shape}")
                    x = torch.cat([x, skips[-(i+1)]], dim=1) 

                x = block(x)

        
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
            # print('embed_sum.shape', embed_sum.shape)
            embed_sum = embed_sum.transpose(0,1)
            # print('embed_sum transposed shape', embed_sum.shape)
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
    def __init__(self, in_channels, out_channels, hidden_channels=[64, 128, 256, 512], num_embeddings=64, embedding_dim=512):
        super().__init__()
        self.unet = UNet(in_channels, out_channels, hidden_channels)
        self.vq_layer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost=0.25)

    def forward(self, x):
        x, skips = self.unet(x, encoder=True)
        quantized, loss, perplexity, encoding_indices = self.vq_layer(x)
        x = self.unet(quantized, skips, encoder=False)  # Pass both z and skips to the decoder
        return x, loss, perplexity, encoding_indices

    
def create_multi_scale_inputs(input_3d, scales):
    multi_scale_inputs = []
    for scale in scales:
        # resized_input = F.interpolate(input_3d, scale_factor=(1, scale), mode='nearest')
        if scales == 1.:
            resized_input = input_3d
        else:
            resized_input = F.interpolate(input_3d, scale_factor=scale, mode='linear')
        multi_scale_inputs.append(resized_input)
    return multi_scale_inputs    

def spectrogram_mse_loss(source, mixes, target, window_size=1024, hop_length=512):
    assert source.shape == target.shape, f"Source and target tensors should have the same shape, {source.shape}, {target.shape}"
    batch_size, num_channels, seq_len = source.shape
    
    mix_stft = torch.stft(mixes.squeeze(1), n_fft=window_size, hop_length=hop_length, return_complex=True)
    mix_magnitude = torch.abs(mix_stft) + 1e-8
    mix_magnitude_max = mix_magnitude.max()
    # print('mix_magnitude shape:', mix_magnitude.shape)

    loss = 0
    for i in range(num_channels):
        source_channel = source[:, i, :]
        target_channel = target[:, i, :]

        source_stft = torch.stft(source_channel, n_fft=window_size, hop_length=hop_length, return_complex=True)
        target_stft = torch.stft(target_channel, n_fft=window_size, hop_length=hop_length, return_complex=True)

        source_magnitude = torch.abs(source_stft) + 1e-8
        target_magnitude = torch.abs(target_stft) + 1e-8

        # # Normalize the magnitude spectrograms using mix_magnitude_max
        # source_magnitude /= mix_magnitude_max
        # target_magnitude /= mix_magnitude_max

        loss += nn.MSELoss()(source_magnitude, target_magnitude)

    # Average the loss across the channels
    loss /= num_channels
    # Divide the loss by the bumber of dimensions in the spectrogram
    loss /= (window_size // 2 + 1)

    return loss

def amplitude_matching_loss(sources, mixes, alpha=0.03):
    reconstructed_mixes = sources.sum(dim=1, keepdim=True)
    amplitude_difference = torch.abs(reconstructed_mixes.sum(dim=-1) - mixes.sum(dim=-1))
    return alpha * amplitude_difference.mean()

def multi_resolution_stft_loss_with_amplitude_constraint(sources, mixes, target, window_sizes=[1024, 2048, 4096], hop_lengths=[256, 512, 1024], alpha=0.03):
    total_loss = 0
    for ws, hl in zip(window_sizes, hop_lengths):
        # for src in range(sources.size(1)):
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

    # Add the amplitude constraint term
    # amplitude_difference = torch.abs(source.sum(dim=-1) - mix.sum(dim=-1))
    # amplitude_constraint = alpha * amplitude_difference.mean()
    # print('si_snr_loss_with_amplitude_constraint loss and amplitude_matching_loss:', loss.item(), amplitude_matching_loss(source, mix, alpha).item())
    # loss += amplitude_matching_loss(source, mix, alpha)

    return loss

iteration = 0 # Define a global variable to count the number of iterations
# Define a hook function to print gradients every dataloader iteration
def print_grad(module, grad_input, grad_output):
    global iteration
    # if iteration % len(dataloader) == 0:
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

# Define a helper function to compute inverse STFT for a single complex spectrogram
def compute_istft(complex_spectrogram, n_fft, hop_length):
    signal = torch.istft(complex_spectrogram, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft).to(complex_spectrogram.device), return_complex=False)
    return signal


readaudio = readAudio('/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Mixtures/Dev', '/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Sources/Dev', False, True, device)
audios = readaudio.__getitem__(50, 44100*SONGLEN)

dataloader = DataLoader(audios, batch_size=BATCH_SIZE, shuffle=True, persistent_workers=True, num_workers=8, pin_memory=True)

model = VQVAEWithUNet(1, 4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

scaler = None
if(ENABLE_AMP):
    scaler = GradScaler()


for epoch in range(NUM_EPOCH):
# for epoch in range(1):
    print(epoch)

    # Define the number of iterations to accumulate gradients before a parameter update
    num_iterations = 4
    # Define a counter to keep track of the accumulated iterations
    iteration_count = 0
    # Initialize the gradients
    optimizer.zero_grad()

    for i, data in enumerate(dataloader):
        # if i != 0: break
        mixes_uncut = data[0].to(device)
        sources_uncut = list(map(lambda x: x.to(device), data[1]))

        mixes_uncut
        sources_arr = []
        for tensors in sources_uncut:
            tensor_seg = tensors
            sources_arr.append(tensor_seg)

        mixes = mixes_uncut.unsqueeze(1).float() # make dimension from (batch size, seq len) to (batch size, num of input, seq len)

        sources = torch.stack([sources_arr[0], sources_arr[1], sources_arr[2], sources_arr[3]], dim=1).float()

        # print('mixes', mixes.shape)
        # print('sources', sources.shape)
        multi_scale_inputs = create_multi_scale_inputs(mixes, SCALES)
        multi_scale_targets = create_multi_scale_inputs(sources, SCALES)

        loss = torch.tensor(0., device=device)

        recon_loss_scale = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        spectvar_loss_scale = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        sisnr_loss_scale = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        ampmatch_loss_scale = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        commit_loss_scale = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]
        perplexity_scale = [torch.tensor(0., device=device) for _ in range(len(multi_scale_inputs))]

        recon_loss_sum = torch.tensor(0., device=device)
        spectvar_loss_sum = torch.tensor(0., device=device)
        sisnr_loss_sum = torch.tensor(0., device=device)
        ampmatch_loss_sum = torch.tensor(0., device=device)    
        commit_loss_sum = torch.tensor(0., device=device)   
        perplexity_sum = torch.tensor(0., device=device)

        # optimizer.zero_grad()

        with autocast(ENABLE_AMP):
            for i_scale in range(len(multi_scale_inputs)):
                # print(f'multi_scale_inputs[{i_scale}].shape', inputs_recon.shape)
                # print(f'multi_scale_targets[{i_scale}].shape', inputs_recon.shape)

                window_sizes=[1024, 2048, 4096]
                hop_lengths=[256, 512, 1024]
                for ws, hl in zip(window_sizes, hop_lengths):

                    inputs_stft_complex = torch.stack([compute_stft(signal, ws, hl) for signal in multi_scale_inputs[i_scale]])
                    targets_stft_complex = torch.stack([compute_stft(signal, ws, hl) for signal in multi_scale_targets[i_scale]])
                    inputs_stft_amplitude = torch.abs(inputs_stft_complex)
                    inputs_stft_phase = torch.angle(inputs_stft_complex)
                    targets_stft_amplitude = torch.abs(targets_stft_complex)
                    targets_stft_phase = torch.angle(targets_stft_complex)

                    # print('inputs_stft_amplitude', inputs_stft_amplitude.shape)
                    # print('inputs_stft_phase', inputs_stft_phase.shape)
                    # print('targets_stft_amplitude', targets_stft_amplitude.shape)
                    # print('targets_stft_phase', targets_stft_phase.shape)
                    # print('inputs_stft_phase max', torch.max(inputs_stft_phase))
                    # print('inputs_stft_phase min', torch.min(inputs_stft_phase))
                    output_stft_amplitude, commit_loss, perplexity, encoding_indices = model(inputs_stft_amplitude)

                    # print('output_stft_amplitude', output_stft_amplitude.shape)
                    # output_stft_complex = 

                    # Reconstruct the complex STFT using amplitude and phase
                    inputs_stft_complex_reconstructed = inputs_stft_amplitude * torch.exp(1j * inputs_stft_phase)
                    targets_stft_complex_reconstructed = targets_stft_amplitude * torch.exp(1j * targets_stft_phase)
                    output_stft_complex_reconstructed = output_stft_amplitude * torch.exp(1j * targets_stft_phase)

                    inputs_recon = torch.stack([compute_istft(complex_spectrogram, ws, hl) for complex_spectrogram in inputs_stft_complex_reconstructed])
                    targets_recon = torch.stack([compute_istft(complex_spectrogram, ws, hl) for complex_spectrogram in targets_stft_complex_reconstructed])
                    output_recon = torch.stack([compute_istft(complex_spectrogram, ws, hl) for complex_spectrogram in output_stft_complex_reconstructed])

                    # display(ipd.Audio(inputs_recon[1,0,:].detach().cpu().numpy().ravel(), rate = 16000))
                    #                 ########################

                    # calculate reconstruction loss log-likehood of the input which is MSE loss
                    recon_loss = RECONSCALE * F.mse_loss(output_recon, targets_recon)
                    # print('output_recon', output_recon.shape)
                    # print('inputs_recon', inputs_recon.shape)
                    # print('targets_recon', targets_recon.shape)
                    spectvar_loss = SPECTVARSCALE * multi_resolution_stft_loss_with_amplitude_constraint(output_recon, inputs_recon, targets_recon)
                    sisnr_loss = SISNRSCALE * si_snr_loss_with_amplitude_constraint(output_recon,  targets_recon, inputs_recon)
                    ampmatch_loss = amplitude_matching_loss(output_recon, inputs_recon, alpha=ALPHA) / SCALES[i_scale]

                    recon_loss_scale[i_scale] = recon_loss
                    spectvar_loss_scale[i_scale] = spectvar_loss
                    sisnr_loss_scale[i_scale] = sisnr_loss
                    ampmatch_loss_scale[i_scale] = ampmatch_loss
                    commit_loss_scale[i_scale] = commit_loss
                    perplexity_scale[i_scale] = perplexity

                    recon_loss_sum += recon_loss
                    spectvar_loss_sum += spectvar_loss
                    sisnr_loss_sum += sisnr_loss
                    ampmatch_loss_sum += ampmatch_loss
                    commit_loss_sum += commit_loss
                    perplexity_sum += perplexity

        loss = spectvar_loss_sum + sisnr_loss_sum + ampmatch_loss_sum + commit_loss_sum

        if(ENABLE_AMP):
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # Scale the loss to avoid overflow during backpropagation
            scaler.scale(loss / num_iterations).backward()
            # scaler.update()
        else:
            loss.backward()
            # optimizer.step()    

        iteration += 1 # Update the iteration count
        iteration_count += 1

        if iteration_count == num_iterations:
            # Call the optimizer step after accumulating gradients for the specified number of iterations
            if ENABLE_AMP:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        # Check the elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time // time_limit >= 1:
            # Save the model
            torch.save(model.state_dict(), f"output/model_vq-vae-spectro_lr_{LEARNING_RATE}_epoch_{epoch}_songlen_{SONGLEN}.pth")
            print("Model saved at {:.2f} hours.".format(elapsed_time / 3600))
            if TIMLIM != -999:
                break
            
    wandb.log({"perplexity sum": perplexity_sum, "loss": loss, "commit_loss_sum": commit_loss_sum, "recon_loss_sum": recon_loss_sum, "spectvar_loss_sum": spectvar_loss_sum, "sisnr_loss_sum": sisnr_loss_sum, "ampmatch_loss_sum": ampmatch_loss_sum})

    # # Clear GPU memory after the iteration
    # del mixes_uncut, sources_uncut, mixes, sources
    # torch.cuda.empty_cache()         
    if elapsed_time // time_limit >= 1:
        if TIMLIM != -999:
            break
            
# Save the model to disk
torch.save(model.state_dict(), f"output/model_vq-vae-spectro_lr_{LEARNING_RATE}_epoch_{NUM_EPOCH}_songlen_{SONGLEN}.pth")

wandb.finish()

readvalid = readAudio('/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Mixtures/Test', '/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Sources/Test', False, True, 'cpu')
# readvalid = readAudio('/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Mixtures/Dev', '/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Sources/Dev', False, True, device)
valids = readvalid.__getitem__(8, 44100*VALEN)
valiloader = DataLoader(valids, batch_size=4, shuffle=False, persistent_workers=True, num_workers=8, pin_memory=True)


model.eval()
torch.cuda.empty_cache()  
with torch.no_grad():
  # for i, data in enumerate(dataloader):
  for i, data in enumerate(valiloader):
        print(i)
        if i != 0: break
        mixes_uncut = data[0].to(device)
        sources_uncut = list(map(lambda x: x.to(device), data[1]))

        mixes_uncut
        sources_arr = []
        for tensors in sources_uncut:
            tensor_seg = tensors
            sources_arr.append(tensor_seg)

        mixes = mixes_uncut.unsqueeze(1).float() # make dimension from (batch size, seq len) to (batch size, num of input, seq len)

        sources = torch.stack([sources_arr[0], sources_arr[1], sources_arr[2], sources_arr[3]], dim=1).float()

        # print('mixes', mixes.shape)
        # print('sources', sources.shape)
        multi_scale_inputs = create_multi_scale_inputs(mixes, SCALES)
        multi_scale_targets = create_multi_scale_inputs(sources, SCALES)

        for i_scale in range(len(multi_scale_inputs)):
          # print(f'multi_scale_inputs[{i_scale}].shape', inputs_recon.shape)
          # print(f'multi_scale_targets[{i_scale}].shape', inputs_recon.shape)

            window_sizes=[1024, 2048, 4096]
            hop_lengths=[256, 512, 1024]
            for ws, hl in zip(window_sizes, hop_lengths):

                inputs_stft_complex = torch.stack([compute_stft(signal, ws, hl) for signal in multi_scale_inputs[i_scale]])
                targets_stft_complex = torch.stack([compute_stft(signal, ws, hl) for signal in multi_scale_targets[i_scale]])
                inputs_stft_amplitude = torch.abs(inputs_stft_complex)
                inputs_stft_phase = torch.angle(inputs_stft_complex)
                targets_stft_amplitude = torch.abs(targets_stft_complex)
                targets_stft_phase = torch.angle(targets_stft_complex)

                # print('inputs_stft_amplitude', inputs_stft_amplitude.shape)
                # print('inputs_stft_phase', inputs_stft_phase.shape)
                # print('targets_stft_amplitude', targets_stft_amplitude.shape)
                # print('targets_stft_phase', targets_stft_phase.shape)
                # print('inputs_stft_phase max', torch.max(inputs_stft_phase))
                # print('inputs_stft_phase min', torch.min(inputs_stft_phase))
                output_stft_amplitude, commit_loss, perplexity, encoding_indices = model(inputs_stft_amplitude)
                plt.figure(figsize=(10, 10))
                plt.subplot(3, 3, 1)
                plt.imshow(inputs_stft_amplitude[1,0,].detach().cpu().numpy())
                plt.subplot(3, 3, 2)
                plt.imshow(targets_stft_amplitude[1,0,].detach().cpu().numpy())
                plt.subplot(3, 3, 3)
                plt.imshow(output_stft_amplitude[1,0,].detach().cpu().numpy())
                plt.subplot(3, 3, 4)
                plt.imshow(targets_stft_amplitude[1,1,].detach().cpu().numpy())
                plt.subplot(3, 3, 5)
                plt.imshow(output_stft_amplitude[1,1,].detach().cpu().numpy())
                plt.subplot(3, 3, 6)
                plt.imshow(targets_stft_amplitude[1,2,].detach().cpu().numpy())
                plt.subplot(3, 3, 7)
                plt.imshow(output_stft_amplitude[1,2,].detach().cpu().numpy())
                plt.subplot(3, 3, 8)
                plt.imshow(targets_stft_amplitude[1,3,].detach().cpu().numpy())
                plt.subplot(3, 3, 9)
                plt.imshow(output_stft_amplitude[1,3,].detach().cpu().numpy())
                plt.show()
                plt.clf()

                # print('output_stft_amplitude', output_stft_amplitude.shape)
                # output_stft_complex = 

                # Reconstruct the complex STFT using amplitude and phase
                inputs_stft_complex_reconstructed = inputs_stft_amplitude * torch.exp(1j * inputs_stft_phase)
                targets_stft_complex_reconstructed = targets_stft_amplitude * torch.exp(1j * targets_stft_phase)
                output_stft_complex_reconstructed = output_stft_amplitude * torch.exp(1j * targets_stft_phase)

                inputs_recon = torch.stack([compute_istft(complex_spectrogram, ws, hl) for complex_spectrogram in inputs_stft_complex_reconstructed])
                targets_recon = torch.stack([compute_istft(complex_spectrogram, ws, hl) for complex_spectrogram in targets_stft_complex_reconstructed])
                output_recon = torch.stack([compute_istft(complex_spectrogram, ws, hl) for complex_spectrogram in output_stft_complex_reconstructed])

                display(ipd.Audio(inputs_recon[1,0,:].detach().cpu().numpy().ravel(), rate = 16000))
                display(ipd.Audio(targets_recon[1,0,:].detach().cpu().numpy().ravel(), rate = 16000))
                display(ipd.Audio(output_recon[1,0,:].detach().cpu().numpy().ravel(), rate = 16000))
                display(ipd.Audio(targets_recon[1,1,:].detach().cpu().numpy().ravel(), rate = 16000))
                display(ipd.Audio(output_recon[1,1,:].detach().cpu().numpy().ravel(), rate = 16000))
                display(ipd.Audio(targets_recon[1,2,:].detach().cpu().numpy().ravel(), rate = 16000))
                display(ipd.Audio(output_recon[1,2,:].detach().cpu().numpy().ravel(), rate = 16000))
                display(ipd.Audio(targets_recon[1,3,:].detach().cpu().numpy().ravel(), rate = 16000))
                display(ipd.Audio(output_recon[1,3,:].detach().cpu().numpy().ravel(), rate = 16000))

                for i_batch in range(len(inputs_recon)):
                    wavfile.write(f'output/mix_{i}_{i_batch}_scale_{i_scale}_ws_{ws}_hl_{hl}.wav', 16000, inputs_recon[i_batch,0,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'output/tar_bas_{i}_{i_batch}_scale_{i_scale}_ws_{ws}_hl_{hl}.wav', 16000, targets_recon[i_batch,0,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'output/tar_drm_{i}_{i_batch}_scale_{i_scale}_ws_{ws}_hl_{hl}.wav', 16000, targets_recon[i_batch,1,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'output/tar_voc_{i}_{i_batch}_scale_{i_scale}_ws_{ws}_hl_{hl}.wav', 16000, targets_recon[i_batch,2,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'output/tar_oth_{i}_{i_batch}_scale_{i_scale}_ws_{ws}_hl_{hl}.wav', 16000, targets_recon[i_batch,3,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'output/out_bas_{i}_{i_batch}_scale_{i_scale}_ws_{ws}_hl_{hl}.wav', 16000, output_recon[i_batch,0,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'output/out_drm_{i}_{i_batch}_scale_{i_scale}_ws_{ws}_hl_{hl}.wav', 16000, output_recon[i_batch,1,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'output/out_voc_{i}_{i_batch}_scale_{i_scale}_ws_{ws}_hl_{hl}.wav', 16000, output_recon[i_batch,2,:].detach().cpu().numpy().ravel())
                    wavfile.write(f'output/out_oth_{i}_{i_batch}_scale_{i_scale}_ws_{ws}_hl_{hl}.wav', 16000, output_recon[i_batch,3,:].detach().cpu().numpy().ravel())


