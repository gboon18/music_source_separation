# import nbimporter
import torch
import torch.nn as nn
import ptwt, pywt
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
# get_ipython().system('pip install torchaudio')
import torchaudio
import signal
import sys

import argparse

import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from dataset_cpuDataLoader_ptwt import readAudio

from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.optim.lr_scheduler import StepLR

import wandb
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--enable_dwt", action='store_true', help='Wanna do DWT?')
parser.add_argument("--enable_wandb", action='store_true', help='WandB use?')
parser.add_argument("--nlevel", default=6, type=int, help='number of dwt level')
parser.add_argument("--train_unet_only", action='store_true', help='Wanna train UNet only?')
parser.add_argument("--alpha", default=1./500., type=float, help='alpha for audio intensity matching')
parser.add_argument("--num_epoch", default=100, type=int, help='number of epochs?')
parser.add_argument("--lr", default=1e-4, type=float, help='learning rate?')
parser.add_argument("--len", default=3, type=int, help='length of the song')
parser.add_argument("--valen", default=3, type=int, help='length of the song for validation')
args = parser.parse_args()

# wandb.login(key='7a7346b9e3ee9dfebc6fc65da44ef3644f03298a')
api_key='7a7346b9e3ee9dfebc6fc65da44ef3644f03298a'
subprocess.call(f'wandb login {api_key}', shell=True)
if args.enable_wandb:
    wandb.init(project='VQ-VAE-DWT7', mode='online')
else:
    wandb.init(project='VQ-VAE-DWT7', mode='offline')

#####HYPER PARAMETERS#####
LEARNING_RATE = args.lr
NUM_EPOCH = args.num_epoch
##########################
DWT = args.enable_dwt
NLEVEL = args.nlevel
TRAIN_UNET_ONLY = args.train_unet_only
alpha = args.alpha
SONGLEN = int(args.len)
print('train_unet_only:', TRAIN_UNET_ONLY)
print('dwt nlevel:', NLEVEL)
print('alpha:', alpha)
print('num_epoch:', NUM_EPOCH)
print('lr:', LEARNING_RATE)
print('song length:', SONGLEN, 'sec')
print('song validation length:', args.valen, 'sec')

readaudio = readAudio('/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Mixtures/Dev', '/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Sources/Dev', False, True, 'cpu', DWT, NLEVEL)
audios = readaudio.__getitem__(1, 44100*SONGLEN)
# readvalid = readAudio('/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Mixtures/Test', '/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Sources/Test', False, True, 'cpu', DWT, NLEVEL)
readvalid = readAudio('/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Mixtures/Dev', '/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Sources/Dev', False, True, 'cpu', DWT, NLEVEL)
valids = readvalid.__getitem__(1, 44100*args.valen)

class UNetBlock(nn.Module):
    # def __init__(self, in_channels, out_channels, downsample=True, dropout_prob=0.5):
    def __init__(self, in_channels, out_channels, downsample=True, dropout_prob=0.5, last_block=False):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.relu = nn.Tanh()
        self.last_block = last_block
        self.tanh = nn.Tanh()
        self.batchnorm1 = nn.BatchNorm1d(out_channels)
        self.batchnorm2 = nn.BatchNorm1d(out_channels)
        self.max_pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x + 1e-8)))
        x = self.dropout(x)
        # x = self.relu(self.batchnorm2(self.conv2(x + 1e-8)))
        x = self.batchnorm2(self.conv2(x + 1e-8))
        if self.last_block:
            x = self.tanh(x)
            x = x - x.mean(dim=-1, keepdim=True) # ensuring the mean is around zero
        else:
            x = self.relu(x)
        x = self.dropout(x)        
        if self.downsample:
            return self.max_pool(x), x
        else:
            return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=[64, 128, 256, 512], dropout_prob=0.5):
        super().__init__()
        self.down_blocks = nn.ModuleList([UNetBlock(in_channels, hidden_channels[0], dropout_prob=dropout_prob)])
        self.down_blocks.extend([UNetBlock(hidden_channels[i - 1], hidden_channels[i], dropout_prob=dropout_prob) for i in range(1, len(hidden_channels))])

        self.up_blocks = nn.ModuleList()
        for i in range(len(hidden_channels) - 1, 0, -1):
            self.up_blocks.append(UNetBlock(hidden_channels[i] * 2, hidden_channels[i - 1], downsample=False, dropout_prob=dropout_prob))
        self.up_blocks.append(UNetBlock(hidden_channels[0] * 2, out_channels, downsample=False, dropout_prob=dropout_prob, last_block=True)) # in the last block, make it possible to have negativ value
        # self.up_blocks.append(UNetBlock(hidden_channels[0] * 2, out_channels, downsample=False, dropout_prob=dropout_prob, last_block=False)) # in the last block, make it possible to have negativ value
        # self.up_blocks.append(UNetBlock(hidden_channels[0] * 2, out_channels, downsample=False, dropout_prob=dropout_prob))

    def forward(self, x, skips=None, encoder=True):
        if encoder:
            skips = []
            for block in self.down_blocks:
                x, skip = block(x)
                skips.append(skip)
                # print('x =', x.shape)
                # [print('skip =', skip.shape) for skip in skips]
                if torch.isnan(x).any(): print('UNet encoder x is nan', torch.isnan(x).any())
                if torch.isnan(skip).any(): print('UNet encoder skip is nan', torch.isnan(skip).any())
            return x, skips

        else:  # decoder
            for i, block in enumerate(self.up_blocks):
                x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
                if skips and i < len(skips):
                    if torch.isnan(x).any(): print('UNet decoder x is nan', torch.isnan(x).any())
                    if torch.isnan(skips[-(i+1)]).any(): print('UNet decoder skip is nan', torch.isnan(skips[-(i+1)]).any())

                    # Pad the tensor if needed
                    diff = skips[-(i+1)].size(2) - x.size(2)
                    if diff > 0:
                        x = F.pad(x, (0, diff)) # (0, diff) means pad (left, right)
                    elif diff < 0:
                        skips[-(i+1)] = F.pad(skips[-(i+1)], (0, -diff))
                    # else:
                    #     skips[-(i+1)] = F.pad(skips[-(i+1)], (0, -diff))
                    
                    x = torch.cat([x, skips[-(i+1)]], dim=1) 
                x = block(x)
        
            return x

#learns the embeddings during training, and the commitment cost parameter controls the trade-off between reconstruction quality and the usage of the codebook entries.
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings) # this is to initialize the embedding weights to make sure that the initial distances between the embeddings are equal
        self.commitment_cost = commitment_cost # controls the trade-off between reconstruction quality and the usage of the codebook entries
        
    def forward(self, inputs):
        # Reshape the inputs so that the spatial dimensions are collapsed into a single dimension
        # (Batch_size, num_channels, sequence_length) -> (Batch_size, sequence_length, num_channels)
        if torch.isnan(inputs).any(): print('inputs is nan:', torch.isnan(inputs).any())
        inputs = inputs.permute(0, 2, 1).contiguous() # contiguous() helps memory management by making sure that the data is stored in a single, contiguous chunk of memory
        if torch.isnan(inputs).any(): print('inputs reshaped is nan:', torch.isnan(inputs).any())
        
        #flatten the inputs
        inputs_flat = inputs.view(-1, self.embedding_dim)

        # print('inputs_flat:', inputs_flat.shape)
        if torch.isnan(self.embedding.weight).any(): print('self.embedding.weight is nan:', torch.isnan(self.embedding.weight).any())

        # Calculate distances between inputs and embedding
        distances = (torch.sum(inputs_flat ** 2, dim=1, keepdim=True)
                        + torch.sum(self.embedding.weight ** 2, dim=1)
                        - 2 * torch.matmul(inputs_flat, self.embedding.weight.t()))
        
        if torch.isnan(distances).any(): print('distances is nan:', torch.isnan(distances).any())

        # Encoding indices are the indices of the embedding vectors that are closest to the inputs
        # Calculate the encoding indices using the distances
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        if torch.isnan(encoding_indices).any(): print('encoding_indices is nan:', torch.isnan(encoding_indices).any())
        if torch.isnan(encodings).any(): print('encodings is nan:', torch.isnan(encodings).any())

        # Get the quantized version of the inputs by multiplying the encodings with the embedding vectors
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        if torch.isnan(quantized).any(): print('quantized is nan:', torch.isnan(quantized).any())

        # Calculate the loss between the quantized version of the inputs and the original inputs
        e_latent_loss = F.mse_loss(quantized.detach(), inputs) # This is the loss that is backpropagated to the encoder to make it learn the codebook. 
        # detach() is used to make sure that the gradients are not propagated to the encoder
        q_latent_loss = F.mse_loss(quantized, inputs.detach()) # This is the loss that is backpropagated to the embedding layer to make it learn the codebook.
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Return the quantized version of the inputs, the loss, the encoding indices, and the encodings
        quantized = inputs + (quantized - inputs).detach() # this is to make sure that the gradients are not propagated to the encoder by detaching the quantized tensor from the computational graph
        # Why the above line have inputs + (quantized - inputs).detach() instead of just quantized.detach()?
        # The reason is that the quantized tensor is not the same as the inputs. The quantized tensor is the closest embedding vector to the inputs.
        # So, we need to add the difference between the quantized tensor and the inputs to the inputs to get the quantized tensor.
        # This is because the gradients are not propagated to the encoder, so the encoder will not learn anything.
        # The above line is to make sure that the gradients are propagated to the encoder.

        avg_probs = torch.mean(encodings, dim=0) # this is to calculate the perplexity. For each channel, we calculate the average probability of each embedding vector being used.
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))) # perplexity is a measure of how well the codebook is able to predict the next token in the sequence

        # No need to reshape because we will process the quantized tensor in the transformer encoder
        # # Reshape the quantized version of the inputs to the original shape of the inputs
        # quantized = quantized.permute(0, 2, 1).contiguous()
        # print('quantized reshaped:', quantized.shape)

        # print('VectorQuantizer forward quantized:', quantized.shape)
        if torch.isnan(quantized).any(): print('VectorQuantizer forward quantized is nan:', torch.isnan(quantized).any())
        
        return quantized, loss, perplexity, encoding_indices
    
class VQVAEWithUNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=[64, 128, 256, 512], num_embeddings=64, embedding_dim=512, dropout_prob=0.5, train_unet_only=False, num_levels=6):
        super().__init__()

        self.num_levels = num_levels+1
        self.unets = nn.ModuleList([UNet(in_channels, out_channels, hidden_channels, dropout_prob=dropout_prob) for _ in range(self.num_levels)])
        self.vq_layers = nn.ModuleList([VectorQuantizer(num_embeddings, embedding_dim, commitment_cost=0.25) for _ in range(self.num_levels)])
        self.train_unet_only = train_unet_only

    def set_train_unet_only(self, train_unet_only):
        self.train_unet_only = train_unet_only
        # print('set train unet only to', train_unet_only)
        
    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        output_list = []
        loss_list = []
        perplexity_list = []
        encoding_indices_list = []

        for i in range(self.num_levels):
            x, skips = self.unets[i](inputs[i], encoder=True)
            
##########################
            if self.train_unet_only:
                x = self.unets[i](x, skips, encoder=False)
                output_list.append(x)
                
            else:
                # print('VQVAEWithUNet Forward level:', i)
                quantized, loss, perplexity, encoding_indices = self.vq_layers[i](x)
                x = self.unets[i](quantized.permute(0, 2, 1).contiguous(), skips, encoder=False)
                output_list.append(x)
                loss_list.append(loss)
                perplexity_list.append(perplexity)
                encoding_indices_list.append(encoding_indices)
##########################
#             x, skips = self.unets[i](inputs[i], encoder=True)
#             quantized, loss, perplexity, encoding_indices = self.vq_layers[i](x)
#             x = self.unets[i](quantized.permute(0, 2, 1).contiguous(), skips, encoder=False)
            
#             output_list.append(x)
#             loss_list.append(loss)
#             perplexity_list.append(perplexity)
#             encoding_indices_list.append(encoding_indices)
##########################

        if self.train_unet_only:
            return output_list
        else:
            total_loss = sum(loss_list)
            total_perplexity = sum(perplexity_list)
            # encoding_indices = torch.stack(encoding_indices_list, dim=1)

            # return output_list, total_loss, total_perplexity, encoding_indices
            # return output_list, total_loss, total_perplexity, encoding_indices_list    
            return output_list, loss_list, perplexity_list, encoding_indices_list    


dataloader = DataLoader(audios, batch_size=2, shuffle=True, persistent_workers=True, num_workers=8, pin_memory=True)
valiloader = DataLoader(valids, batch_size=2, shuffle=False, persistent_workers=True, num_workers=8, pin_memory=True)

def spectrogram_mse_loss(mixes, source, target, window_size=1024, hop_length=512):
    assert source.shape == target.shape, "Source and target tensors should have the same shape"
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
    
    # print('multi_resolution_stft_loss_with_amplitude_constraint loss and amplitude_matching_loss:', total_loss.item(), amplitude_matching_loss(sources, mixes, alpha).item())
    # total_loss += amplitude_matching_loss(sources, mixes, alpha)
    
    total_loss /= len(list(zip(window_sizes, hop_lengths)))
    
    return total_loss


# def si_snr_loss(source, target, eps=1e-8):
#     target = target - target.mean(dim=-1, keepdim=True)
#     source = source - source.mean(dim=-1, keepdim=True)
    
#     s_target = torch.sum(target * source, dim=-1, keepdim=True) * target / (torch.sum(target**2, dim=-1, keepdim=True) + eps)
#     e_noise = source - s_target
    
#     si_snr = 10 * torch.log10(torch.sum(s_target**2, dim=-1) / (torch.sum(e_noise**2, dim=-1) + eps) + eps)
    
#     # Average the loss across the channels
#     loss = -si_snr.mean(dim=1).mean()

#     return loss

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

    
local_rank = 0
device = torch.device('cuda:%d'%local_rank)
model_vqvaeunet = VQVAEWithUNet(in_channels=1, out_channels=4, dropout_prob=0.2, num_levels=NLEVEL).to(device)
optimizer = torch.optim.Adam(model_vqvaeunet.parameters(), lr=LEARNING_RATE)

# attach hook to the first convolutional layer of UNet encoder
# handle = model_vqvaeunet.unet.down_blocks[0].register_full_backward_hook(print_grad)
# scheduler = StepLR(optimizer, step_size=40, gamma=0.1) # every 1 epoch, lower the lr by 10% 
for epoch in range(NUM_EPOCH):
    if epoch % 100 == 0:
        print('epoch', epoch)

    model_vqvaeunet.set_train_unet_only(TRAIN_UNET_ONLY)
    # if epoch < 10000:
    #     TRAIN_UNET_ONLY = True
    #     model_vqvaeunet.set_train_unet_only(TRAIN_UNET_ONLY)
    for i, data in enumerate(dataloader):
            
        mix_coeffs = data[0]
        source_coeffs = data[1]
        input_levels = [mix.float().to(device) for mix in mix_coeffs] 

        target_levels = [
            torch.stack([bass, drum, vocal, other], dim=1).squeeze(2).float().to(device)
            for bass, drum, vocal, other in zip(*source_coeffs)
        ]

        # Pass input_levels and target_levels to the model
        # Your model should be able to handle multiple input and target levels
#         print('len(input_levels)', len(input_levels))
#         print('len(input_levels[0])', len(input_levels[0]))
#         print('input_levels[0][0].shape', input_levels[0][0].shape)
#         print('input_levels[0][1].shape', input_levels[0][1].shape)
        if TRAIN_UNET_ONLY:
            output_levels = model_vqvaeunet(input_levels)
            commit_loss = []
            perplexities = []
            encoding_indices_list = []
        else:
            output_levels, commit_loss, perplexities, encoding_indices_list = model_vqvaeunet(input_levels)

#         print('input_levels len:', len(input_levels))    
#         print('output_levels len:', len(output_levels))    
            
        loss = torch.tensor(0., device=device)
        recon_loss_sum = torch.tensor(0., device=device)
        spect_loss_sum = torch.tensor(0., device=device)
        spectvar_loss_sum = torch.tensor(0., device=device)
        sisnr_loss_sum = torch.tensor(0., device=device)
        ampmatch_loss_sum = torch.tensor(0., device=device)
        for i_lv in range(len(output_levels)):
            # print(i_lv, "output:", output_levels[i_lv].shape)
            # print(i_lv, "commit_loss:", commit_loss[i_lv])
            # print(i_lv, "perplexity:", perplexities[i_lv])
            # print(i_lv, "encoding_indices_list;", encoding_indices_list[i_lv].shape)
            recon_loss = F.mse_loss(output_levels[i_lv], target_levels[i_lv])
            spect_loss = spectrogram_mse_loss(input_levels[i_lv], output_levels[i_lv], target_levels[i_lv])
            spectvar_loss = multi_resolution_stft_loss_with_amplitude_constraint(input_levels[i_lv], output_levels[i_lv], target_levels[i_lv])
            sisnr_loss = si_snr_loss_with_amplitude_constraint(output_levels[i_lv], target_levels[i_lv], input_levels[i_lv])
            # print(i_lv, "recon_loss:", recon_loss)
            # print(i_lv, "spect_loss:", spect_loss)
            # print(i_lv, "spectvar_loss:", spectvar_loss)
            # print(i_lv, "sisnr_loss:", sisnr_loss)
            # loss += commit_loss[i_lv] + spectvar_loss + sisnr_loss/20
            recon_loss_sum += recon_loss
            spect_loss_sum += spect_loss
            spectvar_loss_sum += spectvar_loss
            sisnr_loss_sum += sisnr_loss
            
            ampmatch_loss = amplitude_matching_loss(output_levels[i_lv], input_levels[i_lv], alpha=alpha)
            ampmatch_loss_sum += ampmatch_loss
            
            # if TRAIN_UNET_ONLY:
            #     wandb.log({"i_lv": i_lv, "recon_loss": recon_loss.item(), "spect_loss": spect_loss.item(), "spectvar_loss": spectvar_loss.item(), "sisnr_loss": sisnr_loss.item(), f"ampmatch_loss": ampmatch_loss.item()})
            # else:
            #     wandb.log({"i_lv": i_lv, "perplexity": perplexities[i_lv], "commit_loss": commit_loss[i_lv], "recon_loss": recon_loss.item(), "spect_loss": spect_loss.item(), "spectvar_loss": spectvar_loss.item(), "sisnr_loss": sisnr_loss.item(), f"ampmatch_loss": ampmatch_loss.item()})

        if TRAIN_UNET_ONLY:
            perplexity_sum = torch.tensor(0., device=device)
            commit_loss_sum = torch.tensor(0., device=device)
        else:
            perplexity_sum = sum(perplexities)    
            commit_loss_sum = sum(commit_loss)

        if TRAIN_UNET_ONLY:
            # if iteration < 5000:
            #     loss += spectvar_loss_sum + sisnr_loss_sum/20 + ampmatch_loss_sum
            # else:
            #     loss += sisnr_loss_sum/15 + ampmatch_loss_sum
            sisnr_loss_sum /= 20
            loss += spectvar_loss_sum + sisnr_loss_sum + ampmatch_loss_sum
        else:
            # if iteration < 4e3:
            #     loss += commit_loss_sum + spectvar_loss_sum/2 + sisnr_loss_sum/40 + ampmatch_loss_sum
            # elif iteration >= 4e3:
            #     loss += commit_loss_sum*2 + sisnr_loss_sum/40 + ampmatch_loss_sum
            spectvar_loss_sum /= 2
            sisnr_loss_sum /=40
            loss += commit_loss_sum + spectvar_loss_sum + sisnr_loss_sum + ampmatch_loss_sum
        
        if i==0: 
            wandb.log({"perplexity sum": perplexity_sum, "loss": loss, "commit_loss_sum": commit_loss_sum, "recon_loss_sum": recon_loss_sum, "spect_loss_sum": spect_loss_sum, "spectvar_loss_sum": spectvar_loss_sum, "sisnr_loss_sum": sisnr_loss_sum, f"ampmatch_loss_sum": ampmatch_loss_sum})

                    
        loss.backward()

        optimizer.step()    
        optimizer.zero_grad()

        iteration += 1 # Update the iteration count
        
        
#         # Set up the signal handler to save the model on keyboard interrupt
#         model_path = 'output7/model_vq-vae-dwt3_epoch_{}.pth'
#         if TRAIN_UNET_ONLY: model_path = 'output7/model_vq-vae-dwt3_unetonly_epoch_{}.pth'
#         def signal_handler(signum, frame):
#             print("Keyboard interrupt received. Saving model and exiting...")
            
#             # Construct the model file name with the current epoch number
#             model_file = model_path.format(epoch)

#             # Save the model to the local file system
#             torch.save(model_vqvaeunet.state_dict(), model_file)

#             # # Save the model to WandB
#             # wandb.save(model_file)
                
#             print("Byeeeee")
#             # Exit the program
#             sys.exit(0)

#         # Set up the signal handler to save the model on keyboard interrupt
#         signal.signal(signal.SIGINT, signal_handler)

#     # Validation loop
#     with torch.no_grad():
#         for i, data in enumerate(valiloader):

#             mix_coeffs = data[0]
#             source_coeffs = data[1]
#             input_levels = [mix.float().to(device) for mix in mix_coeffs] 

#             target_levels = [
#                 torch.stack([bass, drum, vocal, other], dim=1).squeeze(2).float().to(device)
#                 for bass, drum, vocal, other in zip(*source_coeffs)
#             ]

#             # Pass input_levels and target_levels to the model
#             # Your model should be able to handle multiple input and target levels

#             if TRAIN_UNET_ONLY:
#                 output_levels = model_vqvaeunet(input_levels)
#                 commit_val_loss = []
#                 val_perplexities = []
#                 encoding_indices_list = []
#             else:
#                 output_levels, commit_val_loss, val_perplexities, encoding_indices_list = model_vqvaeunet(input_levels)

#             val_loss = torch.tensor(0., device=device)
#             recon_val_loss_sum = torch.tensor(0., device=device)
#             spect_val_loss_sum = torch.tensor(0., device=device)
#             spectvar_val_loss_sum = torch.tensor(0., device=device)
#             sisnr_val_loss_sum = torch.tensor(0., device=device)
#             ampmatch_val_loss_sum = torch.tensor(0., device=device)
#             for i_lv in range(len(output_levels)):
#                 # print(i_lv, "output:", output_levels[i_lv].shape)
#                 # print(i_lv, "commit_val_loss:", commit_val_loss[i_lv])
#                 # print(i_lv, "val_perplexity:", val_perplexities[i_lv])
#                 # print(i_lv, "encoding_indices_list;", encoding_indices_list[i_lv].shape)
#                 recon_val_loss = F.mse_val_loss(output_levels[i_lv], target_levels[i_lv])
#                 spect_val_loss = spectrogram_mse_val_loss(input_levels[i_lv], output_levels[i_lv], target_levels[i_lv])
#                 spectvar_val_loss = multi_resolution_stft_val_loss_with_amplitude_constraint(input_levels[i_lv], output_levels[i_lv], target_levels[i_lv])
#                 sisnr_val_loss = si_snr_val_loss_with_amplitude_constraint(output_levels[i_lv], target_levels[i_lv], input_levels[i_lv])
#                 # print(i_lv, "recon_val_loss:", recon_val_loss)
#                 # print(i_lv, "spect_val_loss:", spect_val_loss)
#                 # print(i_lv, "spectvar_val_loss:", spectvar_val_loss)
#                 # print(i_lv, "sisnr_val_loss:", sisnr_val_loss)
#                 # val_loss += commit_val_loss[i_lv] + spectvar_val_loss + sisnr_val_loss/20
#                 recon_val_loss_sum += recon_val_loss
#                 spect_val_loss_sum += spect_val_loss
#                 spectvar_val_loss_sum += spectvar_val_loss
#                 sisnr_val_loss_sum += sisnr_val_loss
#                 ampmatch_val_loss = amplitude_matching_val_loss(output_levels[i_lv], input_levels[i_lv], alpha=alpha)
#                 ampmatch_val_loss_sum += ampmatch_val_loss
#                 # if TRAIN_UNET_ONLY:
#                 #     wandb.log({"val i_lv": i_lv, "val recon_val_loss": recon_val_loss.item(), "val spect_val_loss": spect_val_loss.item(), "val spectvar_val_loss": spectvar_val_loss.item(), "val sisnr_val_loss": sisnr_val_loss.item(), f"val ampmatch_val_loss": ampmatch_val_loss.item()})
#                 # else:
#                 #     wandb.log({"val i_lv": i_lv, "val val_perplexity": val_perplexities[i_lv], "val commit_val_loss": commit_val_loss[i_lv], "val recon_val_loss": recon_val_loss.item(), "val spect_val_loss": spect_val_loss.item(), "val spectvar_val_loss": spectvar_val_loss.item(), "val sisnr_val_loss": sisnr_val_loss.item(), f"val ampmatch_val_loss": ampmatch_val_loss.item()})

#             if TRAIN_UNET_ONLY:
#                 val_perplexity_sum = torch.tensor(0., device=device)
#                 commit_val_loss_sum = torch.tensor(0., device=device)
#             else:
#                 val_perplexity_sum = sum(val_perplexities)    
#                 commit_val_loss_sum = sum(commit_val_loss)

#             if TRAIN_UNET_ONLY:
#                 # if iteration < 5000:
#                 #     val_loss += spectvar_val_loss_sum + sisnr_val_loss_sum/20 + ampmatch_val_loss_sum
#                 # else:
#                 #     val_loss += sisnr_val_loss_sum/15 + ampmatch_val_loss_sum
#                 sisnr_val_loss_sum /= 20
#                 val_loss += spectvar_val_loss_sum + sisnr_val_loss_sum + ampmatch_val_loss_sum
#             else:
#                 # if iteration < 4e3:
#                 #     val_loss += commit_val_loss_sum + spectvar_val_loss_sum/2 + sisnr_val_loss_sum/40 + ampmatch_val_loss_sum
#                 # elif iteration >= 4e3:
#                 #     val_loss += commit_val_loss_sum*2 + sisnr_val_loss_sum/40 + ampmatch_val_loss_sum
#                 spectvar_val_loss_sum /= 2
#                 sisnr_val_loss_sum /=40
#                 val_loss += commit_val_loss_sum + spectvar_val_loss_sum + sisnr_val_loss_sum + ampmatch_val_loss_sum
        
#             wandb.log({"val perplexity sum": val_perplexity_sum, "val loss": val_loss, "val commit_loss_sum": commit_val_loss_sum, "val recon_loss_sum": recon_val_loss_sum, "val spect_loss_sum": spect_val_loss_sum, "val spectvar_loss_sum": spectvar_val_loss_sum, "val sisnr_loss_sum": sisnr_val_loss_sum, f"val ampmatch_loss_sum": ampmatch_val_loss_sum})

    # scheduler.step()
    
# Detach hook
# handle.remove()
# Save the model to disk
if TRAIN_UNET_ONLY:
    torch.save(model_vqvaeunet.state_dict(), f"output7/model_vq-vae-dwt3_unetonly_lr_{LEARNING_RATE}_epoch_{NUM_EPOCH}_nlevel_{NLEVEL}_songlen_{SONGLEN}.pth")
else:    
    torch.save(model_vqvaeunet.state_dict(), f"output7/model_vq-vae-dwt3_lr_{LEARNING_RATE}_epoch_{NUM_EPOCH}_nlevel_{NLEVEL}_songlen_{SONGLEN}.pth")

wandb.finish()

# Calculate bits per dimension
# Obtain nagative log-likelihood of the input
# bits_per_dim = commit_loss / len(mixes[-1]*4) / np.log(2) # 4 is the number of sources.
# print("Bits per dimension:", bits_per_dim)

# Let's evaluate the model

from scipy.io import wavfile

model_vqvaeunet.eval()

with torch.no_grad():
    model_vqvaeunet.set_train_unet_only(TRAIN_UNET_ONLY)
    # for i, data in enumerate(dataloader):
    for i, data in enumerate(valiloader):
        mix_coeffs = data[0]
        source_coeffs = data[1]
        input_levels = [mix.float().to(device) for mix in mix_coeffs] 

        target_levels = [
            torch.stack([bass, drum, vocal, other], dim=1).squeeze(2).float().to(device)
            for bass, drum, vocal, other in zip(*source_coeffs)
        ]

        if TRAIN_UNET_ONLY:
            output_levels = model_vqvaeunet(input_levels)
            commit_loss = []
            perplexities = []
            encoding_indices_list = []
        else:
            output_levels, commit_loss, perplexities, encoding_indices_list = model_vqvaeunet(input_levels)

        loss = torch.tensor(0., device=device)
        recon_loss_sum = torch.tensor(0., device=device)
        spect_loss_sum = torch.tensor(0., device=device)
        spectvar_loss_sum = torch.tensor(0., device=device)
        sisnr_loss_sum = torch.tensor(0., device=device)
        ampmatch_loss_sum = torch.tensor(0., device=device)
        for i_lv in range(len(output_levels)):
            # print(i_lv, "output:", output_levels[i_lv].shape)
            # print(i_lv, "commit_loss:", commit_loss[i_lv])
            # print(i_lv, "perplexity:", perplexities[i_lv])
            # print(i_lv, "encoding_indices_list;", encoding_indices_list[i_lv].shape)
            recon_loss = F.mse_loss(output_levels[i_lv], target_levels[i_lv])
            spect_loss = spectrogram_mse_loss(input_levels[i_lv], output_levels[i_lv], target_levels[i_lv])
            spectvar_loss = multi_resolution_stft_loss_with_amplitude_constraint(input_levels[i_lv], output_levels[i_lv], target_levels[i_lv])
            sisnr_loss = si_snr_loss_with_amplitude_constraint(output_levels[i_lv], target_levels[i_lv], input_levels[i_lv])
            # print(i_lv, "recon_loss:", recon_loss)
            # print(i_lv, "spect_loss:", spect_loss)
            # print(i_lv, "spectvar_loss:", spectvar_loss)
            # print(i_lv, "sisnr_loss:", sisnr_loss)
            # loss += commit_loss[i_lv] + spectvar_loss + sisnr_loss/20
            recon_loss_sum += recon_loss
            spect_loss_sum += spect_loss
            spectvar_loss_sum += spectvar_loss
            sisnr_loss_sum += sisnr_loss
            ampmatch_loss = amplitude_matching_loss(output_levels[i_lv], input_levels[i_lv], alpha=alpha)
            ampmatch_loss_sum += ampmatch_loss

        if TRAIN_UNET_ONLY:
            perplexity_sum = torch.tensor(0., device=device)
            commit_loss_sum = torch.tensor(0., device=device)
        else:
            perplexity_sum = sum(perplexities)    
            commit_loss_sum = sum(commit_loss)

        if TRAIN_UNET_ONLY:
            # if iteration < 5000:
            #     loss += spectvar_loss_sum + sisnr_loss_sum/20 + ampmatch_loss_sum
            # else:
            #     loss += sisnr_loss_sum/15 + ampmatch_loss_sum
            sisnr_loss_sum /= 20
            loss += spectvar_loss_sum + sisnr_loss_sum + ampmatch_loss_sum
        else:
            # if iteration < 4e3:
            #     loss += commit_loss_sum + spectvar_loss_sum/2 + sisnr_loss_sum/40 + ampmatch_loss_sum
            # elif iteration >= 4e3:
            #     loss += commit_loss_sum*2 + sisnr_loss_sum/40 + ampmatch_loss_sum
            spectvar_loss_sum /= 2
            sisnr_loss_sum /=40
            loss += commit_loss_sum + spectvar_loss_sum + sisnr_loss_sum + ampmatch_loss_sum
               
        coeff_input = [coeff_input[0] for coeff_input in input_levels]

        coeff_target_bas = [coeff_target[0][0].unsqueeze(0) for coeff_target in target_levels]
        coeff_target_drm = [coeff_target[0][1].unsqueeze(0) for coeff_target in target_levels]
        coeff_target_voc = [coeff_target[0][2].unsqueeze(0) for coeff_target in target_levels]
        coeff_target_oth = [coeff_target[0][3].unsqueeze(0) for coeff_target in target_levels]
            
        coeff_output_bas = [coeff_output[0][0].unsqueeze(0) for coeff_output in output_levels]
        coeff_output_drm = [coeff_output[0][1].unsqueeze(0) for coeff_output in output_levels]
        coeff_output_voc = [coeff_output[0][2].unsqueeze(0) for coeff_output in output_levels]
        coeff_output_oth = [coeff_output[0][3].unsqueeze(0) for coeff_output in output_levels]
            
        print(coeff_input[0].shape)
        print(coeff_target_bas[0].shape)
        mix_l_rec = readaudio.waverec(coeffs=coeff_input, wavelet='db8')
        tar_bas_l_rec = readaudio.waverec(coeffs=coeff_target_bas, wavelet='db8')
        tar_drm_l_rec = readaudio.waverec(coeffs=coeff_target_drm, wavelet='db8')
        tar_voc_l_rec = readaudio.waverec(coeffs=coeff_target_voc, wavelet='db8')
        tar_oth_l_rec = readaudio.waverec(coeffs=coeff_target_oth, wavelet='db8')

        out_bas_l_rec = readaudio.waverec(coeffs=coeff_output_bas, wavelet='db8')
        out_drm_l_rec = readaudio.waverec(coeffs=coeff_output_drm, wavelet='db8')
        out_voc_l_rec = readaudio.waverec(coeffs=coeff_output_voc, wavelet='db8')
        out_oth_l_rec = readaudio.waverec(coeffs=coeff_output_oth, wavelet='db8')
        
#         print('input')
#         display(ipd.Audio(mix_l_rec.cpu().numpy().ravel(), rate = 16000))
#         print('target bass')
#         display(ipd.Audio(tar_bas_l_rec.cpu().numpy().ravel(), rate = 16000))
#         print('output bass')
#         display(ipd.Audio(out_bas_l_rec.cpu().numpy().ravel(), rate = 16000))
#         print('target drum')
#         display(ipd.Audio(tar_drm_l_rec.cpu().numpy().ravel(), rate = 16000))
#         print('output drum')
#         display(ipd.Audio(out_drm_l_rec.cpu().numpy().ravel(), rate = 16000))
#         print('target vocal')
#         display(ipd.Audio(tar_voc_l_rec.cpu().numpy().ravel(), rate = 16000))
#         print('output vocal')
#         display(ipd.Audio(out_voc_l_rec.cpu().numpy().ravel(), rate = 16000))
#         print('target other')
#         display(ipd.Audio(tar_oth_l_rec.cpu().numpy().ravel(), rate = 16000))
#         print('output other')
#         display(ipd.Audio(out_oth_l_rec.cpu().numpy().ravel(), rate = 16000))

        if TRAIN_UNET_ONLY:
            wavfile.write(f'output7/trained_unetonly_dwt_bass_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, out_bas_l_rec.cpu().numpy().ravel())
            wavfile.write(f'output7/trained_unetonly_dwt_drum_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, out_drm_l_rec.cpu().numpy().ravel())
            wavfile.write(f'output7/trained_unetonly_dwt_vocal_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, out_voc_l_rec.cpu().numpy().ravel())
            wavfile.write(f'output7/trained_unetonly_dwt_other_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, out_oth_l_rec.cpu().numpy().ravel())
            wavfile.write(f'output7/target_unetonly_dwt_bass_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, tar_bas_l_rec.cpu().numpy().ravel())
            wavfile.write(f'output7/target_unetonly_dwt_drum_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, tar_drm_l_rec.cpu().numpy().ravel())
            wavfile.write(f'output7/target_unetonly_dwt_vocal_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, tar_voc_l_rec.cpu().numpy().ravel())
            wavfile.write(f'output7/target_unetonly_dwt_other_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, tar_oth_l_rec.cpu().numpy().ravel())
        else:
            wavfile.write(f'output7/trained_dwt_bass_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, out_bas_l_rec.cpu().numpy().ravel())
            wavfile.write(f'output7/trained_dwt_drum_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, out_drm_l_rec.cpu().numpy().ravel())
            wavfile.write(f'output7/trained_dwt_vocal_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, out_voc_l_rec.cpu().numpy().ravel())
            wavfile.write(f'output7/trained_dwt_other_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, out_oth_l_rec.cpu().numpy().ravel())
            wavfile.write(f'output7/target_dwt_bass_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, tar_bas_l_rec.cpu().numpy().ravel())
            wavfile.write(f'output7/target_dwt_drum_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, tar_drm_l_rec.cpu().numpy().ravel())
            wavfile.write(f'output7/target_dwt_vocal_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, tar_voc_l_rec.cpu().numpy().ravel())
            wavfile.write(f'output7/target_dwt_other_lr_{LEARNING_RATE}_nlevel_{NLEVEL}_{i}.wav', 16000, tar_oth_l_rec.cpu().numpy().ravel())

#         #plot the mixes tensor and sources tensor
#         plt.figure(figsize=(20, 10))
#         plt.subplot(2, 1, 1)

#         plt.plot(mix_l_rec.cpu().numpy().reval())
#         plt.title('mixes')
#         plt.subplot(2, 1, 2)
#         plt.plot(tar_bas_l_rec.cpu().numpy().reval(), label='target bass', color='red', alpha=0.3)
#         plt.plot(tar_drm_l_rec.cpu().numpy().reval(), label='target drums', color='blue', alpha=0.3)
#         plt.plot(tar_voc_l_rec.cpu().numpy().reval(), label='target vocals', color='cyan', alpha=0.3)
#         plt.plot(tar_oth_l_rec.cpu().numpy().reval(), label='target others', color='orange', alpha=0.3)
#         plt.legend(loc='upper left')
#         plt.title('sources')
#         plt.show()

#         #plot the output tensor with the sources tensor
#         plt.figure(figsize=(20, 10))
#         plt.subplot(2, 1, 1)
#         plt.plot(out_bas_l_rec.cpu().numpy().reval(), label='trained bass', color='red', alpha=0.3)
#         plt.plot(out_drm_l_rec.cpu().numpy().reval(), label='trained drums', color='blue', alpha=0.3)
#         plt.plot(out_voc_l_rec.cpu().numpy().reval(), label='trained vocals', color='cyan', alpha=0.3)
#         plt.plot(out_oth_l_rec.cpu().numpy().reval(), label='trained others', color='orange', alpha=0.3)
#         plt.legend(loc='upper left')
#         plt.title('output')
#         plt.subplot(2, 1, 2)
#         plt.plot(tar_bas_l_rec.cpu().numpy().reval(), label='target bass', color='red', alpha=0.3)
#         plt.plot(tar_drm_l_rec.cpu().numpy().reval(), label='target drums', color='blue', alpha=0.3)
#         plt.plot(tar_voc_l_rec.cpu().numpy().reval(), label='target vocals', color='cyan', alpha=0.3)
#         plt.plot(tar_oth_l_rec.cpu().numpy().reval(), label='target others', color='orange', alpha=0.3)
#         plt.legend(loc='upper left')
#         plt.title('sources')
#         plt.show()

