#!/usr/bin/env python
# import nbimporter # pip install import_ipynb
# import sys
# sys.path.append("/global/u2/h/hsko/jupyter/ML/LSTM")
from dataset_cpuDataLoader import readAudio
from model import LSTMModel
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.optim.lr_scheduler import StepLR

# from torchinfo import summary
# import traceback
# import gc #empty memory
import sys
import random

from scipy.io import wavfile

import os
import glob
import time
import pprint
import subprocess
import wandb
import matplotlib.pyplot as plt
    
import argparse

from torch.utils.tensorboard import SummaryWriter

import logging
from utils import logging_utils
logging_utils.config_logger()

# def train_epoch(model, dataloader, criterion, optimizer, epoch, iters, tr_loss, scaler):
def train_epoch(device, model, dataloader, criterion, optimizer, epoch, scaler, iters, batch_size):
    cumu_loss = 0
    
    start_time = time.time()
    tr_loss = []
    tr_time = 0.
    dat_time = 0.
    log_time = 0.
    
    step_count = 0
    
    # Train the model
    # for epoch in range(NUM_EPOCHS):
    # input_tensors = []
    # output_tensors = []
    
    for i, data in enumerate(dataloader):
        # print(i, len(dataloader), len(dataloader.dataset), len(data), data[0].shape, len(data[1]), data[1][0].shape, data[1][1].shape, data[1][2].shape, data[1][3].shape)

        if not args.num_chunks == None: 
            if args.num_chunks == i: break
        if (args.enable_manual_profiling and world_rank==0):
            if (epoch == 1 and i == 0):
                torch.cuda.profiler.start()
            if (epoch == 1 and i == 59):
                torch.cuda.profiler.stop()
                                      
        if args.enable_manual_profiling: torch.cuda.nvtx.range_push(f"step {i}")
        iters += 1
        dat_start = time.time()
        if args.enable_manual_profiling: torch.cuda.nvtx.range_push(f"data copy in {i}")
        # inp, tar = map(lambda x: x.to(device), data)
        # data0_device = data[0].get_device()
        # print('data[0].device: ', data0_device)

        mixes_uncut = data[0].to(device)
        # print('mixes_uncut.device: ', mixes_uncut.get_device())
        # print(type(mixes_uncut))
        sources_uncut = list(map(lambda x: x.to(device), data[1]))
#         for x in sources_uncut:
#             print('sources_uncut.device: ', x.get_device())
        
        if args.enable_manual_profiling: torch.cuda.nvtx.range_pop() # copy in  
        tr_start = time.time()
        # b_size = mixes_uncut.size(0)
                
        mixes = mixes_uncut
        sources = []
        for tensors in sources_uncut:
            tensor_seg = tensors
            sources.append(tensor_seg)
            
        mixes_lstm = mixes.unsqueeze(-1).float() # make dimension from (batch size, seq len) to (batch size, seq len, num of input)

        max_mix, _ = torch.max(torch.abs(mixes_lstm), dim=1, keepdims=True)
        if LOGSCALE == False:
            mixes_lstm_norm = torch.div(mixes_lstm, max_mix)
            mixes_lstm_norm = torch.nan_to_num(mixes_lstm_norm, nan=0.0)# switch nan to zero
        elif LOGSCALE == True:
            mixes_lstm_norm = mixes_lstm

        # sources_lstm = torch.stack([sources[0], sources[1], sources[2], sources[3]], dim=-1).float()
        sources_lstm = torch.stack([sources[0], sources[1], sources[2]], dim=-1).float()

        # max_src, _ = torch.max(torch.abs(sources_lstm), dim=1, keepdims=True)
        # sources_lstm_norm = torch.div(sources_lstm, max_src) # Don't do this! mix - src = 0. i.e. if you add all the sources, it become the mix!
        if LOGSCALE == False:
            sources_lstm_norm = torch.div(sources_lstm, max_mix)
            sources_lstm_norm = torch.nan_to_num(sources_lstm_norm, nan=0.0)# switch nan to zero
        elif LOGSCALE == True:
            sources_lstm_norm = sources_lstm

        # output = model(mixes_lstm_norm)

        # loss = criterion(output, sources_lstm_norm)
        
        # Backward and optimize
        optimizer.zero_grad()
        if args.enable_manual_profiling: torch.cuda.nvtx.range_push(f"forward")
        with autocast(args.enable_amp):
            output = model(mixes_lstm_norm) 
            #27032023(start)
            # print('output.shape= ',output.shape)
            # print('sources_lstm_norm.shape= ',sources_lstm_norm.shape)
            loss = criterion(output, sources_lstm_norm)
            # if CRITERION == 'rmse' or  CRITERION == 'rmsel1l2': loss = torch.sqrt(loss)

#             loss = 0
#             delta = 0.35

#             for j in range(output.shape[0]):
#                 bass_diff = torch.mean(torch.abs(output[j,:,0] - sources_lstm_norm[j,:,0]), dim=0)
#                 # drums_diff = torch.mean(torch.abs(output[j,:,1] - sources_lstm_norm[j,:,1]), dim=0)
#                 drums_diff = torch.mean(torch.abs(output[j,:,1] - sources_lstm_norm[j,:,1]), dim=0)

#             ###############
#                 # bass_diff = 0
#                 # drums_diff = 0
#                 # for k in range(output[j,:,0].shape[0]):
#                 #     bass_diff += torch.abs(output[j,k,0] - sources_lstm_norm[j,k,0])
#                 #     drums_diff += torch.abs(output[j,k,1] - sources_lstm_norm[j,k,1])
#             ###############

#                 loss_bass = 0.5*(bass_diff) if torch.abs(bass_diff) < delta else delta * torch.abs(bass_diff) - 0.5 * delta**2
#                 loss_drums = 0.5*(drums_diff) if torch.abs(drums_diff) < delta else delta * torch.abs(drums_diff) - 0.5 * delta**2
#                 loss += (loss_bass+loss_drums)
                # loss += (loss_drums)
            # print(output.shape, output)
            #27032023(finish)
            tr_loss.append(loss.item())
        if args.enable_manual_profiling: torch.cuda.nvtx.range_pop() #forward

        if(args.enable_amp):
            scaler.scale(loss).backward()
            # print('enable_amp, backward')
            if args.enable_manual_profiling: torch.cuda.nvtx.range_push(f"optimizer")
            scaler.step(optimizer)
            # print('enable_amp, optimizer step')
            if args.enable_manual_profiling: torch.cuda.nvtx.range_pop() # optimizer
            scaler.update()
        else:
            loss.backward()
            # print('no enable_amp, backward')
            if args.enable_manual_profiling: torch.cuda.nvtx.range_push(f"optimizer")
            optimizer.step()
            # print('no enable_amp, step')
            if args.enable_manual_profiling: torch.cuda.nvtx.range_pop() # optimizer 

        if args.enable_manual_profiling: torch.cuda.nvtx.range_pop() # step

        cumu_loss += loss.item()
        if WANDB == False:
            wandb.log({"loss": loss.item()}) #WandB
        
        #test(start)
        # print(i)
        # print(f'mixes_lstm_norm.shape = {mixes_lstm_norm.shape}')
        # print(f'mixes_lstm_norm,:,:,0].shape = {mixes_lstm_norm[0,:,0].shape}')
        # print(f'sources_lstm_norm.shape = {sources_lstm_norm.shape}')
        # print(f'sources_lstm_norm[0,:,0].shape = {sources_lstm_norm[0,:,0].shape}')
        # if WANDB == False:
        if WANDB == False and epoch % 10000 == 0:
            mixes_plt  = mixes_lstm_norm[0,:,0].clone().detach().cpu()
            bass_plt   = sources_lstm_norm[0,:,0].clone().detach().cpu()
            # drums_plt  = sources_lstm_norm[0,:,1].clone().detach().cpu()
            drums_plt  = sources_lstm_norm[0,:,1].clone().detach().cpu()
            # vocals_plt = sources_lstm_norm[0,:,2].clone().detach().cpu()
            other_plt  = sources_lstm_norm[0,:,2].clone().detach().cpu()

            bass_out_plt   = output[0,:,0].clone().detach().cpu()
            # drums_out_plt  = output[0,:,1].clone().detach().cpu()
            drums_out_plt  = output[0,:,1].clone().detach().cpu()
            # vocals_out_plt = output[0,:,2].clone().detach().cpu()
            other_out_plt  = output[0,:,2].clone().detach().cpu()

            
            if epoch == 0 and i == 0: 
                plt.plot(mixes_plt, label='mix', color='gray', alpha=0.5)# plot the 0th batch
                plt.plot(other_plt, label='others', color='green', alpha=0.5)
                plt.plot(bass_plt, label='bass', color='blue', alpha=0.5)
                plt.plot(drums_plt, label='drums', color='red', alpha=0.5)
                # plt.plot(vocals_plt, label='vocals', color='magenta', alpha=0.5)
                # plt.xlim([0, 4410]) # commented out for debugging
                plt.legend(loc='upper left')
                fig_original = wandb.Image(plt)
                wandb.log({'original': fig_original})
                plt.clf()

            wandb_png_pattern = os.path.join(WANDB_dir, 'media/images/trained_*.png')
            # Get list of file paths matching pattern
            wandb_png_paths = glob.glob(wandb_png_pattern)
            try:
                for wandb_png_path in wandb_png_paths:
                    os.remove(wandb_png_path)
            except OSError:
                pass

            plt.plot(other_plt, label='others original', color='gray', alpha=0.5)
            plt.plot(other_out_plt, label='others train', color='black', alpha=0.3)
            plt.plot(bass_plt, label='bass original', color='cyan', alpha=0.5)
            plt.plot(bass_out_plt, label='bass train', color='blue', alpha=0.5)
            plt.plot(drums_plt, label='drums original', color='magenta', alpha=0.5)
            plt.plot(drums_out_plt, label='drums train', color='red', alpha=0.5)
            # plt.plot(vocals_out_plt, label='vocals', color='magenta', alpha=0.5)
            plt.legend(loc='upper left')
            random_index = random.randint(0, len(other_plt)-441-1)
            plt.xlim([random_index,random_index+441]) # commented out for debugging
            fig_trained = wandb.Image(plt)
            wandb.log({'trained': fig_trained})
            plt.clf()

            mixes_wav = wandb.Audio(mixes_plt.float(), sample_rate=44100)
            bass_wav = wandb.Audio(bass_plt.float(), sample_rate=44100)
            drums_wav = wandb.Audio(drums_plt.float(), sample_rate=44100)
            other_wav = wandb.Audio(other_plt.float(), sample_rate=44100)
            bass_out_wav = wandb.Audio(bass_out_plt.float(), sample_rate=44100)
            drums_out_wav = wandb.Audio(drums_out_plt.float(), sample_rate=44100)
            other_out_wav = wandb.Audio(other_out_plt.float(), sample_rate=44100)
            
            
            wandb_wav_pattern = os.path.join(WANDB_dir, 'media/table/*.json')
            wandb_wav_paths = glob.glob(wandb_wav_pattern)
            try:
                for wandb_wav_path in wandb_wav_paths:
                    os.remove(wandb_wav_path)
            except OSError:
                pass

            sound_dat = [[mixes_wav, bass_wav, bass_out_wav, drums_wav, drums_out_wav, other_wav, other_out_wav]]
            sound_col = ['mixes', 'bass original', 'bass trained', 'drums original', 'drums trained', 'others original', 'others trained']

            sound_tab = wandb.Table(data=sound_dat, columns=sound_col)
            wandb.log({"sounds": sound_tab}, commit=False)
            # wandb.log({'mixes': mixes_wav})
            # Refresh the table and keep only the most recent values
            wandb.log({}, commit=True)
            
            #30032023(start)
            for i in range(len(mixes_plt)):
                wandb.log({"mixes": mixes_plt[i].numpy(), "bass original": bass_plt[i].numpy(), "bass trained": bass_out_plt[i].numpy(), "drums original": drums_plt[i].numpy(), "drums trained": drums_out_plt[i].numpy(), "others": other_plt[i].numpy(), "others trained": other_out_plt[i].numpy(), "time": i})
            #30032023(finish)

#####################################################################################            
            

#             mixes_dat = np.array([mixes_plt[y].numpy() for y in range(len(mixes_plt))])
#             bass_dat = np.array([bass_plt[y].numpy() for y in range(len(bass_plt))])
#             drums_dat = np.array([drums_plt[y].numpy() for y in range(len(drums_plt))])
#             other_dat = np.array([other_plt[y].numpy() for y in range(len(other_plt))])

#             orig_dat = np.array([mixes_dat, bass_dat, drums_dat, other_dat])

#             bass_out_dat = np.array([bass_out_plt[y].numpy() for y in range(len(bass_out_plt))])
#             drums_out_dat = np.array([drums_out_plt[y].numpy() for y in range(len(drums_out_plt))])
#             other_out_dat = np.array([other_out_plt[y].numpy() for y in range(len(other_out_plt))])
            
#             train_dat = np.array([bass_dat, bass_out_dat, drums_dat, drums_out_dat, other_dat, other_out_dat])


#             orig_plot = wandb.plot.line_series(
#                 xs=[i for i in range(len(mixes_plt))],
#                 ys=orig_dat,
#                 keys=["mixes", "bass", "drums", "other"],
#                 title="Originals",
#                 xname="time",
#             )

#             train_plot = wandb.plot.line_series(
#                 xs=[i for i in range(len(bass_plt))],
#                 ys=train_dat,
#                 keys=["bass_in", "bass_out", "drums_in", "drums_out", "other_in", "other_out"],
#                 title="Training Data",
#                 xname="time",
#             )

#             wandb.log({"Originals": orig_plot, "Training Data": train_plot}, commit=False)
            
#             # Refresh the table and keep only the most recent values
#             wandb.log({}, commit=True)
#####################################################################################            
            del mixes_plt, bass_plt, drums_plt, bass_out_plt, drums_out_plt, other_plt, other_out_plt
            del mixes_wav, bass_wav, drums_wav, other_wav, bass_out_wav, drums_out_wav, other_out_wav
            del sound_dat, sound_col, sound_tab
            #test(finish)
            
        tr_end = time.time()
        tr_time += tr_end - tr_start
        dat_time += tr_start - dat_start
        step_count += 1
    end_time = time.time()    
    if world_rank==0:
        if epoch%100000 == 0:
            logging.info('Time taken for epoch {} is {} sec, avg {} samples/sec'.format(epoch + 1, end_time-start_time,
                                                                                        # (step_count * params["global_batch_size"])/(end_time-start_time)))
                                                                                        (step_count * batch_size)/(end_time-start_time)))
            logging.info('  Avg train loss=%f'%np.mean(tr_loss))
        args.tboard_writer.add_scalar('Loss/train', np.mean(tr_loss), iters)
        args.tboard_writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], iters) #??
        args.tboard_writer.add_scalar('Avg iters per sec', step_count/(end_time-start_time), iters)
        
    return cumu_loss / len(dataloader)        

def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))
    
def train(local_rank=0, world_rank=0, world_size=0, distributed_run=False):


#     NUM_EPOCHS = 1 # I don't like this being here but hey, it's temporary.

#     print(f'Previous epoch size = {NUM_EPOCHS}')
#     if CHUNK_SIZE == 44100: NUM_EPOCHS = 30
#     elif CHUNK_SIZE == 441000: NUM_EPOCHS = 3
#     elif CHUNK_SIZE == 1323000: NUM_EPOCHS = 1
#     print(f'Seg_size = {CHUNK_SIZE} --> Updated epoch size = {NUM_EPOCHS}')
#     # Shorten the training. For now at least (finish)

    torch.backends.cudnn.benchmark = True # not sure I need this. I don't do convolution
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:%d'%local_rank)

    # get data loader
    logscale = LOGSCALE
    logging.info('rank %d, begin data loader init'%world_rank)
    readaudio = readAudio('/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Mixtures/Dev', '/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Sources/Dev', logscale, device)

    # Reduce time for now (start)
    # valiaudio = readAudio('/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Mixtures/Test', '/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Sources/Test', logscale = True, device)
    # Reduce time for now (finish)

    NUM_SAMPLE = readaudio.__len__()
    NUM_SAMPLE = 1 # debugging
    print(f'number of samples = {NUM_SAMPLE}')

    # Reduce time for now (start)
    # NUM_VALILE = readaudio.__len__()
    # print(f'number of validation samples = {NUM_VALILE}')
    # Reduce time for now (finish)

    audios = readaudio.__getitem__(NUM_SAMPLE, CHUNK_SIZE) #WandB
    
    # #test(start)
    # for i, a in enumerate(audios):
    #     print("cunt ", i, a[0].shape, a[1][0].shape, a[1][1].shape, a[1][2].shape, a[1][3].shape)
    # #test(finish)
    
    
    # print("Size of audios: " + str(audios.__sizeof__()/1024/1024) + "MB")
    size_in_bytes = sys.getsizeof(audios)
    size_in_mb = size_in_bytes / (1024**2)
    print(f"Size of audios: {size_in_mb:.2f} MB")
    global BATCH_SIZE # I don't like this... But hey, then give me a better solution to reasign 
    print(f'BATCH_SIZE = {BATCH_SIZE}, len(audios) = {len(audios)}.')
    if BATCH_SIZE > len(audios): 
        BATCH_SIZE = len(audios)
        print(f'BATCH_SIZE ({BATCH_SIZE}) > len(audios) ({len(audios)}). Update the BATCH_SIZE to {BATCH_SIZE}')

    train_sampler = DistributedSampler(audios) if distributed_run else None
    dataloader = None
    if not args.enable_benchy:    
        dataloader = DataLoader(audios, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler, worker_init_fn=worker_init, persistent_workers=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    else:
        from benchy.torch import BenchmarkDataLoader
        dataloader = BenchmarkDataLoader(audios, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler, worker_init_fn=worker_init, persistent_workers=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    del audios # free memory?

    logging.info('rank %d, data loader initialized'%(world_rank))    

    # Create an instance of the LSTMModel class
    print('getting the model.....')
    # model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    # model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE_LONG, HIDDEN_SIZE_SHORT, NUM_LAYERS_LONG, NUM_LAYERS_SHORT, OUTPUT_SIZE, DROPOUT).to(device)
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, DROPOUT).to(device)
    # print(summary(model))
    scaler=None
    if args.enable_amp:
        scaler = GradScaler()


    # For now, assume we are not using distributed data parallel
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)#torch.optim.Optimizer([])
    criterion = nn.L1Loss()

    if OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.5)
    elif OPTIMIZER == "sgd" and CRITERION == 'rmsel1l2':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.5, weight_decay=0.01)
    elif OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if CRITERION == "mse" or CRITERION == "rmse" or CRITERION == "rmsel1l2":
        criterion = nn.MSELoss()
    elif CRITERION == "mae":
        criterion = nn.L1Loss()

    # start training
    if world_rank==0:
        logging.info("Starting Training Loop...")

    iters = 0
    t1 = time.time()
    
    # learning rate scheduler #26032023
    # scheduler = StepLR(optimizer, step_size=200, gamma=0.1) # every 1 epoch, lower the lr by 10% 
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.1) # every 1 epoch, lower the lr by 10% 

    for epoch in range(NUM_EPOCHS): #WandB
        model.train()
        if epoch%10000 == 0: print(f'train epoch: {epoch}/{NUM_EPOCHS}')
        avg_loss = train_epoch(device, model, dataloader, criterion, optimizer, epoch, scaler, iters, BATCH_SIZE)
        if epoch%10000 == 0: print(f'loss: {avg_loss}, epoch: {epoch}')

        args.tboard_writer.flush()
        if world_rank==0: args.tboard_writer.flush()
        try:
            os.remove(
  f"/pscratch/sd/h/hsko/jupyter/ML/LSTM/model_gpu/hierarchy/one_sample/cpuDataLoader/model_opt_{OPTIMIZER}_cri_{CRITERION}_eph_{epoch-1}_batch_{BATCH_SIZE}_scheduledlr_{LEARNING_RATE}_seg_{CHUNK_SIZE}_hidden_{HIDDEN_SIZE}_layer_{len(HIDDEN_SIZE)}_dropout_{DROPOUT}_logscale{LOGSCALE}.pth"
            )
        except OSError:
            pass

        torch.save(model.state_dict(), f"/pscratch/sd/h/hsko/jupyter/ML/LSTM/model_gpu/hierarchy/one_sample/cpuDataLoader/model_opt_{OPTIMIZER}_cri_{CRITERION}_eph_{epoch}_batch_{BATCH_SIZE}_scheduledlr_{LEARNING_RATE}_seg_{CHUNK_SIZE}_hidden_{HIDDEN_SIZE}_layer_{len(HIDDEN_SIZE)}_dropout_{DROPOUT}_logscale{LOGSCALE}.pth")
        # Update the learning rate scheduler #26032023
        # scheduler.step() #comment out for debugging
        #27032023(start)
        if epoch%10000 == 0: 
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f'{name}: {param.grad.norm(2).item()}')
        #27032023(finish)

    del dataloader # free memory
    t2 = time.time()
    tottime = t2 - t1    
    return tottime        

def main(local_rank=0, world_rank=0, world_size=0, distributed_run=False):

    tottime = train(local_rank, world_rank, world_size, distributed_run)
    
    return tottime

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable_manual_profiling", action='store_true', help='enable manual nvtx ranges and profiler start/stop calls')
    parser.add_argument("--enable_benchy", action='store_true', help='enable benchy tool usage')
    parser.add_argument("--enable_amp", action='store_true', help='enable automatic mixed precision')
    parser.add_argument("--num_epochs", default=30, type=int, help='number of epochs to run')
    parser.add_argument("--batch_size", default=1, type=int, help='batch size')
    parser.add_argument("--learning_rate", default=0.01, type=float, help='learning rate')
    parser.add_argument("--dropout", default=0, type=float, help='dropout')
    parser.add_argument("--num_data_workers", default=None, type=int, help='number of data workers for data loader')
    parser.add_argument("--hidden_size", default=[4, 16, 32], type=int, nargs='+', help='number of hidden features. e.g. [4, 16, 32]')
    parser.add_argument("--chunk_size", default=44100, type=int, help='size of one segment')
    parser.add_argument("--num_chunks", default=None, type=int, help='number of chunks for testing')
    parser.add_argument("--logscale", default=False, type=bool, help='train the sound in log scale?')
    parser.add_argument("--nowandb", default=False, type=bool, help='do you want wandb?')
    args = parser.parse_args()
    
    if (args.enable_benchy and args.enable_manual_profiling):
        raise RuntimeError("Enable either benchy profiling or manual profiling, not both.")

    # Define hyperparameters
    INPUT_SIZE = 1
    HIDDEN_SIZE = args.hidden_size
    OUTPUT_SIZE = 3
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    CHUNK_SIZE = args.chunk_size
    OPTIMIZER = 'adam'
    CRITERION = 'mse'
    # CRITERION = 'rmsel1l2' #26032023
    DROPOUT = args.dropout
    PERSISTENT_WORKERS = True
    PIN_MEMORY = torch.cuda.is_available()
    LOGSCALE = False
    WANDB = False
    if args.logscale:
        LOGSCALE = True
    if args.nowandb:
        WANDB = True
    WANDB_dir = None
    

    if args.dropout != 0 and len(args.hidden_size) == 1:
        print(f'dropout = {args.dropout} and num_layers = {len(args.hidden_size)}.')
        print(f'When num_layers = 1, you need dropout = 0')
        DROPOUT = 0
        print(f'Changed dropout to {DROPOUT}')
    NUM_WORKERS = 8
    if args.num_data_workers == None: 
        NUM_WORKERS = 0
        PERSISTENT_WORKERS = False
        PIN_MEMORY = False
    # elif args.num_data_workers == 0: 
    #     NUM_WORKERS = args.num_data_workers
    #     PERSISTENT_WORKERS = False
    #     PIN_MEMORY = False
    else: 
        NUM_WORKERS = args.num_data_workers
        PERSISTENT_WORKERS = True
        PIN_MEMORY = torch.cuda.is_available()

    
    
    # PIN_MEMORY = torch.cuda.is_available()
    # print(f'NUM_WORKERS = {NUM_WORKERS}')

    ##########################
    world_size = 1
    distributed_run = False #may need to use YAML file for better organization
    if 'WORLD_SIZE' in os.environ:
        distributed_run = int(os.environ['WORLD_SIZE']) > 1
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = 1
    
    world_rank = 0
    local_rank = 0
    if distributed_run:
        torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
        world_rank = torch.distributed.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
    ##########################

    
    # Set up directory  
    # baseDir = '/pscratch/sd/h/hsko/jupyter/ML/LSTM/summary_gpu'
    # expDir = os.path.join(baseDir, '/%dGPU/'%(world_size)+str(run_num)+'/')
    expDir = os.path.join('/pscratch/sd/h/hsko/jupyter/ML/LSTM/hierarchy/one_sample/summary_gpu/%dGPU'%(world_size)+'/')
    # print(f'baseDir = {baseDir}')
    print(f'expDir = {expDir}')
    if  world_rank==0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out_1.log'))
        # params.log()
        args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, 'logs/'))


    api_key = '7a7346b9e3ee9dfebc6fc65da44ef3644f03298a'
    if WANDB == False:
        subprocess.call(f'wandb login {api_key}', shell=True)


    config={
        "architecture": "LSTM",
        "dataset": "DSD100",
        "dataloaded": "cpu",
        "NUM_WORKERS": NUM_WORKERS,
        "PERSISTENT_WORKERS": PERSISTENT_WORKERS,
        "PIN_MEMORY": PIN_MEMORY,
        "INPUT_SIZE" : 1,
        "HIDDEN_SIZE" : args.hidden_size,
        "OUTPUT_SIZE" : 4,
        "BATCH_SIZE" : BATCH_SIZE,
        "LEARNING_RATE" : 0.01,
        "NUM_EPOCHS" : args.num_epochs,
        "CHUNK_SIZE" : args.chunk_size,
        "OPTIMIZER" : OPTIMIZER,
        "CRITERION" : CRITERION,
        "DROPOUT" : DROPOUT, 
        "LOGSCALE": LOGSCALE,
    }


    if WANDB == False:
        wandb.init(
        # set the wandb project where this run will be logged
                                    # project=f"cpu_opt_{OPTIMIZER}_momentum_05_cri_{CRITERION}_eph_{NUM_EPOCHS}_batch_{BATCH_SIZE}_scheduledlr_{LEARNING_RATE}_seg_{CHUNK_SIZE}_hidden_{HIDDEN_SIZE}_layer_{NUM_LAYERS}_dropout_{DROPOUT}_logscale{LOGSCALE}",
                                    project=f"cpu_opt_{OPTIMIZER}_cri_{CRITERION}_eph_{NUM_EPOCHS}_batch_{BATCH_SIZE}_lr_{LEARNING_RATE}_seg_{CHUNK_SIZE}_hidden_{HIDDEN_SIZE}_dropout_{DROPOUT}_logscale{LOGSCALE}",

        # track hyperparameters and run metadata
        config=config
        )
        WANDB_dir = wandb.run.dir

    tottime = main(local_rank, world_rank, world_size, distributed_run)
    print(f'It took {tottime}')
