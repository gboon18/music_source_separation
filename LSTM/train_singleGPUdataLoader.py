#!/usr/bin/env python
# import nbimporter # pip install import_ipynb
# import sys
# sys.path.append("/global/u2/h/hsko/jupyter/ML/LSTM")
from dataset_singleGPUdataLoader import readAudio
from model import LSTMModel
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

# from torchinfo import summary
# import traceback
# import gc #empty memory

import os
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
        # for x in sources_uncut:
        #     print('sources_uncut.device: ', x.get_device())
        
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
        mixes_lstm_norm = torch.div(mixes_lstm, max_mix)
        mixes_lstm_norm = torch.nan_to_num(mixes_lstm_norm, nan=0.0)# switch nan to zero

        sources_lstm = torch.stack([sources[0], sources[1], sources[2], sources[3]], dim=-1).float()

        # max_src, _ = torch.max(torch.abs(sources_lstm), dim=1, keepdims=True)
        # sources_lstm_norm = torch.div(sources_lstm, max_src) # Don't do this! mix - src = 0. i.e. if you add all the sources, it become the mix!
        sources_lstm_norm = torch.div(sources_lstm, max_mix)
        sources_lstm_norm = torch.nan_to_num(sources_lstm_norm, nan=0.0)# switch nan to zero

        # output = model(mixes_lstm_norm)

        # loss = criterion(output, sources_lstm_norm)

        # Backward and optimize
        optimizer.zero_grad()
        if args.enable_manual_profiling: torch.cuda.nvtx.range_push(f"forward")
        with autocast(args.enable_amp):
            output = model(mixes_lstm_norm) 
            loss = criterion(output, sources_lstm_norm)
            tr_loss.append(loss.item())
        if args.enable_manual_profiling: torch.cuda.nvtx.range_pop() #forward

        if(args.enable_amp):
            scaler.scale(loss).backward()
            if args.enable_manual_profiling: torch.cuda.nvtx.range_push(f"optimizer")
            scaler.step(optimizer)
            if args.enable_manual_profiling: torch.cuda.nvtx.range_pop() # optimizer
            scaler.update()
        else:
            loss.backward()
            if args.enable_manual_profiling: torch.cuda.nvtx.range_push(f"optimizer")
            optimizer.step()
            if args.enable_manual_profiling: torch.cuda.nvtx.range_pop() # optimizer 

        if args.enable_manual_profiling: torch.cuda.nvtx.range_pop() # step

        cumu_loss += loss.item()
        wandb.log({"loss": loss.item()}) #WandB
            
        tr_end = time.time()
        tr_time += tr_end - tr_start
        dat_time += tr_start - dat_start
        step_count += 1
    end_time = time.time()    
    if world_rank==0:
        logging.info('Time taken for epoch {} is {} sec, avg {} samples/sec'.format(epoch + 1, end_time-start_time,
                                                                                    # (step_count * params["global_batch_size"])/(end_time-start_time)))
                                                                                    (step_count * batch_size)/(end_time-start_time)))
        logging.info('  Avg train loss=%f'%np.mean(tr_loss))
        args.tboard_writer.add_scalar('Loss/train', np.mean(tr_loss), iters)
        args.tboard_writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], iters) #??
        args.tboard_writer.add_scalar('Avg iters per sec', step_count/(end_time-start_time), iters)
        # args.tboard_writer.flush()

    return cumu_loss / len(dataloader)

def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))
    
# def train(config=None, local_rank=0, world_rank=0, world_size=0, distributed_run=False):
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
    logging.info('rank %d, begin data loader init'%world_rank)
    readaudio = readAudio('/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Mixtures/Dev', '/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100/Sources/Dev', device)

    for sample_index in range(5):
        print(f'sampling {sample_index} out of {5}')
        sample_index = sample_index*5 # 0, 5, 10,..., 45
        audios = readaudio.__getitem__(sample_index, sample_index+5, CHUNK_SIZE) #Not WandB
        print("Size of audios: " + str(audios.__sizeof__()/1024/1024) + "MB")
        global BATCH_SIZE # I don't like this... But hey, then give me a better solution to reasign 
        print(f'BATCH_SIZE = {BATCH_SIZE}, len(audios) = {len(audios)}.')
        if BATCH_SIZE > len(audios): 
            BATCH_SIZE = len(audios)
            print(f'BATCH_SIZE ({BATCH_SIZE}) > len(audios) ({len(audios)}). Update the BATCH_SIZE to {BATCH_SIZE}')
        
        train_sampler = DistributedSampler(audios) if distributed_run else None
        dataloader = None
        if not args.enable_benchy:    
            dataloader = DataLoader(audios, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler, worker_init_fn=worker_init, persistent_workers=PERSISTENT_WORKERS, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        else:
            from benchy.torch import BenchmarkDataLoader
            dataloader = BenchmarkDataLoader(audios, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler, worker_init_fn=worker_init, persistent_workers=PERSISTENT_WORKERS, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        del audios # free memory?

        logging.info('rank %d, data loader initialized'%(world_rank))    

        # Create an instance of the LSTMModel class
        print('getting the model.....')
        model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
        # print(summary(model))
        scaler=None
        if args.enable_amp:
            scaler = GradScaler()


        # For now, assume we are not using distributed data parallel
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)#torch.optim.Optimizer([])
        criterion = nn.L1Loss()

        if OPTIMIZER == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        elif OPTIMIZER == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        if CRITERION == "mse":
            criterion = nn.MSELoss()
        elif CRITERION == "mae":
            criterion = nn.L1Loss()

        # start training
        if world_rank==0:
            logging.info("Starting Training Loop...")

        iters = 0
        t1 = time.time()

        for epoch in range(NUM_EPOCHS): 
            model.train()
            print(f'train epoch: {epoch}/{NUM_EPOCHS} out of {sample_index}/5 samples')
            avg_loss = train_epoch(device, model, dataloader, criterion, optimizer, epoch, scaler, iters, BATCH_SIZE)
            print(f'loss: {avg_loss}, epoch: {epoch}')

            if world_rank==0: args.tboard_writer.flush()
            torch.save(model.state_dict(), f"/pscratch/sd/h/hsko/jupyter/ML/LSTM/model_gpu/batch_run/singleGPUdataLoader/model_opt_{OPTIMIZER}_cri_{CRITERION}_eph_{epoch}_batch_{BATCH_SIZE}_lr_{LEARNING_RATE}_seg_{CHUNK_SIZE}_hidden_{HIDDEN_SIZE}_layer_{NUM_LAYERS}_dropout_{DROPOUT}.pth")

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
    parser.add_argument("--num_data_workers", default=None, type=int, help='number of data workers for data loader')
    parser.add_argument("--hidden_size", default=4, type=int, help='number of hidden features')
    parser.add_argument("--num_layers", default=1, type=int, help='number of layers (1, 2, 3)')
    parser.add_argument("--chunk_size", default=44100, type=int, help='size of one segment') #. (44100, 441000, 1323000). Decides the number of epochs (30, 3, 1) for now')
    parser.add_argument("--num_chunks", default=None, type=int, help='number of chunks for testing')
    args = parser.parse_args()
    
    if (args.enable_benchy and args.enable_manual_profiling):
        raise RuntimeError("Enable either benchy profiling or manual profiling, not both.")
    
    # Define hyperparameters
    INPUT_SIZE = 1
    HIDDEN_SIZE = args.hidden_size
    NUM_LAYERS = args.num_layers
    OUTPUT_SIZE = 4
z    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    CHUNK_SIZE = args.chunk_size
    OPTIMIZER = 'sgd'
    CRITERION = 'mse'
    DROPOUT = 0

    # When we upload the dataset directly to the GPU.
    # There is no need for the number of workers to work between cpu and gpu
    # So no need for persistent worker
    # No need to pin memory
    NUM_WORKERS = 0
    if args.num_data_workers == None: 
        NUM_WORKERS = 0
        PERSISTENT_WORKERS = False
        PIN_MEMORY = False
    else: 
        NUM_WORKERS = args.num_data_workers
        PERSISTENT_WORKERS = False
        PIN_MEMORY = False

    
    
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
    expDir = os.path.join('/pscratch/sd/h/hsko/jupyter/ML/LSTM/batch_run/summary_gpu/%dGPU'%(world_size)+'/')
    # print(f'baseDir = {baseDir}')
    print(f'expDir = {expDir}')
    if  world_rank==0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out_1.log'))
        # params.log()
        args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, 'logs/'))


    api_key = '7a7346b9e3ee9dfebc6fc65da44ef3644f03298a'
    subprocess.call(f'wandb login {api_key}', shell=True)

    config={
        "architecture": "LSTM",
        "dataset": "DSD100",
        "dataloaded": "gpu",
        "INPUT_SIZE" : 1,
        "HIDDEN_SIZE" : args.hidden_size,
        "NUM_LAYERS" : args.num_layers,
        "OUTPUT_SIZE" : 4,
        "BATCH_SIZE" : BATCH_SIZE,
        "LEARNING_RATE" : 0.01,
        "NUM_EPOCHS" : args.num_epochs,
        "CHUNK_SIZE" : args.chunk_size,
        "OPTIMIZER" : 'sgd',
        "CRITERION" : 'mse',
        "DROPOUT" : 0, 
    }

    wandb.init(
    # set the wandb project where this run will be logged
    project=f"sound_cpudataloader_opt_{OPTIMIZER}_cri_{CRITERION}_eph_{NUM_EPOCHS}_batch_{BATCH_SIZE}_lr_{LEARNING_RATE}_seg_{CHUNK_SIZE}_hidden_{HIDDEN_SIZE}_layer_{NUM_LAYERS}_dropout_{DROPOUT}",

    # track hyperparameters and run metadata
    config=config
    )

    tottime = main(local_rank, world_rank, world_size, distributed_run)
    print(f'It took {tottime}')
