VQ-VAE with UNet encoder/decoder.

- UNet takes audio .wav file in time domain.
- At the highest latent space of the UNet (encoded), [VQ-VAE](https://doi.org/10.48550/arXiv.1711.00937) is used for the embedding.
- The embedding vector is put to UNet decoder and produce the otput .wav file in time domain.

- For the model to be smaller, input files are downsampled from 44.1kHz to 16kHz and chunked to 2 seconds.

- Example run command:
UNet + VQ-VAE: 
```
python VQ-VAE-UNet.py --enable_amp --len 2 --valen 2 --batch_size 4 --num_epoch 100000 --unet_only --datareamp 0.7 1.0 --l1l2 l1 --cylr 5e-6 1e-4
```

UNet only: 
```
python VQ-VAE-UNet.py --enable_amp --len 2 --valen 2 --batch_size 4 --num_epoch 100000 --unet_only --datareamp 0.7 1.0 --l1l2 l1 --cylr 5e-6 1e-4 --unet_only
```

- The current model takes the mixed audio .wav and fource sources audio .wav (bass, drum, vocal, other)
- It only takes left channel and consider it as mono channel
- Update will come soon: stereo, transformer, etc.
