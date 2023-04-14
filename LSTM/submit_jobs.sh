#!/bin/bash

for chunk_size in 44100 441000 1323000
do
    for hidden_size in 1 4 100 400 1000
    do
        # for num_layers in 2 3 4 5
        for num_layers in 1
        do
            for learning_rate in 0.01 0.001
            do
                for batch_size in 50, 100, 1000
                do
                    for dropout in 0 0.2
                    do
                        if [ ${chunk_size} -eq 44100 ]; then num_epochs=30
                        elif [ ${chunk_size} -eq 441000 ]; then num_epochs=3
                        elif [ ${chunk_size} -eq 1323000 ]; then num_epochs=1
                        fi
                        echo "chunk_size = ${chunk_size}, num_epochs = ${num_epochs}, hidden_size = ${hidden_size}, num_layers = ${num_layers}, learning_rate = ${learning_rate}"
                        sbatch -n 1 ./submit_cpuDataLoader.sh  --num_data_workers 8 --chunk_size ${chunk_size} --num_epochs ${num_epochs} --hidden_size ${hidden_size} --num_layers ${num_layers} --learning_rate ${learning_rate} --batch_size ${batch_size}  --dropout ${dropout} --enable_amp
                    done
                done
            done  
        done
    done
done
