import sys
import pandas as pd
import numpy as np
import time
from model import LSTM_TrajGAN

if __name__ == '__main__':
    n_epochs = int(sys.argv[1])
    n_batch_size = int(sys.argv[2])
    n_sample_interval = int(sys.argv[3])

    latent_dim = 50
    max_length = 133

    keys = ['sub_acc', 'sub_distance', 'sub_speed', 'distance_sub_to_pre', 'relative_speed',
            'mask']
    vocab_size = {key: 1 for key in keys}

    gan = LSTM_TrajGAN(latent_dim, keys, vocab_size, max_length)

    # 记录开始时间
    start_time = time.time()
    print("开始时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

    gan.train(epochs=n_epochs, batch_size=n_batch_size, sample_interval=n_sample_interval)

    # 记录结束时间
    end_time = time.time()
    print("结束时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))

    # 计算并打印运行总时长
    total_time = end_time - start_time
    print("运行总时长: {:.2f} 秒".format(total_time))