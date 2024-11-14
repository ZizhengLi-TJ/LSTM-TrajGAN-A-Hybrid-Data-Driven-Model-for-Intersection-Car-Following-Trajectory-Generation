import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model import LSTM_TrajGAN
# 加载模型架构和权重
def load_model(model_json_path, model_weights_path):
    with open(model_json_path, "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(model_weights_path)
    return model

# 实例化 LSTM_TrajGAN 类
latent_dim = 50
max_length = 133

keys = ['sub_acc', 'sub_distance', 'sub_speed', 'distance_sub_to_pre', 'relative_speed', 'mask']
vocab_size = {key: 1 for key in keys}

gan = LSTM_TrajGAN(latent_dim, keys, vocab_size, max_length)

# 加载判别器模型
gan.discriminator = load_model("params/D_model.json", "lzz_training_params/D_model_300.h5")

# 评估判别器分数
real_npy = "data/last_test.npy"
synthetic_npy = "data/result_apply.npy"
gan.evaluate_discriminator(real_npy, synthetic_npy)
