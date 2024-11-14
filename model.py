import tensorflow as tf
import keras
import numpy as np
import random
import csv

random.seed(2020)
np.random.seed(2020)
tf.random.set_random_seed(2020)

from keras.layers import Input, Add, Average, Dense, LSTM, Lambda, TimeDistributed, Concatenate, Embedding
from keras.initializers import he_uniform
from keras.regularizers import l1
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

from losses import d_bce_loss, trajLoss
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
class LSTM_TrajGAN():
    def __init__(self, latent_dim, keys, vocab_size, max_length):
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        self.keys = keys
        self.vocab_size = vocab_size

        
        self.x_train = None
        
        # Define the optimizer
        self.optimizer = Adam(0.001, 0.5)

        # Build the trajectory generator
        self.generator = self.build_generator()

        # The trajectory generator takes real trajectories and noise as inputs
        noise = Input(shape=(self.latent_dim,), name='input_noise')
        inputs = []
        for idx, key in enumerate(self.keys):
            i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
            inputs.append(i)
        inputs.append(noise)
        
        # The trajectory generator generates synthetic trajectories
        gen_trajs = self.generator(inputs)
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=d_bce_loss(gen_trajs[-1]), optimizer=self.optimizer, metrics=['accuracy'])

        # The combined model only trains the trajectory generator
        self.discriminator.trainable = False

        # The discriminator takes generated trajectories as input and makes predictions
        pred = self.discriminator(gen_trajs[:-1])

        # The combined model (combining the generator and the discriminator)
        self.combined = Model(inputs, pred)
        self.combined.compile(loss=trajLoss(inputs, gen_trajs), optimizer=self.optimizer)
        
        C_model_json = self.combined.to_json()
        with open("params/C_model.json", "w") as json_file:
            json_file.write(C_model_json)
            
        G_model_json = self.generator.to_json()
        with open("params/G_model.json", "w") as json_file:
            json_file.write(G_model_json)
        
        D_model_json = self.discriminator.to_json()
        with open("params/D_model.json", "w") as json_file:
            json_file.write(D_model_json)

    def build_discriminator(self):
        
        # Input Layer
        inputs = []
        embeddings = []

        for key in self.keys:
            if key == 'mask':
                continue
            i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
            d = Dense(units=64, activation='relu', use_bias=True, kernel_initializer=he_uniform(seed=1),
                      name=f'emb_{key}')
            e = TimeDistributed(d)(i)
            inputs.append(i)
            embeddings.append(e)


        # 特征融合层
        concat_input = Concatenate(axis=2)(embeddings)
        d = Dense(units=100, use_bias=True, activation='relu', kernel_initializer=he_uniform(seed=1), name='emb_fused')
        fused_output = TimeDistributed(d)(concat_input)

        # Adding additional Dense layers for better feature fusion
        dense_fusion = Dense(256, activation='relu')(fused_output)
        dense_fusion = Dense(128, activation='relu')(dense_fusion)

        # LSTM建模层
        lstm_cell = LSTM(units=100, recurrent_regularizer='l1')(dense_fusion)

        # 输出层
        sigmoid = Dense(1, activation='sigmoid')(lstm_cell)

        return Model(inputs=inputs, outputs=sigmoid)


    def build_generator(self):
        
        # # Input Layer

        inputs = []

        # Embedding Layer
        embeddings = []
        noise = Input(shape=(self.latent_dim,), name='input_noise')
        mask = Input(shape=(self.max_length, 1), name='input_mask')
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                inputs.append(mask)
                continue
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=64, activation='relu', use_bias=True, kernel_initializer=he_uniform(seed=1),
                          name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            inputs.append(i)
            embeddings.append(e)
        inputs.append(noise)

        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(concat_input)
        d = Dense(units=100, use_bias=True, activation='relu', kernel_initializer=he_uniform(seed=1),
                  name='emb_trajpoint')
        dense_outputs = [d(Concatenate(axis=1)([x, noise])) for x in unstacked]
        emb_traj = Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)




        # Adding additional Dense layers for better feature fusion
        dense_fusion = Dense(256, activation='relu')(emb_traj)
        dense_fusion = Dense(128, activation='relu')(dense_fusion)

        # LSTM Modeling Layer (many-to-many)
        lstm_cell = LSTM(units=100, batch_input_shape=(None, self.max_length, 128), return_sequences=True,
                         recurrent_regularizer=l1(0.02))(dense_fusion)




        # Outputs
        outputs = []
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                output_mask = Lambda(lambda x: x)(mask)
                outputs.append(output_mask)
            else:
                output = TimeDistributed(Dense(1, activation='tanh'), name=f'output_{key}')(lstm_cell)
                outputs.append(output)

        return Model(inputs=inputs, outputs=outputs)


    def train(self, epochs=200, batch_size=256, sample_interval=10):
    # def train(self, epochs=10, batch_size=256, sample_interval=5):
        
        # Training data
        x_train = np.load('data/last_train.npy',allow_pickle=True)
        self.x_train = x_train

        # Padding zero to reach the maxlength
        X_train = [pad_sequences(f, self.max_length, padding='pre', dtype='float64') for f in x_train]
        self.X_train = X_train
        


        # Open a CSV file for writing
        csv_file = 'losses.csv'
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['Epoch', 'D Loss', 'G Loss'])

            for epoch in range(1, epochs + 1):
                # Select a random batch of real trajectories
                idx = np.random.randint(0, X_train[0].shape[0], batch_size)

                # Ground truths for real trajectories and synthetic trajectories
                real_bc = np.ones((batch_size, 1))
                syn_bc = np.zeros((batch_size, 1))

                # real_trajs_bc = [X_train[self.keys.index(key)][idx] for key in self.keys]
                real_trajs_bc = [X_train[self.keys.index(key)][idx].reshape(batch_size, self.max_length, 1) for key in
                                 self.keys]

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                real_trajs_bc.append(noise)

                gen_trajs_bc = self.generator.predict(real_trajs_bc)

                # Train the discriminator
                # No mask and noise are used
                d_loss_real = self.discriminator.train_on_batch(real_trajs_bc[:len(self.keys) - 1], real_bc)
                d_loss_syn = self.discriminator.train_on_batch(gen_trajs_bc[:len(self.keys) - 1], syn_bc)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_syn)

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                real_trajs_bc[-1] = noise
                g_loss = self.combined.train_on_batch(real_trajs_bc, real_bc)

                print("[%d/%d] D Loss: %f | G Loss: %f" % (epoch, epochs, d_loss[0], g_loss))

                # Write the loss values to the CSV file
                writer.writerow([epoch, d_loss[0], g_loss])

                # Print and save the losses/params
                if epoch % sample_interval == 0:
                    self.save_checkpoint(epoch)
                    print('Model params saved to the disk.')

        print('Training complete. Loss values saved to losses.csv')

    
    def save_checkpoint(self, epoch):
        self.combined.save_weights("lzz_training_params/C_model_"+str(epoch)+".h5")
        self.generator.save_weights("lzz_training_params/G_model_"+str(epoch)+".h5")
        self.discriminator.save_weights("lzz_training_params/D_model_"+str(epoch)+".h5")
        print("Training Params Saved")

    def evaluate_discriminator(self, real_npy, synthetic_npy):
        # 从npy文件中读取真实和合成轨迹
        real_trajs = np.load(real_npy, allow_pickle=True)
        synthetic_trajs = np.load(synthetic_npy, allow_pickle=True)

        # 对数据进行必要的预处理，确保其形状与模型输入匹配
        real_trajs_processed = [
            pad_sequences(np.array(real_trajs[i]), self.max_length, padding='pre', dtype='float64').reshape(-1,
                                                                                                            self.max_length,
                                                                                                            1) for i in
            range(len(self.keys)) if self.keys[i] != 'mask']
        synthetic_trajs_processed = [
            pad_sequences(np.array(synthetic_trajs[i]), self.max_length, padding='pre', dtype='float64').reshape(-1,
                                                                                                                 self.max_length,
                                                                                                                 1) for
            i in range(len(self.keys)) if self.keys[i] != 'mask']

        # 计算判别器分数
        real_scores = self.discriminator.predict(real_trajs_processed[:len(self.keys) - 1])
        synthetic_scores = self.discriminator.predict(synthetic_trajs_processed[:len(self.keys) - 1])
        real_scores = np.array(real_scores).flatten()
        synthetic_scores = np.array(synthetic_scores).flatten()

        # 将分数和类型组合到一个字典中
        scores_dict = {
            'Score_real': real_scores,
            'Score_synthetic': synthetic_scores,
        }

        # 创建DataFrame
        scores_df = pd.DataFrame(scores_dict)

        # 保存DataFrame到CSV文件
        scores_df.to_csv('score1.csv', index=False)


        # # 绘制分数分布图
        # plt.figure(figsize=(12, 6))
        # sns.kdeplot(real_scores.flatten(), label='真实轨迹', color='blue', fill=True, alpha=0.5)
        # sns.kdeplot(synthetic_scores.flatten(), label='合成轨迹', color='orange', fill=True, alpha=0.5)
        # plt.title('判别器分数分布')
        # plt.xlabel('判别器分数')
        # plt.ylabel('密度')
        # plt.legend()
        # plt.show()
        #
        # # 计算并打印真实和合成分数的均值和标准差
        # real_mean = np.mean(real_scores)
        # real_std = np.std(real_scores)
        # synthetic_mean = np.mean(synthetic_scores)
        # synthetic_std = np.std(synthetic_scores)
        # print(f"真实轨迹: 均值 = {real_mean}, 标准差 = {real_std}")
        # print(f"合成轨迹: 均值 = {synthetic_mean}, 标准差 = {synthetic_std}")
        #
        # # 绘制箱型图
        # scores_data = {
        #     '类型': ['真实轨迹'] * len(real_scores.flatten()) + ['合成轨迹'] * len(synthetic_scores.flatten()),
        #     '分数': np.concatenate([real_scores.flatten(), synthetic_scores.flatten()])
        # }
        # scores_df = pd.DataFrame(scores_data)
        #
        # plt.figure(figsize=(12, 6))
        # sns.boxplot(x='类型', y='分数', data=scores_df, palette=['blue', 'orange'])
        # plt.title('判别器分数箱型图')
        # plt.xlabel('轨迹类型')
        # plt.ylabel('判别器分数')
        # plt.show()
# 这段代码定义了一个名为 LSTM_TrajGAN 的类，它代表了一个基于 LSTM 和 GAN 的轨迹数据生成模型。类中包含了构建生成器 (Generator)、鉴别器 (Discriminator) 的方法，以及一个训练函数，用于训练生成合成轨迹的模型。以下是对代码的详细解释：
# 初始化和设置
# 类构造函数接收多个参数，包括潜在空间的维度 (latent_dim)、序列的最大长度 (max_length)、特征键 (keys)、词汇大小 (vocab_size)、经纬度中心点 (lat_centroid 和 lon_centroid) 以及缩放因子 (scale_factor)。
# 设置了随机种子，以确保结果的可重现性。
# 定义了优化器为 Adam。
# 构建生成器 (Generator)
# build_generator 方法创建了生成器模型，该模型使用 LSTM 层和全连接层来生成合成轨迹。
# 生成器接收特征输入和随机噪声向量，并输出合成的轨迹特征。
# 构建鉴别器 (Discriminator)
# build_discriminator 方法创建了鉴别器模型，该模型使用 LSTM 和全连接层来判断输入轨迹的真伪。
# 鉴别器的输出是一个概率值，表示输入轨迹是真实的概率。
# 训练过程 (train 方法)
# 在训练过程中，首先加载训练数据，并对数据进行必要的填充处理，使其达到最大长度。
# 训练循环中，每个 epoch 都会选择一个随机批次的真实轨迹数据。
# 生成器产生合成轨迹，鉴别器分别对真实轨迹和合成轨迹进行判别，并更新权重。
# 生成器和鉴别器的损失通过训练来最小化，生成器尝试生成更加真实的轨迹，而鉴别器尝试更准确地鉴别真伪。
# 保存模型参数 (save_checkpoint 方法)
# save_checkpoint 方法保存模型的权重到硬盘，用于后续的加载或分析。
# 损失函数
# 使用了自定义的损失函数 d_bce_loss 和 trajLoss，这些损失函数用于训练鉴别器和生成器。
# 模型输出和保存
# 在初始化方法中，生成器、鉴别器和组合模型的结构被转换为 JSON 格式并保存到硬盘。
# 整体上，LSTM_TrajGAN 类的代码实现了一个生成对抗网络，专门用于生成和鉴别轨迹数据。通过反复训练生成器和鉴别器，模型能够学会生成看起来像真实轨迹的合成数据。