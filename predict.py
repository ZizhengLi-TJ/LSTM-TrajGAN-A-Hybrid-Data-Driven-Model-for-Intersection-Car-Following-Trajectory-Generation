import sys
import pandas as pd
import numpy as np

from model import LSTM_TrajGAN
from keras.preprocessing.sequence import pad_sequences

def inverse_standardization(data, mean, std):
    return data * std + mean
# def conditional_inverse_standardization(df, column, mean, std):
#     # 对小于5的值进行逆标准化
#     df.loc[df[column] < 5, column] = inverse_standardization(df.loc[df[column] < 5, column], mean, std)
#     # 将大于等于5的值直接设置为10000
#     df.loc[df[column] >= 5, column] = 10000

if __name__ == '__main__':
    n_epochs = int(sys.argv[1])

    latent_dim = 50
    max_length = 133

    keys = ['sub_acc', 'sub_distance', 'sub_speed', 'distance_sub_to_pre', 'relative_speed', 'mask']
    vocab_size = {key: 1 for key in keys}

    gan = LSTM_TrajGAN(latent_dim, keys, vocab_size, max_length)

    # Test data
    x_test = np.load('data/last_test.npy', allow_pickle=True)

    x_test = [x_test[0], x_test[1], x_test[2], x_test[3], x_test[4], x_test[5],x_test[6].reshape(-1, 1)]
    # X_test = [pad_sequences(f, max_length, padding='pre', dtype='float64') for f in x_test[:8]]
    # Ensure the data has the correct shape (batch_size, max_length, 1)
    X_test = [pad_sequences(f, max_length, padding='pre', dtype='float64').reshape(-1, max_length, 1) for f in x_test[:6]]


    # Add random noise to the data
    noise = np.random.normal(0, 1, (X_test[0].shape[0], latent_dim))
    X_test.append(noise)

    # Load params for the generator
    gan.generator.load_weights('lzz_training_params/G_model_' + str(n_epochs) + '.h5') # params/G_model_2000.h5
    # gan.generator.load_weights('lzz_training_params/G_model_10.h5')  # params/G_model_2000.h5

    # Make predictions
    prediction = gan.generator.predict(X_test)

    # Process predictions
    traj_attr_concat_list = []

    for attributes in prediction:
        traj_attr_list = []
        idx = 0
        for row in attributes:
            traj_attr_list.append(row[max_length-x_test[6][idx][0]:])
            idx += 1
        traj_attr_concat = np.concatenate(traj_attr_list)
        traj_attr_concat_list.append(traj_attr_concat)
    traj_data = np.concatenate(traj_attr_concat_list,axis=1)
    # Load test CSV file to get the label and tid
    df_test = pd.read_csv('data/last_test.csv')

    tid = np.array(df_test['Vehicle ID']).reshape(-1, 1)
    direction = np.array(df_test['Direction']).reshape(-1, 1)
    front_length = np.array(df_test['front_car_length']).reshape(-1, 1)
    time = np.array(df_test['Time Sequence']).reshape(-1, 1)


    # Combine generated data with labels and tid
    traj_data = np.concatenate([tid, direction, traj_data, front_length, time], axis=1)
    df_traj_fin = pd.DataFrame(traj_data)

    # Set column names
    df_traj_fin.columns = ['Vehicle_ID', 'Direction', 'sub_acc', 'sub_distance', 'sub_speed', 'distance_sub_to_pre',
                           'relative_speed', 'mask', 'front_car_length', 'Time Sequence']

    # Remove the mask column
    del df_traj_fin['mask']

    # train:
    means = {
        'sub_acc': -0.15258029789351643,  # Replace with actual mean value from training data
        'sub_distance': 79.84301270469355,  # Replace with actual mean value from training data
        'sub_speed': 10.34862417909988,  # Replace with actual mean value from training data
        'distance_sub_to_pre': 31.182598896513802,  # Replace with actual mean value from training data
        'relative_speed': -1.37323481001532,  # Replace with actual mean value from training data
    }

    stds = {
        'sub_acc': 0.7951885678470367,  # Replace with actual std value from training data
        'sub_distance': 40.02963451402058,  # Replace with actual std value from training data
        'sub_speed': 2.9068830575789604,  # Replace with actual std value from training data
        'distance_sub_to_pre': 21.04622026011329,  # Replace with actual std value from training data
        'relative_speed': 3.499073343437109,  # Replace with actual std value from training data
    }


    # test:
    # means = {
    #     'sub_acc': -0.1860032494229614,  # Replace with actual mean value from training data
    #     'sub_distance': 79.21181257111589,  # Replace with actual mean value from training data
    #     'sub_speed': 10.523826287774959,  # Replace with actual mean value from training data
    #     'distance_sub_to_pre': 29.19157136187569,  # Replace with actual mean value from training data
    #     'relative_speed': -1.3657460840856375,  # Replace with actual mean value from training data
    # }
    #
    # stds = {
    #     'sub_acc': 0.7234987657988217,  # Replace with actual std value from training data
    #     'sub_distance': 37.67130351481293,  # Replace with actual std value from training data
    #     'sub_speed': 2.4151166770445225,  # Replace with actual std value from training data
    #     'distance_sub_to_pre': 14.526218693982944,  # Replace with actual std value from training data
    #     'relative_speed': 2.6012386931484635,  # Replace with actual std value from training data
    # }

    # Inverse standardize the data for all columns except those with conditional checks
    for key in ['sub_acc', 'sub_distance', 'sub_speed','distance_sub_to_pre', 'relative_speed']:
        df_traj_fin[key] = inverse_standardization(df_traj_fin[key], means[key], stds[key])


    # Convert types
    df_traj_fin['Vehicle_ID'] = df_traj_fin['Vehicle_ID'].astype(np.int32)
    df_traj_fin['Direction'] = df_traj_fin['Direction'].astype(np.str_)
    df_traj_fin['sub_acc'] = df_traj_fin['sub_acc'].astype(np.float64)
    df_traj_fin['sub_distance'] = df_traj_fin['sub_distance'].astype(np.float64)
    df_traj_fin['sub_speed'] = df_traj_fin['sub_speed'].astype(np.float64)
    df_traj_fin['distance_sub_to_pre'] = df_traj_fin['distance_sub_to_pre'].astype(np.float64)
    df_traj_fin['relative_speed'] = df_traj_fin['relative_speed'].astype(np.float64)
    df_traj_fin['front_speed'] = df_traj_fin['relative_speed'] + df_traj_fin['sub_speed']
    df_traj_fin['front_distance'] = df_traj_fin['sub_distance'] - df_traj_fin['distance_sub_to_pre']

    # Save synthetic trajectory data
    df_traj_fin.to_csv('results/last_train_result_6.csv', index=False)
