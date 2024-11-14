from keras.losses import binary_crossentropy
import tensorflow as tf
import keras.backend as K


# BCE loss for the discriminator
def d_bce_loss(mask):
    def loss(y_true, y_pred):
        d_bce_loss = binary_crossentropy(y_true, y_pred)
        return d_bce_loss

    return loss

# means = {
#         'sub_acc': -0.15258029789351643,  # Replace with actual mean value from training data
#         'sub_distance': 79.84301270469355,  # Replace with actual mean value from training data
#         'sub_speed': 10.34862417909988,  # Replace with actual mean value from training data
#         'distance_sub_to_pre': 31.182598896513802,  # Replace with actual mean value from training data
#         'relative_speed': -1.37323481001532,  # Replace with actual mean value from training data
#     }
#
#     stds = {
#         'sub_acc': 0.7951885678470367,  # Replace with actual std value from training data
#         'sub_distance': 40.02963451402058,  # Replace with actual std value from training data
#         'sub_speed': 2.9068830575789604,  # Replace with actual std value from training data
#         'distance_sub_to_pre': 21.04622026011329,  # Replace with actual std value from training data
#         'relative_speed': 3.499073343437109,  # Replace with actual std value from training data
#     }
# trajLoss for the generator

mean_acc = -0.15258029789351643
std_acc = 0.7951885678470367
mean_speed = 10.34862417909988
std_speed = 2.9068830575789604
mean_distance = 79.84301270469355
std_distance = 40.02963451402058


def trajLoss(real_traj, gen_traj):
    def loss(y_true, y_pred):
        traj_length = K.sum(real_traj[5], axis=1)

        # Binary Cross-Entropy loss
        bce_loss = binary_crossentropy(y_true, y_pred)

        # Mean Squared Error loss for continuous variables
        mse_acc = K.sum(K.sum(tf.multiply(tf.square(gen_traj[0] - real_traj[0]), real_traj[5]), axis=1), axis=1,
                        keepdims=True)
        mse_acc = K.sum(tf.math.divide(mse_acc, traj_length))

        mse_distance = K.sum(K.sum(tf.multiply(tf.square(gen_traj[1] - real_traj[1]), real_traj[5]), axis=1), axis=1,
                             keepdims=True)
        mse_distance = K.sum(tf.math.divide(mse_distance, traj_length))

        mse_speed = K.sum(K.sum(tf.multiply(tf.square(gen_traj[2] - real_traj[2]), real_traj[5]), axis=1), axis=1,
                          keepdims=True)
        mse_speed = K.sum(tf.math.divide(mse_speed, traj_length))

        mse_distance_sub_to_pre = K.sum(K.sum(tf.multiply(tf.square(gen_traj[3] - real_traj[3]), real_traj[5]), axis=1),
                                        axis=1, keepdims=True)
        mse_distance_sub_to_pre = K.sum(tf.math.divide(mse_distance_sub_to_pre, traj_length))

        mse_relative_speed = K.sum(K.sum(tf.multiply(tf.square(gen_traj[4] - real_traj[4]), real_traj[5]), axis=1),
                                   axis=1, keepdims=True)
        mse_relative_speed = K.sum(tf.math.divide(mse_relative_speed, traj_length))

        # Inverse standardization
        gen_acc_inv = gen_traj[0] * std_acc + mean_acc
        gen_speed_inv = gen_traj[2] * std_speed + mean_speed
        gen_distance_inv = gen_traj[1] * std_distance + mean_distance

        # Assuming time step (delta_t) is 0.1 seconds
        delta_t = 0.1

        # Physics-based constraint 1: speed = distance / time
        distance_diff = -(gen_distance_inv[:, 1:] - gen_distance_inv[:, :-1])  # Calculating distance difference
        speed_estimated = distance_diff / delta_t  # Estimating speed

        # Padding speed_estimated to have the same shape as gen_traj[2] by adding a column of zeros at the end
        speed_estimated_padded = tf.concat([speed_estimated, tf.zeros_like(speed_estimated[:, :1])], axis=1)

        # Physics-based constraint 2: acc = speed_diff / time
        speed_diff = gen_speed_inv[:, 1:] - gen_speed_inv[:, :-1]  # Calculating speed difference
        acc_estimated = speed_diff / delta_t  # Estimating acceleration

        # Padding acc_estimated to have the same shape as gen_traj[0] by adding a column of zeros at the end
        acc_estimated_padded = tf.concat([acc_estimated, tf.zeros_like(acc_estimated[:, :1])], axis=1)

        # Physics-based constraint losses
        physics_constraint1 = tf.reduce_mean(tf.square(gen_speed_inv - speed_estimated_padded))
        physics_constraint2 = tf.reduce_mean(tf.square(gen_acc_inv - acc_estimated_padded))

        # Weights for different loss components
        p_bce = 10
        p_acc = 10
        p_distance = 10
        p_speed = 10
        p_distance_sub_to_pre = 10
        p_relative_speed = 10
        p_physics1 = 100
        p_physics2 = 100

        return (bce_loss * p_bce +
                mse_acc * p_acc +
                mse_distance * p_distance +
                mse_speed * p_speed +
                mse_distance_sub_to_pre * p_distance_sub_to_pre +
                mse_relative_speed * p_relative_speed +
                physics_constraint1 * p_physics1 +
                physics_constraint2 * p_physics2)

    return loss
