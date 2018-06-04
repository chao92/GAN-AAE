import tensorflow as tf
class Config:
    # training process
    batch_size = 64
    show_res_per_steps = 100
    internal_test_per_steps = 5 * show_res_per_steps
    checkpoint_per_steps = 20 * show_res_per_steps
    negative_sampling_per_step = 5 * show_res_per_steps
    negative_sampling_ratio = 1
    epochs = 1

    # loss
    margin = 0.3
    ratio = 0.5
    # checkpoint_path

    checkpoint_path = 'model/'
    epochs = 1
    train_ratio = 0.05
    max_step = 3000
    infor_step = 100
    middle_size = 10
    # input
    dtype=tf.float32
    num_class = 10
    input_dim = 28 * 28
    operation_dim = 256
    embed=False
    x_dim=input_dim

    # generator
    z_dim = 16

