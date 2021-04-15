import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten, Dropout, MaxPool2d


def get_generator(shape, gf_dim=64, o_size=32, o_channel=3): # Dimension of gen filters in first conv layer. [64]
    image_size = o_size
    s4 = image_size // 4
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    ni = Input(shape)
    nn = Dense(n_units=(gf_dim * 4 * s4 * s4))(ni)
    nn = Reshape(shape=(-1, s4, s4, gf_dim * 4))(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu)(nn)
    nn = DeConv2d(gf_dim * 2, (5, 5), (1, 1))(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu)(nn)
    nn = DeConv2d(gf_dim, (5, 5), (2, 2))(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu)(nn)
    nn = DeConv2d(o_channel, (5, 5), (2, 2), act=tf.nn.tanh)(nn)

    return tl.models.Model(inputs=ni, outputs=nn, name='generator')


def get_discriminator(shape, df_dim=64): # Dimension of discrim filters in first conv layer. [64]
    lrelu = lambda x : tf.nn.leaky_relu(x, 0.2)

    ni = Input(shape)
    nn = Conv2d(df_dim, (5, 5), (2, 2), act=lrelu)(ni)
    nn = Conv2d(df_dim * 2, (5, 5), (2, 2))(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu)(nn)
    nn = Conv2d(df_dim*4, (5, 5), (2, 2))(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu)(nn)
    nn = Flatten()(nn)
    nn = Dense(n_units=1)(nn)

    return tl.models.Model(inputs=ni, outputs=nn, name='discriminator')
