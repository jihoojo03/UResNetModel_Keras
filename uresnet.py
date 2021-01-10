from keras.models import Input, Model
from keras.layers import Add, Conv2D, Concatenate, MaxPooling2D, ReLU
from keras.layers import Conv2DTranspose, UpSampling2D, Dropout, BatchNormalization

'''
U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)
Residual Block: Deep Residual Learning for Image Recognition
(https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
---
img_shape: (height, width, channels)
out_ch: number of output channels
start_ch: number of channels of the first conv
depth: zero indexed depth of the U-structure
inc_rate: rate at which the conv channels will increase
activation: activation function after convolutions
dropout: amount of dropout in the contracting part
res_rate: rate at which the residual blocks repeat
batchnorm: adds Batch Normalization if true
maxpool: use strided conv instead of maxpooling if false
upconv: use transposed conv instead of upsamping + conv if false
residual: add residual connections around each conv block if true
'''

def get_uresnet_model(input_channel_num=3, out_ch=3, start_ch=16, depth=5, inc_rate=2., activation='relu',
         dropout=0.5, res_rate=4, batchnorm=True, maxpool=True, upconv=True, residual=True):
    def _conv_block(m, dim, acti, bn, res, do=0):
        n = Conv2D(dim, 3, activation=acti, padding='same')(m)
        n = BatchNormalization()(n) if bn else n

        return n

    def _res_block(m, dim, acti, bn, res, do=0):
        n = Conv2D(dim, 3, activation=acti, padding='same', kernel_initializer='he_normal') (m)
        n = BatchNormalization() (n) if bn else n
        n = Dropout(do) (n) if do else n
        n = Conv2D(dim, 3, padding='same', kernel_initializer='he_normal') (n)
        n = BatchNormalization() (n) if bn else n
        x = Add() ([m, n]) if res else n
        x = ReLU() (x)

        return x

    def _level_block(m, dim, depth, inc, acti, do, bn, mp, up, res, res_rate):
        if depth > 0:
            # Contracting Path
            n = _conv_block(m, dim, acti, bn, res)
            for i in range(res_rate):
                m = _res_block(n, dim, acti, bn, res) if res else n
            m = MaxPooling2D()(m) if mp else Conv2D(dim, 3, strides=2, padding='same')(m)

            # Contracting with Recursive
            m = _level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res, res_rate)

            # Expanding Path
            for i in range(res_rate):
                m = _res_block(m, int(inc * dim), acti, bn, res) if res else m
            if up:
                m = UpSampling2D()(m)
                m = Conv2D(dim, 2, activation=acti, padding='same')(m)
            else:
                m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
            m = Concatenate()([n, m])
            m = _conv_block(m, dim, acti, bn, res)
        else:
            m = _conv_block(m, dim, acti, bn, res, do)

        return m

    i = Input(shape=(None, None, input_channel_num))
    o = _level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual, res_rate)
    o = Conv2D(out_ch, 1)(o)
    model = Model(inputs=i, outputs=o)

    return model
