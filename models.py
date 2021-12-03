import os                                               # nopep8
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"   # nopep8
import keras
import time
import math
import losses
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, DepthwiseConv2D, BatchNormalization
from keras.layers import Concatenate, Activation, SeparableConv2D, Input, GlobalAveragePooling2D, Add, Lambda
from keras.layers import AveragePooling2D, LeakyReLU, Conv2DTranspose, UpSampling2D
import keras.backend as K
import misc
import yield_data
from os import path
conv2d_common_args = (lambda **kwargs: kwargs)(
    padding='same',
    data_format='channels_last',
    activation=None)
used_names = {}
eps = 1e-6


def assign_name(name):
    idx = used_names.get(name, -1)+1
    used_names[name] = idx
    return name+'_%d' % idx


def _Conv2D(channels=None, size=None, activate=True, name=None, **kwargs):
    def inner(input):
        nonlocal name, activate, channels, size
        if(channels is None):
            channels = input.shape.dims[-1]
        if(size is None):
            size = (3, 3)
        kwargs_local = {}
        kwargs_local.update(conv2d_common_args)
        kwargs_local.update(kwargs)
        if(name is None):
            name = "Conv2D(%d,%dx%d)" % (channels, *size)
        name = assign_name(name)

        if(activate):
            out = Conv2D(channels, size, **kwargs_local)(input)
            out = LeakyReLU(name=name)(out)
        else:
            out = Conv2D(channels, size, name=name, **kwargs_local)(input)    # nopep8
        return out
    return inner


def weight_block(channels, **kwargs):
    kwargs_local = {}
    kwargs_local.update(conv2d_common_args)
    kwargs_local.update(kwargs)

    def __Conv2D(channels=None, size=None, activate=True, name=None, **kwargs):
        def inner(input):
            nonlocal kwargs_local, name, activate, channels, size
            if(channels is None):
                channels = input.shape.dims[-1]
            if(size is None):
                size = (3, 3)
            kwargs_local_local = {}
            kwargs_local_local.update(kwargs_local)
            kwargs_local_local.update(kwargs)
            if(name is None):
                name = "Conv2D(%d,%dx%d)" % (channels, *size)
            name = assign_name(name)

            if(activate):
                out = Conv2D(channels, size, **kwargs_local_local)(input)
                out = LeakyReLU(name=name)(out)
            else:
                out = Conv2D(channels, size, name=name, **kwargs_local_local)(input)    # nopep8
            return out
        return inner

    def bottleneck(out_ch, inner_conv):
        def inner(input):
            out = input
            in_ch = out.shape.dims[-1]
            ch = min(in_ch, out_ch >> 1)
            if(ch != in_ch):
                out = __Conv2D(ch, (1, 1))(out)
            out = inner_conv(out)
            if(out_ch != out.shape.dims[-1]):
                out = __Conv2D(out_ch, (1, 1))(out)
            return out

    def residual(input):
        nonlocal channels
        name = "residual(%d)" % channels
        name = assign_name(name)

        input_channels = input.shape.dims[-1]

        if(input_channels == channels):
            shortcut = input
        else:
            shortcut = __Conv2D(channels, (1, 1), activate=False)(input)

        ch = min(input_channels >> 1, channels >> 1)
        if(ch != input_channels):
            out = __Conv2D(ch, (1, 1), activate=False)(input)
        else:
            out = input
        out = __Conv2D(ch, (5, 5))(out)
        if(channels != out.shape.dims[-1]):
            out = __Conv2D(channels, (1, 1), activate=False)(out)

        out = Add()([out, shortcut])
        out = LeakyReLU(name=name)(out)
        return out

    def inception(input):
        nonlocal channels
        name = "inception(%d)" % channels
        name = assign_name(name)

        in_channels = input.shape.dims[-1]
        in_size = input.shape.dims[1]

        def bottleneck_channels(in_ch, out_ch):
            ret = min(in_ch, out_ch) >> 1
            ret = max(ret, 2)
            return ret

        tmp = bottleneck_channels(
            in_channels, channels >> 1)  # bottleneck channels

        out0 = __Conv2D(tmp, (1, 1), name="bottleneck")(input)
        out0 = __Conv2D(tmp, (3, 3))(out0)
        out0 = __Conv2D(tmp << 1, (1, 1), name="bottleneck_up")(out0)

        out1 = __Conv2D(tmp, (1, 1), name="bottleneck")(input)
        out1 = __Conv2D(tmp, (5, 5))(out1)
        out1 = __Conv2D(tmp << 1, (1, 1), name="bottleneck_up")(out1)

        out = Concatenate(name=name)([out0, out1])
        return out

    def residual_inception(input):
        name = "res_inception"
        name = assign_name(name)
        out = inception(input)
        out_channels = out.shape.dims[-1]
        if(input.shape.dims[-1] == out_channels):
            shortcut = input
        else:
            shortcut = __Conv2D(out_channels, (1, 1), activate=False)(input)

        out = Add()([out, shortcut])
        out = LeakyReLU(name=name)(out)
        return out
    return residual


def upsample_block(method="combine"):
    dic = {}

    def deco(func):
        dic[func.__name__[6:]] = func
        return func

    @deco
    def inner_naive(input):
        return UpSampling2D((2, 2), interpolation='bilinear')(input)

    @deco
    def inner_conv2dt(input, activate=True):
        in_ch = input.shape.dims[-1]
        out = _Conv2D(in_ch >> 1, (1, 1), activate=False)(input)
        out = Conv2DTranspose(in_ch >> 1, (4, 4), strides=(2, 2), **conv2d_common_args)(out)    # nopep8
        out = _Conv2D(in_ch, (1, 1), activate=activate)(out)
        out = LeakyReLU()(out)
        return out

    @deco
    def inner_combine(input):
        a = inner_naive(input)
        b = inner_conv2dt(input, activate=False)
        out = Add()([a, b])
        out = LeakyReLU()(out)
        return out

    @deco
    def inner_concat(input):
        in_ch = input.shape.dims[-1]
        input = Conv2D(in_ch >> 1, (1, 1), **conv2d_common_args)(input)
        a = inner_naive(input)
        b = inner_conv2dt(input)
        out = Concatenate()([a, b])
        return out

    def inner(input):
        return dic[method](input)
    return inner


class InstanceNormalization(keras.layers.Layer):
    def __init__(self, normalize_mean=True, normalize_var=True):
        super().__init__()
        self.normalize_mean = normalize_mean
        self.normalize_var = normalize_var

    def get_config(self):
        return {"normalize_mean": self.normalize_mean, "normalize_var": self.normalize_var}

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(1, 1, 1, input_shape[-1]),
            initializer='glorot_normal',
            trainable=True, name='w')
        self.b = self.add_weight(
            shape=(1, 1, 1, input_shape[-1]),
            initializer='glorot_normal',
            trainable=True, name='b')

        super().build(input_shape)

    def call(self, inputs):
        var = K.var(inputs, axis=[1, 2], keepdims=True)
        mean = K.mean(inputs, axis=[1, 2], keepdims=True)
        ret = inputs
        if(self.normalize_mean):
            ret -= mean
        if(self.normalize_var):
            ret /= K.sqrt(var+1e-3)
        # ret=(inputs-mean)/K.sqrt(var+1e-3)
        return ret*self.w+self.b


def downsample_block(method='combine'):
    dic = {}

    def deco(func):
        dic[func.__name__[6:]] = func
        return func

    @deco
    def inner_pooling(input):
        return AveragePooling2D((2, 2))(input)

    @deco
    def inner_conv2d(input, activate=True):
        in_ch = input.shape.dims[-1]
        out = _Conv2D(in_ch >> 1, (1, 1), activate=False)(input)
        out = _Conv2D(in_ch >> 1, (4, 4), strides=(2, 2), activate=False)(out)  # nopep8
        out = _Conv2D(in_ch, (1, 1), activate=activate)(out)
        return out

    @deco
    def inner_combine(input):
        a = inner_pooling(input)
        b = inner_conv2d(input, activate=False)
        out = Add()([a, b])
        return LeakyReLU()(out)

    def inner(input):
        return dic[method](input)
    return inner


def high_bit(i):
    i = int(i+eps)
    i |= i >> 1
    i |= i >> 2
    i |= i >> 4
    i |= i >> 8
    i |= i >> 16
    i |= i >> 32
    return i-(i >> 1)


def show_plot(model):
    model.summary()
    from io import BytesIO
    f = BytesIO()
    keras.utils.plot_model(model, to_file='tmp.png', show_shapes=True)
    from PIL import Image
    im = Image.open('tmp.png')
    im.show()


class mean_var(keras.layers.Layer):
    def __init__(self, mean=0, var=1, **kwargs):
        self.mean = mean
        self.var = var
        super().__init__(**kwargs)

    def get_config(self):
        return {"mean": self.mean, "var": self.var}

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)

    def call(self, input):
        shape = input.shape.dims
        axis = list(range(1, len(shape)))
        mean = K.mean(input, axis=axis, keepdims=True)
        var = K.var(input, axis=axis, keepdims=True)

        return (input-mean)/K.sqrt(var+eps)


def encoder(input_size=128, out_dims=256, start_ch=16, end_ch=None):
    if(end_ch is None):
        end_ch = out_dims
    depth = int(math.log(input_size/2)/math.log(2)+eps)

    alpha = (end_ch/start_ch)**(1/depth)

    input = Input((input_size, input_size, 3), name='encoder_input')
    out = _Conv2D(start_ch, (3, 3))(input)
    cur_ch = start_ch
    for i in range(depth):
        cur_ch *= alpha
        _cur_ch = high_bit(cur_ch)
        out = weight_block(_cur_ch)(out)
        out = downsample_block()(out)
    out = weight_block(_cur_ch)(out)
    out = Flatten()(out)
    out = Dense(out_dims, activation='tanh')(out)
    # out=mean_var()(out)
    print(out.shape)
    return keras.models.Model(input, out)


def decoder(output_size=128, end_ch=16, start_ch=None, input_dims=256):
    if(start_ch is None):
        start_ch = input_dims
    start_size = output_size
    depth = 0
    while(start_size//2 > 1):
        start_size //= 2
        depth += 1

    alpha = (end_ch/start_ch)**(1/depth)

    input = Input((input_dims,), name='decoder_input')
    out = Dense(start_size*start_size*start_ch)(input)
    out = LeakyReLU()(out)
    out = keras.layers.Reshape((start_size, start_size, start_ch))(out)
    cur_ch = start_ch
    for i in range(depth):
        cur_ch *= alpha
        _cur_ch = high_bit(cur_ch)
        out = weight_block(_cur_ch)(out)
        out = upsample_block()(out)
    out = weight_block(_cur_ch)(out)
    out = Conv2D(3, (3, 3), padding='same', activation='tanh')(out)
    return keras.models.Model(input, out)


def ae():
    E = encoder()
    D = decoder()
    input = Input(E.get_input_at(0).shape.dims[1:])
    print(E.get_input_at(0).shape.dims[1:])
    encoded = E(input)
    print(encoded.shape)
    decoded = D(encoded)
    AE = keras.models.Model(input, decoded)
    return E, D, AE


def name_by_E_D(E, D):
    tmp = []

    def prt(end='\n', *args):
        nonlocal tmp
        tmp.append(' '.join([str(i) for i in args])+end)
    E.summary(print_fn=prt)

    D.summary(print_fn=prt)
    name = misc.base32(''.join(tmp))
    return name


def save(E, D, AE):
    name = name_by_E_D(E, D)
    import numpy as np
    import cv2
    pth = path.join(path.dirname(__file__), 'saved', name)
    if(not path.exists(pth)):
        os.makedirs(pth)

    true = yield_data.yield_data(8, 128)
    restored = AE.predict([np.array(true)])
    ls = []
    for i in range(8):
        ls.append(true[i])
        ls.append(restored[i])
    im = yield_data.plot_given_img(4, ls)
    E.save(path.join(pth, 'encoder.h5'))
    D.save(path.join(pth, 'decoder.h5'))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path.join(pth, 'sample.jpg'), im)


if(__name__ == '__main__'):
    D = decoder()
    show_plot(D)

    '''E=encoder()
    show_plot(E)'''
