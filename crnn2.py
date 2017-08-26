# pylint:skip-file
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math

from mxnet.rnn import BidirectionalCell, LSTMCell

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states", "forward_state", "backward_state",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])


def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx):
    """LSTM Cell symbol"""
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def crnn(num_lstm_layer, batch_size,seq_len,num_hidden, num_classes,num_label, dropout=0.):
    last_states = []
    forward_param = []
    backward_param = []
    for i in range(num_lstm_layer * 2):
        last_states.append(LSTMState(c=mx.sym.Variable("l%d_init_c" % i), h=mx.sym.Variable("l%d_init_h" % i)))
        if i % 2 == 0:
            forward_param.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                           i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                           h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                           h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        else:
            backward_param.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                            i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                            h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                            h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))

    # input
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    kernel_size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (2, 2)]
    padding_size = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (0, 0)]
    layer_size = [64, 128, 256, 256, 512, 512, 512]

    def convRelu(i, input_data, bn=False):
        layer = mx.symbol.Convolution(name='conv-%d' % i, data=input_data, kernel=kernel_size[i], pad=padding_size[i],
                                      num_filter=layer_size[i])
        if bn:
            layer = mx.symbol.BatchNorm(data=layer, name='batchnorm-%d' % i)
        layer = mx.symbol.Activation(data=layer, act_type='relu', name='relu-%d' % i)
        return layer

    net = convRelu(0, data)
    net = mx.symbol.Pooling(data=net, name='pool-0', pool_type='max', kernel=(2, 2), stride=(2, 2))
    net = convRelu(1, net)
    net = mx.symbol.Pooling(data=net, name='pool-1', pool_type='max', kernel=(2, 2), stride=(2, 2))
    net = convRelu(2, net, True)
    net = convRelu(3, net)
    net = mx.symbol.Pooling(data=net, name='pool-2', pool_type='max', kernel=(2, 2), stride=(2, 1), pad=(0, 1))
    net = convRelu(4, net, True)
    net = convRelu(5, net)
    net = mx.symbol.Pooling(data=net, name='pool-3', pool_type='max', kernel=(2, 2), stride=(2, 1), pad=(0, 1))
    net = convRelu(6, net, True)
    if dropout > 0:
        net = mx.symbol.Dropout(data=net, p=dropout)

    net = mx.symbol.reshape(data=net,shape=(batch_size,seq_len,-1))
    slices_net = mx.symbol.split(data=net,axis=1,num_outputs=seq_len,squeeze_axis=1)

    init_c = [('l%d_init_c' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer * 2)]
    init_h = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer * 2)]
    init_states = init_c + init_h
    init_values = {x[0]: x[1] for x in init_states}

    forward_hidden = []
    for seqidx in range(seq_len):
        hidden = slices_net[seqidx]
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[2 * i],
                              param=forward_param[i],
                              seqidx=seqidx, layeridx=0)
            hidden = next_state.h
            last_states[2 * i] = next_state
        forward_hidden.append(hidden)

    backward_hidden = []
    for seqidx in range(seq_len):
        k = seq_len - seqidx - 1
        hidden = slices_net[k]
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[2 * i + 1],
                              param=backward_param[i],
                              seqidx=k, layeridx=1)
            hidden = next_state.h
            last_states[2 * i + 1] = next_state
        backward_hidden.insert(0, hidden)

    hidden_all = []
    for i in range(seq_len):
        hidden_all.append(mx.sym.Concat(*[forward_hidden[i], backward_hidden[i]], dim=1))

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_classes)

    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm = mx.sym.WarpCTC(data=pred, label=label, label_length=num_label, input_length=seq_len)
    # sm = mx.sym.reshape(data=sm,shape=(-1,seq_len,num_classes))

    # arg_shape, output_shape, aux_shape = sm.infer_shape(**dict(init_values, **{"data": (batch_size, 1, 32, 200),"label":(batch_size,seq_len)}))
    # print(output_shape)

    return sm


# if __name__ == '__main__':
#     model = crnn(2,8,17,256,3820,17,0.7)

