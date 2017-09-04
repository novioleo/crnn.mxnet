from __future__ import print_function

import argparse
import logging
import os

import mxnet as mx
import numpy as np
from PIL import Image

from crnn import crnn


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class OCRIter(mx.io.DataIter):
    def __init__(self, batch_size, classes, data_shape, num_label, init_states,dataset_lst):
        super(OCRIter, self).__init__()
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.num_label = num_label
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.classes = classes
        self.dataset_lst_file = open(dataset_lst)
        self.provide_data = [('data', (batch_size, 1, data_shape[1], data_shape[0]))] + init_states
        self.provide_label = [('label', (self.batch_size, num_label))]

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
        data = []
        label = []
        cnt = 0
        for m_line in self.dataset_lst_file:
            img_path,img_label = m_line.strip().split('\t')
            cnt += 1
            img = Image.open(img_path).convert('L').resize(self.data_shape)
            img = np.array(img).reshape((1, data_shape[1], data_shape[0]))
            data.append(img)
            plate_str = img_label
            ret = np.zeros(self.num_label, int)
            for number in range(len(plate_str)):
                ret[number] = self.classes.index(plate_str[number]) + 1
            label.append(ret)
            if cnt % self.batch_size == 0:
                data_all = [mx.nd.array(data)] + self.init_state_arrays
                label_all = [mx.nd.array(label)]
                data_names = ['data'] + init_state_names
                label_names = ['label']
                data.clear()
                label.clear()
                yield SimpleBatch(data_names, data_all, label_names, label_all)
                continue

    def reset(self):
        if self.dataset_lst_file.seekable():
            self.dataset_lst_file.seek(0)


def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i + 1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret


def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break

        ret.append(l[i])
    return ret


def Accuracy(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                    match = False
                    break
            if match:
                hit += 1.0
        total += 1.0
    return hit / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',required=True,help='the model name')
    parser.add_argument('--charset',required=True,help='the charset file')
    parser.add_argument('--train_lst',required=True,help='the csv which contains all train list')
    parser.add_argument('--val_lst',required=True,help='the csv which contains all val list')
    parser.add_argument('--batch_size',type=int,default=64,help='train/val batch size,default is 64')
    parser.add_argument('--seq_len',type=int,default=17,help='the sequence length effected by image width')
    parser.add_argument('--num_label',type=int,default=9,help='output label length,must less than seq_len')
    parser.add_argument('--imgH',type=int,default=32,help='image height,must divided by 16')
    parser.add_argument('--imgW',type=int,default=200,help='image width')
    parser.add_argument('--num_hidden',type=int,default=256,help='num of parameters in lstm hidden layer')
    parser.add_argument('--num_lstm',type=int,default=2,help='num of lstm layers')
    parser.add_argument('--gpu',action='store_true',help='enable train with gpu(0)')
    parser.add_argument('--from_epoch',type=int,help='continue train from specific epoch file')
    parser.add_argument('--learning_rate',type=float,help='the learning rate of adam')
    opt = parser.parse_args()
    model_name = opt.name
    log_file_name = model_name+'.log'
    log_file = open(log_file_name, 'w')
    log_file.close()
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_name)
    logger.addHandler(fh)
    model_dir_path = os.path.join(os.getcwd(), 'model')
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)
    prefix = os.path.join(os.getcwd(), 'model', model_name)

    BATCH_SIZE = opt.batch_size
    SEQ_LENGTH = opt.seq_len

    num_hidden = opt.num_hidden
    num_lstm_layer = opt.num_lstm

    num_label = opt.num_label
    data_shape = (opt.imgW, opt.imgH)
    with open(opt.charset) as to_read: classes = list(to_read.read().strip())
    num_classes = len(classes) + 1


    def sym_gen(seq_len):
        return crnn(num_lstm_layer, BATCH_SIZE, seq_len,
                    num_hidden=num_hidden, num_classes=num_classes,
                    num_label=num_label, dropout=0.3)


    init_c = [('l%d_init_c' % l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer * 2)]
    init_h = [('l%d_init_h' % l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer * 2)]
    init_states = init_c + init_h

    data_train = OCRIter(BATCH_SIZE, classes, data_shape, num_label, init_states,opt.train_lst)
    data_val = OCRIter(BATCH_SIZE, classes, data_shape, num_label, init_states,opt.val_lst)

    ctx = mx.gpu(0) if opt.gpu else mx.cpu(0)
    data_names = ['data', 'l0_init_c', 'l1_init_c', 'l2_init_c', 'l3_init_c', 'l0_init_h', 'l1_init_h', 'l2_init_h',
                        'l3_init_h']
    label_names = ['label', ]
    if opt.from_epoch is None:
        symbol = sym_gen(SEQ_LENGTH)
        model = mx.module.Module(
            symbol=symbol,
            data_names=data_names,
            label_names=label_names,
            context=ctx
        )
    else:
        model = mx.module.Module.load(
            prefix,
            opt.from_epoch,
            data_names=data_names,
            label_names=label_names,
            context=ctx,
        )

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    logger.info('begin fit')

    model.fit(
        train_data=data_train,
        eval_data=data_val,
        eval_metric=mx.metric.np(Accuracy),
        batch_end_callback=mx.callback.Speedometer(BATCH_SIZE, 100),
        epoch_end_callback=mx.callback.do_checkpoint(prefix, 1),
        optimizer='adam',
        optimizer_params={'learning_rate': opt.learning_rate},
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch=100,
        begin_epoch=opt.from_epoch if opt.from_epoch else 0
    )
    model.save_params(model_name)