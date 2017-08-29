from __future__ import print_function

import logging
import os
import pickle

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
    def __init__(self, batch_size, classes, data_shape, num_label, init_states, shuffle=True, train_flag=True):
        super(OCRIter, self).__init__()
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.num_label = num_label
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.classes = classes
        if train_flag:
            self.data_path = os.path.join(os.getcwd(), '../data', 'train', 'text')
            self.label_path = os.path.join(os.getcwd(), '../data', 'train')
        else:
            self.data_path = os.path.join(os.getcwd(), '../data', 'val', 'text')
            self.label_path = os.path.join(os.getcwd(), '../data', 'val')
        self.image_set_index = self._load_image_set_index(shuffle)
        self.count = len(self.image_set_index) / self.batch_size
        self.gt = self._label_path_from_index()
        self.provide_data = [('data', (batch_size, 1, data_shape[1], data_shape[0]))] + init_states
        self.provide_label = [('label', (self.batch_size, num_label))]

    def __iter__(self):
        # print('iter')
        init_state_names = [x[0] for x in self.init_states]
        for k in range(int(self.count)):
            data = []
            label = []
            for i in range(self.batch_size):
                img_name = self.image_set_index[i + k * self.batch_size]
                img = Image.open(os.path.join(self.data_path, img_name + '.jpg')).convert('L').resize(self.data_shape)
                img = np.array(img).reshape((1, data_shape[1], data_shape[0]))
                data.append(img)
                plate_str = self.gt[int(img_name)]
                # print(plate_str)
                ret = np.zeros(self.num_label, int)
                for number in range(len(plate_str)):
                    ret[number] = self.classes.index(plate_str[number]) + 1
                label.append(ret)

            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

    def _load_image_set_index(self, shuffle):
        assert os.path.isdir(self.data_path), 'Path does not exist: {}'.format(self.data_path)
        image_set_index = []
        list_dir = os.walk(self.data_path)
        for root, _, image_names in list_dir:
            for name in image_names:
                if name.endswith('.jpg'):
                    image_set_index.append(name.split('.')[0])
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    def _label_path_from_index(self):
        label_file = os.path.join(self.label_path, 'gt.pkl')
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        gt_file = open(label_file, 'rb')
        label_file = pickle.load(gt_file)
        gt_file.close()
        return label_file


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
    # _classes = classes + [' ', ]
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        # logger.debug('Validation[%03d]:%s' % (
        # i, '|'.join(['%s,%s' % (_classes[int(_x)], _classes[_y]) for _x, _y in zip(l, p)])))
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
    # set up logger
    log_file_name = "crnn2_chinese_plate.log"
    log_file = open(log_file_name, 'w')
    log_file.close()
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_name)
    logger.addHandler(fh)

    prefix = os.path.join(os.getcwd(), 'model', 'crnn2_chinese_ct2')

    BATCH_SIZE = 64
    SEQ_LENGTH = 17

    num_hidden = 128
    num_lstm_layer = 2

    num_label = 9
    data_shape = (200, 32)
    with open('chinesechars.txt') as to_read: classes = list(to_read.read().strip())
    num_classes = len(classes) + 1


    def sym_gen(seq_len):
        return crnn(num_lstm_layer, BATCH_SIZE, seq_len,
                    num_hidden=num_hidden, num_classes=num_classes,
                    num_label=num_label, dropout=0.3)


    init_c = [('l%d_init_c' % l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer * 2)]
    init_h = [('l%d_init_h' % l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer * 2)]
    init_states = init_c + init_h

    data_train = OCRIter(BATCH_SIZE, classes, data_shape, num_label, init_states)
    data_val = OCRIter(BATCH_SIZE, classes, data_shape, num_label, init_states, train_flag=False)

    symbol = sym_gen(SEQ_LENGTH)
    # model = mx.module.Module(
    #     symbol=symbol,
    #     data_names=['data', 'l0_init_c', 'l1_init_c', 'l2_init_c', 'l3_init_c', 'l0_init_h', 'l1_init_h', 'l2_init_h',
    #                 'l3_init_h'],
    #     label_names=['label', ],
    #     context=mx.gpu(0),
    #
    # )

    model = mx.module.Module.load(
        prefix,
        16,
        data_names=['data', 'l0_init_c', 'l1_init_c', 'l2_init_c', 'l3_init_c', 'l0_init_h', 'l1_init_h', 'l2_init_h',
                    'l3_init_h'],
        label_names=['label', ],
        context=mx.gpu(0),
    )

    import logging

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    logger.info('begin fit')

    model.fit(
        train_data=data_train,
        eval_data=data_val,
        eval_metric=mx.metric.np(Accuracy),
        batch_end_callback=mx.callback.Speedometer(BATCH_SIZE, 100),
        epoch_end_callback=mx.callback.do_checkpoint(prefix, 1),
        optimizer='sgd',
        optimizer_params={'learning_rate': 0.001, 'momentum': 0.9},
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch=100

    )
    model.save_params('crnn2ctc')