import glob

import cv2
import mxnet as mx
import numpy as np
from PIL import Image


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
    def __init__(self, batch_size, classes, data_shape, num_label, init_states, imgs):
        super(OCRIter, self).__init__()
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.num_label = num_label
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.classes = classes
        self.images = imgs
        self.provide_data = [('data', (batch_size, 1, data_shape[1], data_shape[0]))] + init_states
        self.provide_label = [('label', (self.batch_size, num_label))]

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]

        data_names = ['data'] + init_state_names
        label_names = ['label']

        data = []
        label = []

        for img in self.images:
            img = Image.fromarray(img).resize(self.data_shape,Image.BILINEAR)
            img = np.array(img).reshape((1, self.data_shape[1], self.data_shape[0]))
            data.append(img)
            ret = np.zeros(self.num_label, int)
            label.append(ret)

        data_all = [mx.nd.array(data)] + self.init_state_arrays
        label_all = [mx.nd.array(label)]

        data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
        yield data_batch

    def reset(self):
        pass


class predict():
    def __init__(self, images, data_shape, model_name, from_epoch, charset,seq_len,num_label,num_hidden, enable_gpu=False):
        self.module = mx.mod.Module.load(
            model_name,
            from_epoch,
            False,
            context=mx.gpu(0) if enable_gpu else mx.cpu(0),
            data_names=(
                'data', 'l0_init_c', 'l1_init_c', 'l2_init_c', 'l3_init_c', 'l0_init_h', 'l1_init_h', 'l2_init_h',
                'l3_init_h'),
            label_names=('label',)
        )
        with open(charset) as to_read: self.classes = list(to_read.read().strip())
        self.img = images
        self.BATCH_SIZE = len(images)
        self.num_hidden = num_hidden
        self.seq_len = seq_len
        num_lstm_layer = 2
        self.data_shape = data_shape
        init_c = [('l%d_init_c' % l, (self.BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer * 2)]
        init_h = [('l%d_init_h' % l, (self.BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer * 2)]
        init_states = init_c + init_h
        self.to_predict = OCRIter(self.BATCH_SIZE, len(charset) + 1, data_shape, num_label, init_states, images)
        self.module.bind(self.to_predict.provide_data, self.to_predict.provide_label, for_training=False)

    def __get_string(self, label_list):
        # Do CTC label rule
        # CTC cannot emit a repeated symbol on consecutive timesteps
        ret = []
        label_list2 = [0] + list(label_list)
        for i in range(len(label_list)):
            c1 = label_list2[i]
            c2 = label_list2[i + 1]
            if c2 == 0 or c2 == c1:
                continue
            ret.append(c2)
        # change to ascii
        s = ''
        for l in ret:
            if l > 0 and l < (len(self.classes) + 1):
                c = self.classes[l - 1]
            else:
                c = ''
            s += c
        return s

    def run(self):
        prob = self.module.predict(self.to_predict).asnumpy()
        label_list = [['' for _ in range(self.seq_len)] for i in range(self.BATCH_SIZE)]
        for i in range(self.seq_len):
            for j in range(self.BATCH_SIZE):
                max_index = np.argsort(prob[i * self.BATCH_SIZE + j])[::-1][0]
                label_list[j][i] = max_index
        result = []
        for i in range(self.BATCH_SIZE):
            result.append(self.__get_string(label_list[i]))
        to_return = []
        for i in range(self.BATCH_SIZE):
            to_return.append([np.array(Image.fromarray(self.img[i]).resize(self.data_shape, Image.BILINEAR)),result[i]])
        return to_return

if __name__ == '__main__':
    files = [
        'test_image.jpg'
    ]
    images = [cv2.imread(x,0) for x in files]
    my_predictor = predict(images,(256,32),'model/digit',0,'./digit.txt',32,24,128,False)
    result = my_predictor.run()
    for m_image,predict_label in result:
        cv2.imshow('result',m_image)
        print(predict_label)
        cv2.waitKey(0)
    cv2.destroyAllWindows()