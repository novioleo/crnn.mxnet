import mxnet as mx
import os
import pickle
from PIL import Image
import numpy as np
data_shape = (200,32)


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None  # TODO: what is index?

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
            self.data_path = os.path.join(os.getcwd(), '../data', 'test', 'text')
            self.label_path = os.path.join(os.getcwd(), '../data', 'test')
        self.image_set_index = self._load_image_set_index(shuffle)
        self.count = len(self.image_set_index) / self.batch_size
        self.gt = self._label_path_from_index()
        self.provide_data = [('data', (batch_size, 1, data_shape[1], data_shape[0]))] + init_states  #+ [('label',(batch_size,num_label))]
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
                # for number in range(len(plate_str)):
                #     ret[number] = self.classes.index(plate_str[number]) + 1
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

# t = Image.open('bees.png').convert('L').resize((200,32),Image.BILINEAR)
import numpy as np
# img = np.array(t)
import matplotlib.pyplot as plt
# plt.imshow(img,'gray')
# plt.show()
with open('chinesechars.txt') as to_read: classes = list(to_read.read().strip())
BATCH_SIZE=100
num_hidden = 256
num_lstm_layer = 2
init_c = [('l%d_init_c' % l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer * 2)]
init_h = [('l%d_init_h' % l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer * 2)]
init_states = init_c + init_h
to_predict = OCRIter(BATCH_SIZE,classes,(200,32),9,init_states,False,False)
to_show = OCRIter(BATCH_SIZE,classes,(200,32),9,init_states,False,False)
# prob = model.predict(to_predict,return_data=True)
module = mx.mod.Module.load(
    'model/crnn_chinese_ctc',
    58,
    False,
    context = mx.gpu(0),
    data_names=('data','l0_init_c','l1_init_c','l2_init_c','l3_init_c','l0_init_h','l1_init_h','l2_init_h','l3_init_h'),
    label_names=('label',)
)

module.bind(to_predict.provide_data,to_predict.provide_label,for_training=False)
prob = module.predict(to_predict,1).asnumpy()


def __get_string(label_list):
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
        if l > 0 and l < (len(classes) + 1):
            c = classes[l - 1]
        else:
            c = ''
        s += c
    return s

for m in to_show:
    sb = m.data
    break
for i in range(prob.shape[0]):
    label_list = []
    for p in prob[i]:
        max_index = np.argsort(p)[::-1][0]
        label_list.append(max_index)
    result = __get_string(label_list)
    print(result)
    img = plt.imshow(sb[0][i][0].asnumpy(),cmap='gray')
    plt.show()
