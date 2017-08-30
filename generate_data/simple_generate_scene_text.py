# You can save all generated images in a single h5py file.
# I just want to have a look with my generated pics directly.

from PIL import Image, ImageDraw, ImageFont
import random
import datetime
import os
import argparse
from multiprocessing import Pool


def write_one(pic_width, pic_height, to_print_text, fonts, pic_dir, shape, m_count):
    to_return = []
    # pick a background pic from gallery randomly
    random_background_num = random.randint(0, background_length - 1)
    # rotate the image
    # random_background_rotate_degree = random.randint(0,360)
    background = Image \
        .open(os.path.join(gallery_dir, backgrounds[random_background_num])) \
        # .rotate(random_background_rotate_degree,expand=1)
    for i in range(len(fonts)):
        try:
            m_font = ImageFont.truetype(os.path.join(font_dir, fonts[i]), FONT_SIZE)
            # crop background as the synthText background randomly
            random_left = random.randint(0, background.size[0] - pic_width - 1)
            random_top = random.randint(0, background.size[1] - pic_height - 1)
            # draw text and rotate it
            scene_text = Image.new('RGBA', (pic_width, pic_height))
            draw = ImageDraw.Draw(scene_text)
            random_RGB = (random.randint(0, 1) * 255, random.randint(0, 1) * 255, random.randint(0, 1) * 255)
            draw.text((10, 10), to_print_text, fill=random_RGB, font=m_font)
            # text rotate degree
            random_text_rotate_degree = random.randint(-TEXT_ROTATE_DEGREE, TEXT_ROTATE_DEGREE)
            random_text_rotate_degree = random_text_rotate_degree if random_text_rotate_degree > 0 else 360 + random_text_rotate_degree
            w = scene_text.rotate(random_text_rotate_degree, expand=1)
            img = background.crop((random_left, random_top, random_left + pic_width, random_top + pic_height))
            # copy the text to background
            img.paste(w, mask=w)
            file_name = os.path.join(pic_dir, '%d_%s.jpg' % (m_count, fonts[i][:-4]))
            img.resize(shape, Image.BILINEAR).save(file_name)
            # to_write.write('%s\t%s\n' % (os.path.abspath(file_name), to_print_text))
            to_return.append('%s\t%s\n' % (os.path.abspath(file_name), to_print_text))
        except Exception as e:
            print(e)
    background.close()
    return to_return


def write2file_callback(lines):
    with open(label_file_path, 'a+') as to_write:
        to_write.writelines(lines)
        to_write.flush()


def write(count: int, mode: str, min_len: int, max_len: int, shape: tuple,worker_num:int):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    pic_dir = os.path.join(data_dir, mode)
    if not os.path.exists(pic_dir):
        os.mkdir(pic_dir)
    global label_file_path
    label_file_path = os.path.join(data_dir, mode + '.csv')
    if os.path.exists(label_file_path):
        os.remove(label_file_path)
    pool = Pool(worker_num)
    for m_count in range(count):
        to_print = [text[random.randint(0, text_len - 1)] for __ in range(random.randint(min_len, max_len))]
        to_print_text = ''.join(to_print)

        pic_width = len(to_print) * FONT_SIZE + BORDER_SIZE * 2
        pic_height = FONT_SIZE + BORDER_SIZE * 2
        pool.apply_async(write_one,(pic_width, pic_height, to_print_text, fonts, pic_dir, shape, m_count),callback=write2file_callback)
    pool.close()
    pool.join()
    print('finish')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='the dataset name')
    parser.add_argument('--num', required=True, type=int, help='how many do you want generate')
    parser.add_argument('--data_dir', required=True, help='where to store the generated images')
    parser.add_argument('--font_dir', required=True, help='the fonts folder location')
    parser.add_argument('--background_dir', required=True, help='the background images folder location')
    parser.add_argument('--font_size', default=32, type=int, help='the font size')
    parser.add_argument('--border_size', default=15, type=int, help='the border of text images')
    parser.add_argument('--trd', default=8, type=int, help='text rotate degree limit(clockwise & counter clockwise)')
    parser.add_argument('--charset', required=True, help='location of charset file,only one line!!!')
    parser.add_argument('--min_len', default=2, type=int, help='the min length of text in generated images ')
    parser.add_argument('--max_len', default=8, type=int, help='the max length of text in generated images ')
    parser.add_argument('--width', default=200, type=int, help='the width of the generated images')
    parser.add_argument('--height', default=32, type=int, help='the height of the generated images')
    parser.add_argument('--shuffle', action='store_true', help='shuffle the csv')
    parser.add_argument('--shuffle_count', default=10000,type=int, help='shuffle the csv')
    parser.add_argument('--worker', default=7, type=int, help='use multiprocess to speed up image generate(num of CPU cores minus 1 is RECOMMEND)')
    opt = parser.parse_args()
    print(opt)

    data_dir = opt.data_dir
    font_dir = opt.font_dir
    gallery_dir = opt.background_dir
    FONT_SIZE = opt.font_size
    BORDER_SIZE = opt.border_size
    TEXT_ROTATE_DEGREE = opt.trd
    shuffle_count = opt.shuffle_count

    fonts = [_ for _ in os.listdir(font_dir)]
    backgrounds = [_ for _ in os.listdir(gallery_dir)]

    with open(opt.charset) as to_read:
        text = to_read.read().strip()
    text_len = len(text)
    background_length = len(backgrounds)

    write(opt.num, opt.name, opt.min_len, opt.max_len, (opt.width, opt.height),opt.worker)
    """
    Because of the traditional shuffle the dataset method is loading all data into memory,
    then run random.shuffle.when the dataset is too large,the memory of poor machine will leak.
    And this part will make the imageIterator much more convenient to load data.
    """
    if opt.shuffle:
        with open(label_file_path[:-4]+'_tmp.csv','w') as to_write,open(label_file_path) as to_read:
            to_shuffle_list = []
            cnt = 0
            for m_line in to_read:
                to_shuffle_list.append(m_line)
                cnt += 1
                if cnt % shuffle_count == 0:
                    random.shuffle(to_shuffle_list)
                    to_write.writelines(to_shuffle_list)
                    to_shuffle_list.clear()
                    to_write.flush()
            if len(to_shuffle_list) > 0:
                random.shuffle(to_shuffle_list)
                to_write.writelines(to_shuffle_list)
                to_shuffle_list.clear()
        os.remove(label_file_path)
        os.rename(label_file_path[:-4]+'_tmp.csv',label_file_path)