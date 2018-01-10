# crnn.mxnet
crnn in mxnet.can train with chinese characters

This project is the base module of the universal credentials' OCR with extra third party module.
~~I'll add **ANDROID SUPPORT** in soon.~~

**ANDROID HAD BEEN SUPPORTED YET**

REQUIREMENTS:
- MxNET:0.11.0 or above,lower would occur some errors.
- PIL:To generate the scene text for train
- Fonts:I collected some FREE Chinese fonts,you can download and uncompress [fonts.7z](https://pan.baidu.com/s/1gfiq53P) into `./generate_data/fonts/`
- Backgrounds:you can collect by yourself.

## HOW TO TRAIN
I got some taobao captchas recently,there is the train script to train taobao captcha.
```bash
unzip captcha.zip ./
# if you don't have NVIDIA-GPU in this machine,please remove the '--gpu' parameter
# parameter interpretation:
# name: the generated mode name,just the name.
# charset: the characters in all dataset.
# train_lst: the data URI and the label of data
# batch_size: the number of data to compute the loss every time
# seq_len: BiLSTM length(Advanced),relate with image width
# num_label: the maximum length of label
# imgH: image will resize to this height
# imgW: image will resize to this width
# learning_rate: lower for accurate,higher for speed.
python train.py --name taobao_captcha --charset digit.txt --train_lst taobao_captcha.csv --batch_size 32 --seq_len 12 --num_label 6 --imgH 30 --imgW 100 --gpu --learning_rate 0.001
```

## HOW TO RUN
> python predictor.py


I just implement [CRNN](https://github.com/bgshih/crnn) with mxnet and there are some difference.
If you can't run this project fluently,please refer me in ISSUES,i'll check it out as soon as i can.

The interface to use this model via mxnet of cpp is uploaded.And i'll update the handbook in some days.

## DOCKER
There are some guys need a portable environment,so i create a docker file.
you can predict via docker,only cpu version.If you need to train with gpu via docker,you'd need to modify the `config.mk`.
rebuild docker image by yourself,and use it via nvidia-docker.

```bash
# build
# docker build . -t novioleo/crnn-mxnet:0.11.0 
docker run -it -rm -v /path/to/your/project:/run novioleo/crnn-mxnet:0.11.0 python
```


## Note

IF YOU WANT A COMFORTABLE COMMUNICATION IN CHINESE,YOU CAN JOIN THE QQ GROUP:129075101