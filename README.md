# crnn.mxnet
crnn in mxnet.can train with chinese characters

This project is the base module of the universal credentials' OCR with extra third party module.
I'll add **ANDROID SUPPORT** in soon.

REQUIREMENTS:
- MxNET:0.11.0 or above,lower would occur some errors.
- PIL:To generate the scene text for train
- tensorboard(OPTIONAL):you can monitor your network training.
- Fonts:I collected some FREE Chinese fonts,you can download and uncompress [fonts.7z](https://pan.baidu.com/s/1gfiq53P) into `./generate_data/fonts/`
- Backgrounds:you can collect by yourself.
> **WARNING**
> - DON'T USE MXNET WITH OPENCV,IT WILL BE WRONG!

HOW TO RUN
> python predictor.py


I just implement [CRNN](https://github.com/bgshih/crnn) with mxnet and there are some difference.
If you can't run this project fluently,please refer me in ISSUES,i'll check it out as soon as i can.