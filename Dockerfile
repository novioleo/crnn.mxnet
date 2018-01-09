FROM mxnet/python:0.11.0
MAINTAINER nsnovio@gmail.com
WORKDIR /opt
RUN git clone https://github.com/baidu-research/warp-ctc.git
RUN apt-get install -y cmake
RUN cd warp-ctc && mkdir build && cd build && cmake .. && make -j4
WORKDIR /mxnet
COPY config.mk ./config.mk
RUN make -j7
WORKDIR /
#COPY Anaconda3-4.2.0-Linux-x86_64.sh ./
 RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-4.2.0-Linux-x86_64.sh
RUN bash Anaconda3-4.2.0-Linux-x86_64.sh -b -p /anaconda3
ENV PATH="/anaconda3/bin:${PATH}"
RUN rm -f Anaconda3-4.2.0-Linux-x86_64.sh
RUN ln -sf /opt/warp-ctc/build/libwarpctc.so /anaconda3/lib/
RUN cd /mxnet/python && pip install --upgrade pip && pip install -e .
