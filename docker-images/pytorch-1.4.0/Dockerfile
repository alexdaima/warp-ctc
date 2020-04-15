FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
EXPOSE 8080 8000

ENV PATH="${PATH}:/opt/conda/lib/python3.7/site-packages/torch/share/cmake/Torch/TorchConfig.cmake"
ENV PATH="${PATH}:/opt/conda/pkgs/pytorch-1.4.0-py3.7_cuda10.1.243_cudnn7.6.3_0/lib/python3.7/site-packages/torch/share/cmake/Torch"

RUN apt-get update -y && \
    apt-get install git cmake -y && \
    git clone https://github.com/asciialex/warp-ctc.git /opt/warp-ctc && \
    cd /opt/warp-ctc/warp-ctc && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cd ../bindings/v1_4_0 && \
    python3 setup.py install --user

ENV PYTHONPATH=$PYTHONPATH:/opt/warp-ctc/warp-ctc/bindings/v1_4_0
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/warp-ctc/warp-ctc/build
