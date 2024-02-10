# Dockerfile to be run on the cluster, has all project dependancies pre-installed
# Assumes that the image is being built from within the gpu-cluster folder
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG requirements

RUN echo $requirements | base64 --decode > req.txt
RUN python3 -m pip install -r req.txt
EXPOSE 6006/tcp 8888/tcp