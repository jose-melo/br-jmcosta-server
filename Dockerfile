FROM python:3.6

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install git-lfs

WORKDIR /back_end

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN cd /backend
RUN git init .
RUN git remote add origin https://github.com/jose-melo/br-jmcosta-server.git
RUN git pull master
RUN git lfs pull

