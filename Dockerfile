FROM python:3.9

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install git-lfs
RUN apt-get install -y python3-h5py 

WORKDIR /back_end

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

