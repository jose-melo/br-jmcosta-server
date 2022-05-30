FROM python:3.6

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /back_end

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
