FROM python:3.6

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

#RUN apt-get install git-lfs

WORKDIR /back_end

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

#RUN git init
#RUN git remote get-url origin 
#RUN git pull origin master
#RUN git lfs pull
#
