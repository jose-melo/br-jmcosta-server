FROM python:3.6

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install git-lfs
RUN apt-get install -y python3-h5py 

WORKDIR /back_end

RUN git init
RUN git remote add origin https://github.com/jose-melo/br-jmcosta-server.git
RUN git pull origin master
RUN git lfs pull


COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt



ENTRYPOINT ["gunicorn", "--log-level", "INFO", "-b", ":8296", "-t", "120", "run:APP"]
