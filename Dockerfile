FROM python:3.6

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install git-lfs

WORKDIR /back_end

RUN git init
RUN git remote add origin https://github.com/jose-melo/br-jmcosta-server.git
#RUN git fetch origin -p
#RUN git checkout origin/master
#RUN git checkout -b master
#RUN git restore .
RUN git pull origin master
RUN git lfs pull


COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


#COPY . .

ENTRYPOINT ["gunicorn", "--log-level", "INFO", "-b", ":8296", "-t", "120", "run:APP"]
