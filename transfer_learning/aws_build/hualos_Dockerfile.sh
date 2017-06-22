FROM ubuntu:latest
MAINTAINER Marco Willi "will5448@umn.edu"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install flask gevent urllib
ENTRYPOINT ["python"]
CMD ["api.py"]