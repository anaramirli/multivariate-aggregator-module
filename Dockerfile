FROM python:3.9.13-slim

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install gcc -y \
&& apt-get clean

WORKDIR /app
ADD requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
&& pip install --user -r /app/requirements.txt

ADD src /app/src
ENV PATH=/root/.local/bin:$PATH
RUN mkdir /app/data
EXPOSE 8080
ENTRYPOINT uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
