FROM python:3.9-slim

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install gcc -y \
&& apt-get clean

WORKDIR /app
VOLUME /app

ADD requirements.txt /app/requirements.txt
RUN pip install --user -r /app/requirements.txt

ADD src /app/src
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8080
ENTRYPOINT uvicorn src.main:app --reload --host 0.0.0.0 --port 8080 
