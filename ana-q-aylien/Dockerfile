FROM python:3.10-bullseye

WORKDIR /srv

RUN pip install --upgrade pip

ADD ana_q_aylien ./ana_q_aylien
ADD requirements.txt ./
ADD VERSION ./
ADD setup.py ./
ADD Makefile ./

RUN make dev

CMD make run
