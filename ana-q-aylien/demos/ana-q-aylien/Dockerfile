FROM python:3.8-bullseye

WORKDIR /srv

RUN ls .

RUN pip install --upgrade pip

ADD ana_q_aylien ./ana_q_aylien
ADD requirements.txt ./
ADD VERSION ./
ADD setup.py ./
ADD demos/ana-q-aylien/demo.py ./
ADD demos/ana-q-aylien/requirements_demo.txt ./
ADD demos/ana-q-aylien/Makefile ./

RUN ls .

RUN make dev
RUN pip install -r requirements_demo.txt

CMD make run
