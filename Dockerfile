FROM python:3.9.15
WORKDIR /tadpole
COPY parsers/. /tadpole/parsers/.
COPY requirements.txt /tadpole/requirements.txt


RUN python -m pip install -r /tadpole/requirements.txt
CMD [ "python", "pajtonparser.py" ]