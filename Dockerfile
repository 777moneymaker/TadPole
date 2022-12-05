FROM python:3.9.15
WORKDIR /tadpole
COPY parsers/pajtonparser/. /tadpole/

RUN python -m pip install alive-progress numpy
CMD [ "python", "pajtonparser.py" ]