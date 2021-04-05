FROM clarinpl/python:3.8

WORKDIR /home/worker
COPY ./src ./src 
COPY ./main.py .
COPY ./requirements.txt .
COPY ./config.ini .
RUN apt-get install -y build-essential libffi-dev
RUN pip install --index-url https://pypi.clarin-pl.eu/simple/ -r requirements.txt
RUN wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
CMD ["python3.8", "main.py"]
