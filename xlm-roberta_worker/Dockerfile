FROM clarinpl/python:3.8

WORKDIR /home/worker
COPY ./main.py .
COPY ./requirements.txt .
COPY ./config.ini .
COPY ./src ./src
RUN apt-get install -y build-essential libffi-dev
RUN pip install --index-url https://pypi.clarin-pl.eu/simple/ -r requirements.txt
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
CMD ["python3.8", "main.py"]
