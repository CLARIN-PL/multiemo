FROM clarinpl/python:3.8

WORKDIR /home/worker
COPY ./src ./src 
COPY ./main.py .
COPY ./requirements.txt .
COPY ./config.ini .
COPY ./models ./models
RUN apt-get install -y build-essential libffi-dev
RUN pip install --index-url https://pypi.clarin-pl.eu/simple/ -r requirements.txt
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m laserembeddings download-models
RUN python -m nltk.downloader punkt
CMD ["python3.8", "main.py"]
