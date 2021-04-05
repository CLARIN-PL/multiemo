FROM clarinpl/python:3.8

WORKDIR /home/worker
COPY ./src ./src 
COPY ./main.py .
COPY ./requirements.txt .
COPY ./config.ini .
COPY ./models ./models
COPY ./data ./data
COPY ./fastai_contrib ./fastai_contrib
COPY ./multifit ./multifit
RUN apt-get install -y build-essential libffi-dev
RUN pip install --index-url https://pypi.clarin-pl.eu/simple/ -r requirements.txt
RUN pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
CMD ["python3.8", "main.py"]
