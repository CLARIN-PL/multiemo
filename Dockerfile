FROM python:3.9-bullseye

RUN apt update
RUN apt install -y p7zip-full

RUN mkdir /opt/multiemo
COPY . /opt/multiemo
WORKDIR /opt/multiemo

RUN pip install /opt/multiemo/mosestokenizer-1.1.0.tar.gz
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN chmod +x ./entrypoint.sh
CMD ./entrypoint.sh
