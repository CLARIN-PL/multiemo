from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

from src.multiemo_labse import MultiEmoLabse
from src.multiemo_laser import MultiEmoLaser

start_time = time.time()
multiemo = MultiEmoLabse()
multiemo_laser = MultiEmoLaser()
print('MultiEmo loading took: ', time.time() - start_time)

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: list


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(request: TextRequest):
    start_time = time.time()
    results = multiemo.predict(request.text)
    print('Processing text took: ', time.time() - start_time)
    return results

@app.post("/predict_laser")
def predict(request: TextRequest):
    start_time = time.time()
    results = multiemo_laser.predict(request.text)
    print('Processing text took: ', time.time() - start_time)
    return results

@app.post("/batch_predict")
def predict(request: BatchRequest):
    start_time = time.time()
    results = multiemo.batch_predict(request.texts)
    print('Processing text took: ', time.time() - start_time)
    return results
