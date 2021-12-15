from src.multiemo_labse import MultiEmoLabse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

start_time = time.time()
multiemo = MultiEmoLabse()
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


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(request: TextRequest):
    start_time = time.time()
    results = multiemo.predict(request.text)
    print('Processing text took: ', time.time() - start_time)
    return results
