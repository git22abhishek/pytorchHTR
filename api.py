from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
# from fastapi import UploadFile, File
import base64

from inference import base64str_to_numpy_array, infer

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


# define the Input class
class Input(BaseModel):
    base64str: str


@app.put("/predict")
async def predict(d: Input):
    image = base64str_to_numpy_array(d.base64str)
    text = infer(image)
    print(text)
    return {"text": text}
