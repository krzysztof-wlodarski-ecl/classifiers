"""
Serves images.
This is required for displaying images, because browser security prevents displaying images from a local file system
on a remote webpage (including localhost host).
"""
import os.path

from fastapi import FastAPI, HTTPException
from fastapi.openapi.models import Response
import uvicorn
from starlette.responses import FileResponse

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root(path):
    """
    Accept full path to a file to simply host it.
    Security: yes.
    """
    if path.split(".")[-1].lower() not in ["jpg", "png", "jpeg"]:
        raise HTTPException(403, "Only image files allowed")

    if not os.path.isfile(path):
        raise HTTPException(403, "Not a file")

    with open(path, "rb") as fp:
        file = fp.read()
    return FileResponse(path)

uvicorn.run(app, port=9000)