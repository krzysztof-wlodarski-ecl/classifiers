"""
Predict user's choices. This is a server part, run page.html for frontend.
Requires your own set of images.
"""

import itertools
import os
import random
import threading
from typing import Literal

import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tqdm import tqdm

false_path = "../images/sharpened/clean"
false_path = os.path.abspath(false_path)
false_set = [os.path.join(false_path, f) for f in os.listdir(false_path) if f.split(".")[-1].lower() in ["png", "jpg", "jpeg"]]
false_set = random.choices(false_set, k=200)

true_path = "../images/sharpened/sharpened"
true_path = os.path.abspath(true_path)
true_set = [os.path.join(true_path, f) for f in os.listdir(true_path) if f.split(".")[-1].lower() in ["png", "jpg", "jpeg"]]
true_set = random.choices(true_set, k=200)

all_keys = list(itertools.chain(false_set))
random.shuffle(all_keys)


# ---------------------
# embedding thread
embeds = {}
def embedding_worker(stop:threading.Event):
    from utils import embedding
    engine = embedding.EmbeddingEngine()

    print("Embedding worker: starting")
    for key in tqdm(all_keys):
        if stop.is_set():
            break
        vector = engine.embed_single(key)
        embeds[key] = vector
    engine.save_cache()
    print("Embedding worker: finished")


stop_worker = threading.Event()
embedding_thread = threading.Thread(target=embedding_worker, args=(stop_worker,))
embedding_thread.start()

# ------------
# classifier

def classifier_factory():
    # clf = SGDClassifier(loss="log_loss")
    return PassiveAggressiveClassifier()
    # return SVC(kernel="rbf", C=1.0, gamma="scale")
    # return GaussianNB()


clf = classifier_factory()
first_batch = True
batch = []

def fit():
    global first_batch
    global clf
    global batch
    labels, vectors = zip(*batch)
    if first_batch and set(labels) != {0, 1}:
        # don't fit yet, we require both labels in initial fit
        return
    first_batch = False
    vectors = np.array(vectors)
    # vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    clf.partial_fit(vectors, labels, classes=[0,1])
    batch = []

# ------------
# api

import fastapi
import uvicorn

api = fastapi.FastAPI()
from fastapi.middleware.cors import CORSMiddleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/")
async def hello():
    return "hello"

@api.get("/get-done")
async def get_done():
    return list(embeds.keys())

@api.get("/mark")
async def mark(label: Literal["0", "1"],
               key: str):
    if key not in embeds.keys():
        return "Key not found"
    e = embeds[key]
    label = int(label)
    batch.append((label, e))
    fit()


@api.get("/get-predictions")
async def get_predictions():
    if first_batch:
        return []
    embeds_ordered = [embeds[key] for key in all_keys]
    predictions = clf.predict(embeds_ordered)
    z = list(zip(all_keys, predictions.tolist()))
    return z


@api.get("/demo-mark")
async def demo_mark():
    """ Convenience utility for testing api without frontend """
    key = random.choice(all_keys)
    label = random.randint(0, 1)
    await mark(label, key)
    return key, label


@api.get("/reset")
async def reset():
    global clf
    global first_batch
    clf = classifier_factory()
    first_batch = True
    batch.clear()


uvicorn.run(api)

stop_worker.set()
embedding_thread.join()