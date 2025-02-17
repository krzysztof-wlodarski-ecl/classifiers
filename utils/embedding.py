import itertools
import os
import pickle

import torch
import torchvision.transforms.v2 as T
from torchvision import io
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import CLIPModel


CACHE_FILE = "../embed.cache"
class EmbeddingEngine:
    """
    Class for encapsulating image embedding. Embeds are cached.
    """
    def __init__(self):
        print("Loading model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.cache = {}

        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as fp:
                self.cache = pickle.load(fp)

        self.transforms = T.Compose([
            T.ToDtype(torch.float32, scale=True),
            T.Resize(224, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def save_cache(self):
        with open(CACHE_FILE, "wb") as fp:
            pickle.dump(self.cache, fp)


    def embed_single(self, filepath):
        """
        Embed single image file. Image must be valid. Returns raw numpy vector.
        Returns cached embedding, otherwise add to cache. It's recommended to call save_cache() to speed up
        subsequent runs.
        """
        if self.cache.get(filepath) is not None:
            return self.cache[filepath]
        vector = self.__embed_single_nocache(filepath)
        self.cache[filepath] = vector
        return vector

    def embed_path(self, root, display_tqdm=True):
        """
        Performs embedding for all images recursively. Adds to cache and saves.
        Returns dict of path and vector.
        """
        images = []
        root = os.path.abspath(root)
        for dir, dirs, files in os.walk(root):
            images.extend([os.path.join(dir, file) for file in files if file.split(".")[-1].lower() in ["jpg", "png", "jpeg"]])

        iterator = tqdm(images) if display_tqdm else images
        vectors = [self.embed_single(file) for file in iterator]
        results = {images[i]: vectors[i] for i in range(len(images))}
        self.save_cache()
        return results

    def __embed_single_nocache(self, filepath):
        with torch.no_grad():
            with open(filepath, "rb") as fp:
                contents = bytearray(fp.read())
            image = io.decode_image(torch.frombuffer(contents, dtype=torch.uint8)).to(self.device)
            processed = self.transforms(image).unsqueeze(0)
            vector = self.model.get_image_features(processed).cpu().numpy().flatten()
        return vector


