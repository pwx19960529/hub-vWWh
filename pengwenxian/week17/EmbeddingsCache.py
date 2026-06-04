import numpy as np
import redis
from typing import List, Union
import hashlib


class EmbeddingsCache:
    def __init__(
            self,
            name: str, ttl: int=3600*24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.ttl = ttl

    @staticmethod
    def _normalize_embeddings(text: List[str], embedding: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        array = np.asarray(embedding, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if len(text) != array.shape[0]:
            raise ValueError("text 和 embedding 的数量不一致")
        return array

    def store(self, text: Union[List[str], str], embedding: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(text, str):
            text = [text]
        text = list(text)
        embedding = self._normalize_embeddings(text, embedding)

        try:
            with self.redis.pipeline() as pipe:
                for i, t in enumerate(text):
                    t_code = hashlib.md5(t.encode("utf-8")).hexdigest()
                    key = f"{self.name}:{t_code}"
                    value = embedding[i].astype(np.float32).tobytes()
                    pipe.setex(key, self.ttl, value)

                return pipe.execute()
        except Exception:
            return -1

    def delete(self, text: Union[List[str], str]):
        if isinstance(text, str):
            text = [text]

        try:
            key_list = []
            for t in text:
                t_code = hashlib.md5(t.encode("utf-8")).hexdigest()
                key_list.append(f"{self.name}:{t_code}")

            return self.redis.delete(*key_list)
        except Exception as e:
            print(f"Delete error: {e}")
            return -1

    def call(self, text: Union[List[str], str]):
        if isinstance(text, str):
            text = [text]

        try:
            key_list = []
            for t in text:
                t_code = hashlib.md5(t.encode("utf-8")).hexdigest()
                key_list.append(f"{self.name}:{t_code}")

            results = self.redis.mget(key_list)

            if not results:
                return None

            embeddings = []
            for result in results:
                if result is None:
                    embeddings.append(None)
                else:
                    embedding = np.frombuffer(result, dtype=np.float32)
                    embeddings.append(embedding)

            if len(embeddings) == 1:
                return embeddings[0]
            return embeddings

        except Exception:
            print("Error")
            return None

    get = call
    __call__ = call

if __name__ == "__main__":
    embed_cache = EmbeddingsCache(
        name="embedding_cache",
        ttl=360,
        redis_url="localhost",
    )

    def get_embedding(text):
        if isinstance(text, str):
            return np.random.rand(768).astype(np.float32)
        return np.random.rand(len(text), 768).astype(np.float32)

    print(embed_cache.store(text="hello world", embedding=get_embedding("hello world")))
    print(embed_cache.call(text="hello world"))
    print(embed_cache.delete(text="hello world"))
