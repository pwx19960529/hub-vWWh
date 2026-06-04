import os
import numpy as np
import redis
from typing import List, Union, Callable, Any, Dict
import faiss

from text_vectorizer import hash_embedding

class SemanticCache:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any] = hash_embedding,
            ttl: int=3600*24, # 过期时间
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold=0.1
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method
        self.index_path = f"{self.name}.index"
        self.prompt_list_key = f"{self.name}:prompts"

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = None

    def _response_key(self, prompt: str) -> str:
        return f"{self.name}:response:{prompt}"

    @staticmethod
    def _to_2d_array(embedding: Any) -> np.ndarray:
        array = np.asarray(embedding, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]):
        if isinstance(prompt, str):
            prompt = [prompt]
            response = [response]
        else:
            response = list(response)

        prompt = list(prompt)
        if len(prompt) != len(response):
            raise ValueError("prompt 和 response 的数量不一致")

        embedding = self._to_2d_array(self.embedding_method(prompt))
        if self.index is None:
            self.index = faiss.IndexFlatL2(embedding.shape[1])

        self.index.add(embedding)
        faiss.write_index(self.index, self.index_path)

        try:
            with self.redis.pipeline() as pipe:
                for q, a in zip(prompt, response):
                    pipe.setex(self._response_key(q), self.ttl, a)
                    pipe.rpush(self.prompt_list_key, q)

                return pipe.execute()
        except Exception:
            import traceback
            traceback.print_exc()
            return -1

    def check(self, prompt: str, top_k: int = 5) -> Union[List[Dict[str, Any]], None]:
        prompt_count = self.redis.llen(self.prompt_list_key)
        if self.index is None or prompt_count == 0:
            return None

        embedding = self._to_2d_array(self.embedding_method(prompt))
        k = min(max(top_k, 1), prompt_count)
        distances, indices = self.index.search(embedding, k=k)
        stored_prompts = self.redis.lrange(self.prompt_list_key, 0, -1)
        matched_results = []

        for distance, index in zip(distances[0], indices[0]):
            if index < 0 or index >= len(stored_prompts):
                continue
            if distance > self.distance_threshold:
                continue

            matched_prompt = stored_prompts[index].decode("utf-8")
            cached_response = self.redis.get(self._response_key(matched_prompt))
            if cached_response is None:
                continue

            matched_results.append(
                {
                    "prompt": matched_prompt,
                    "response": cached_response.decode("utf-8"),
                    "distance": float(distance),
                }
            )

        if not matched_results:
            return None
        return matched_results

    def call(self, prompt: str, top_k: int = 5):
        return self.check(prompt=prompt, top_k=top_k)

    def clear_cache(self):
        prompts = self.redis.lrange(self.prompt_list_key, 0, -1)
        response_keys = [self._response_key(prompt.decode("utf-8")) for prompt in prompts]
        if response_keys:
            self.redis.delete(*response_keys)
        self.redis.delete(self.prompt_list_key)
        if os.path.exists(self.index_path):
            os.unlink(self.index_path)
        self.index = None

    __call__ = check

if __name__ == "__main__":
    embed_cache = SemanticCache(
        name="semantic_cache",
        ttl=360,
        redis_url="localhost",
    )

    embed_cache.clear_cache()

    embed_cache.store(prompt="hello world", response="hello world1232")
    print(embed_cache.check(prompt="hello world"))

    embed_cache.store(prompt="hello my bame", response="nihao")
    print(embed_cache("hello world"))
