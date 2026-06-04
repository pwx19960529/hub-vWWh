import json

import redis
from typing import Optional, List, Union, Any, Dict
import Levenshtein
import numpy as np

from text_vectorizer import hash_embedding

class SemanticMessageHistory:
    def __init__(
            self,
            name: str, # 对话的名字，类似session id
            ttl: int=3600*24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold: float = 0.7,
            embedding_method=hash_embedding,
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
        self.history_key = f"semantic_history:{self.name}"

    def get_history(self):
        history = self.redis.get(self.history_key)
        if not history:
            return []
        return json.loads(history)

    def add_messages(self, messages: List[Dict[Any, Any]]):
        history = self.get_history()
        history.extend(messages)
        self.redis.setex(self.history_key, self.ttl, json.dumps(history, ensure_ascii=False))

    def add_message(self, message: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        if isinstance(message, dict):
            message = [message]
        self.add_messages(message)

    def get_recent(self, role: Optional[Union[str, List[str]]] = None, top_k: int = 10):
        history = self.get_history()
        if role:
            roles = {role} if isinstance(role, str) else set(role)
            selected_history = [message for message in history if message.get("role", "") in roles]
        else:
            selected_history = history

        if top_k:
            selected_history = selected_history[-top_k:]

        return selected_history

    def get_relevant(self, content: str, top_k: int = 10, role: Optional[Union[str, List[str]]] = None):
        history = self.get_history()
        if role:
            roles = {role} if isinstance(role, str) else set(role)
            history = [message for message in history if message.get("role", "") in roles]

        if not history:
            return []

        contents = [message.get("content", "") for message in history]
        query_vector = np.asarray(self.embedding_method(content), dtype=np.float32)
        if query_vector.ndim == 2:
            query_vector = query_vector[0]
        history_vectors = np.asarray(self.embedding_method(contents), dtype=np.float32)
        distances = np.linalg.norm(history_vectors - query_vector, axis=1)

        ranked_items = []
        for message, distance in zip(history, distances):
            lexical_score = Levenshtein.ratio(message.get("content", ""), content)
            ranked_items.append((message, float(distance), lexical_score))

        semantic_matches = [
            item for item in ranked_items
            if item[1] <= self.distance_threshold
        ]
        selected_history = semantic_matches or ranked_items
        selected_history.sort(key=lambda item: (item[1], -item[2]))

        result = []
        for message, distance, lexical_score in selected_history[:top_k]:
            enriched_message = dict(message)
            enriched_message["distance"] = distance
            enriched_message["score"] = lexical_score
            result.append(enriched_message)
        return result

    def delete_history(self, top_k=10):
        history = self.get_history()
        history = history[-top_k:]
        self.redis.setex(self.history_key, self.ttl, json.dumps(history, ensure_ascii=False))

    def clear_history(self):
        return self.redis.delete(self.history_key)

    __call__ = get_relevant


if __name__ == "__main__":
    history = SemanticMessageHistory(
        name="my-session",
        redis_url="localhost",
    )
    history.clear_history()
    history.add_messages([
        {"role": "user", "content": "hello, how are you?"},
        {"role": "llm", "content": "I'm doing fine, thanks."},
        {"role": "user", "content": "what is the weather going to be today?"},
        {"role": "llm", "content": "I don't know", "metadata": {"model": "gpt-4"}},
        {"role": "user", "content": "what is the weather going to be today?"},
    ])

    print("get_history", history.get_history())
    print("get_recent topk=1", history.get_recent(top_k=1))
    print("get_recent role=user", history.get_recent(role="user", top_k=1))

    print("\nget_relevant today", history.get_relevant("today",top_k=1))
    print("get_relevant today", history.get_relevant("thanks",top_k=1))
    #
    # history.clear_history()
    # history.add_message([
    #     {"role": "user", "content": "hello, how are you?"},
    #     {"role": "llm", "content": "I'm doing fine, thanks."},
    #     {"role": "user", "content": "what is the weather going to be today?"},
    # ])
    # history.gen()

