import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import redis

from text_vectorizer import hash_embedding


@dataclass
class Route:
    name: str
    references: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance_threshold: float = 0.3

class SemanticRouter:
    def __init__(
            self,
            name: str = "topic-router",
            routes: Optional[List[Route]] = None,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            ttl: int = 3600 * 24,
            embedding_method=hash_embedding,
            distance_threshold: float = 0.3,
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.ttl = ttl
        self.embedding_method = embedding_method
        self.distance_threshold = distance_threshold
        self.routes: List[Route] = routes or []

    def _cache_key(self, question: str) -> str:
        question_code = hashlib.md5(question.encode("utf-8")).hexdigest()
        return f"{self.name}:route_cache:{question_code}"

    def add_route(
            self,
            questions: Optional[List[str]] = None,
            target: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            distance_threshold: Optional[float] = None,
            route: Optional[Route] = None,
    ):
        if route is not None:
            self.routes.append(route)
            return route

        if not questions or not target:
            raise ValueError("questions 和 target 不能为空")

        route = Route(
            name=target,
            references=list(questions),
            metadata=metadata or {},
            distance_threshold=distance_threshold if distance_threshold is not None else self.distance_threshold,
        )
        self.routes.append(route)
        return route

    def route(self, question: str):
        cached_result = self.redis.get(self._cache_key(question))
        if cached_result:
            return json.loads(cached_result)

        if not self.routes:
            return None

        question_embedding = np.asarray(self.embedding_method(question), dtype=np.float32)
        if question_embedding.ndim == 2:
            question_embedding = question_embedding[0]

        best_result = None
        for route in self.routes:
            reference_embeddings = np.asarray(self.embedding_method(route.references), dtype=np.float32)
            distances = np.linalg.norm(reference_embeddings - question_embedding, axis=1)
            best_distance = float(np.min(distances))
            if best_distance > route.distance_threshold:
                continue

            if best_result is None or best_distance < best_result["distance"]:
                best_result = {
                    "name": route.name,
                    "metadata": route.metadata,
                    "distance": best_distance,
                }

        if best_result is not None:
            self.redis.setex(
                self._cache_key(question),
                self.ttl,
                json.dumps(best_result, ensure_ascii=False)
            )
        return best_result

    __call__ = route


if __name__ == "__main__":
    routes = [
        Route(
            name="greeting",
            references=["Hi, good morning", "Hi, good afternoon", "hello", "hi"],
            metadata={"type": "greeting"},
            distance_threshold=0.3,
        ),
        Route(
            name="refund",
            references=["如何退货", "怎么退款", "我要申请退款"],
            metadata={"type": "refund"},
            distance_threshold=0.3,
        ),
    ]
    router = SemanticRouter(
        routes=routes,
        redis_url="localhost",
    )

    print(router("Hi, good morning"))
