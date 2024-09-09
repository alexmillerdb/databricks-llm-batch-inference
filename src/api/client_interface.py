from abc import ABC, abstractmethod
from typing import Tuple, Any

class APIClientInterface(ABC):
    @abstractmethod
    async def predict(self, text: str) -> Tuple[str, int]:
        pass
