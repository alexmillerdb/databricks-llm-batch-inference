from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Union

class APIClientInterface(ABC):
    @abstractmethod
    async def async_predict(self, text: str) -> Union[Tuple[str, int], Tuple[List[float], int]]:
        """
        Asynchronously predicts based on the provided text input.
        Args:
            text (str): The input text for which the prediction is to be made.
        Returns:
            Union[Tuple[str, int], Tuple[List[float], int]]: A tuple containing the prediction result and a status code.
                - If the prediction is a string, the tuple will be (prediction: str, status_code: int).
                - If the prediction is a list of floats, the tuple will be (prediction: List[float], status_code: int).
        """
        
        pass

    @abstractmethod
    def predict(self, text: str) -> Union[Tuple[str, int], Tuple[List[float], int]]:
        """
        Predicts the output based on the given input text.
        Args:
            text (str): The input text for which the prediction is to be made.
        Returns:
            Union[Tuple[str, int], Tuple[List[float], int]]: A tuple containing the prediction result and a status code.
                - If the prediction is a string, the tuple will be (prediction_str, status_code).
                - If the prediction is a list of floats, the tuple will be (prediction_list, status_code).
        """

        pass