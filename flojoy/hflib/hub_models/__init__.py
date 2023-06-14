from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Any


class HubModel(metaclass=ABCMeta):
    @abstractmethod
    def download_and_cache(self) -> None:
        pass

    @property
    @abstractmethod
    def cached(self) -> bool:
        pass

    @abstractmethod
    def _get_executable_model(self) -> Any:
        pass

    def get_executable_model(self) -> Any:
        if not self.cached:
            raise ValueError("Model not cached. Call download_and_cache() first.")
        return self._get_executable_model()


class HubModelFactory:
    """Factory class for HuggingFace hub models

    Example usage:

    ```python
    from flojoy.hflib.hub_models import HubModelFactory, ImageCaptionModels

    factory = HubModelFactory()
    hub_model = factory.create_model(ImageCaptionModels.NLP_CONNECT_VIT_GPT2)
    hub_model.download_and_cache() # Download the model and cache it locally
    model = hub_model.get_executable_model() # Get the executable model
    # Do work
    ```

    """

    _creators = {}

    @classmethod
    def register(cls, model_type: Enum):
        def decorator(creator_class):
            if model_type in cls._creators:
                raise Exception(f"Model type {model_type} already registered.")
            cls._creators[model_type] = creator_class
            return creator_class

        return decorator

    @classmethod
    def create_model(cls, model_type: Enum):
        if model_type not in cls._creators:
            raise ValueError(f"Invalid model type: {model_type}")
        return cls._creators[model_type]()


from .image_caption import ImageCaptionModels
