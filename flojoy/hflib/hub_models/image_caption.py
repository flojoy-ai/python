from enum import Enum, auto
from typing import NamedTuple
import transformers as ts

from ..hub_models import HubModel, HubModelFactory


class ImageCaptionModels(Enum):
    NLP_CONNECT_VIT_GPT2 = auto()


class ImageCaptionModel(NamedTuple):
    model: ts.VisionEncoderDecoderModel
    feature_extractor: ts.ViTImageProcessor
    tokenizer: ts.AutoTokenizer


@HubModelFactory.register(ImageCaptionModels.NLP_CONNECT_VIT_GPT2)
class NLPConnectVitGPT2(HubModel):
    def __init__(self):
        self._model, self._feature_extractor, self._tokenizer = None, None, None
        self._cached = False

    def download_and_cache(self):
        self._model = ts.VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning", revision="dc68f91"
        )
        self._feature_extractor = ts.ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning", revision="dc68f91"
        )
        self._tokenizer = ts.AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning", revision="dc68f91"
        )
        self._cached = True

    @property
    def cached(self) -> bool:
        return self._cached

    def _get_executable_model(self) -> ImageCaptionModel:
        return ImageCaptionModel(
            model=self._model,
            feature_extractor=self._feature_extractor,
            tokenizer=self._tokenizer,
        )
