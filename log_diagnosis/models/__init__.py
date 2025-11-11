from enum import Enum
from typing import Union

from log_diagnosis.models.model_zero_shot_classifer import ZeroShotModels
from log_diagnosis.models.model_zero_shot_classifer import ModelZeroShotClassifer

AllModels = Union[ZeroShotModels, str]

class ModelType(Enum):
    ZERO_SHOT = 'zero_shot'
    SIMILARITY = 'similarity'

class ModelManager:
    def __init__(self, type=ModelType.ZERO_SHOT, model: AllModels = ZeroShotModels.CROSSENCODER):
        if type == ModelType.ZERO_SHOT:
            self.model = ModelZeroShotClassifer(model)
        elif type == ModelType.SIMILARITY:
            # self.model = ModelSimilarity(model)
            pass
        else:
            raise ValueError(f"Invalid model type: {type}")

        # initialize model
        self.model.init_model()
        print("Initialized model")
    
    def classify_golden_signal(self, input: list[str], batch_size: int=32):
        return self.model.classify_golden_signal(input, batch_size)
    
    def classify_fault_category(self, input: list[str], batch_size: int=32):
        return self.model.classify_fault_category(input, batch_size)
