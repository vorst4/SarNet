from util.base_obj import BaseObj
from typing import Optional


class LearningRate(BaseObj):
    class Settings:
        def __init__(
                self,
                initial: Optional[float],
                step_size: int,
                gamma: float
        ):
            """
            Learning rate settings, learning rate starts at <initial> and
            decays every <step_size> epochs by factor <gamma>. If
            initial=None then the lr_ideal as provided in the model will be
            used
            """
            self.initial = initial
            self.step_size = step_size
            self.gamma = gamma
