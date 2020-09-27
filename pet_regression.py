from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import ImageInput
from bentoml.frameworks.fastai2 import Fastai2ModelArtifact

from utils import get_y_func


@env(pip_dependencies=["fastai", "timm"])
@artifacts([Fastai2ModelArtifact("learner")])
class PetRegressionService(BentoService):
    @api(input=ImageInput(), batch=False)
    def predict(self, image):
        preds = self.artifacts.learner.predict(image)
        return preds
