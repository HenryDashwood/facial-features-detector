from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import ImageInput, JsonOutput
from bentoml.frameworks.fastai2 import Fastai2ModelArtifact

from utils import get_y_func


@env(pip_dependencies=["imageio", "fastai", "timm"])
@artifacts([Fastai2ModelArtifact("learner")])
class PetRegressionService(BentoService):
    @api(input=ImageInput(), output=JsonOutput(), batch=False)
    def predict(self, image):
        preds = self.artifacts.learner.predict(image)
        output = preds[1].tolist()
        return output
