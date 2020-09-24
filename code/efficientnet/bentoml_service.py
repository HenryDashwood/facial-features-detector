from bentoml import BentoService, api, artifacts, env
from bentoml.frameworks.fastai2 import Fastai2ModelArtifact
from bentoml.adapters import ImageInput

from efficientnet import get_y_func


@env(pip_dependencies=['fastai'])
@artifacts([Fastai2ModelArtifact('learner')])
class PetRegressionService(BentoService):

    @api(input=ImageInput(), batch=True)
    def predict(self, image):
        preds = self.artifacts.regressor.predict(image)
        return preds
