from bentoml import BentoService, api, env, artifacts 
from bentoml.artifact import FastaiModelArtifact
from bentoml.adapters import FastaiImageInput

@env(pip_dependencies=['gevent', 'fastai'])
@artifacts([FastaiModelArtifact('pet_regressor')])
class PetRegression(BentoService):
    
    @api(FastaiImageInput)
    def predict(self, image):
        result = self.artifacts.pet_regressor.predict(image)
        return result[2].tolist()
