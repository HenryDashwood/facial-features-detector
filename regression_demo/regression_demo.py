from bentoml import BentoService, api, env, artifacts 
from bentoml.artifact import FastaiModelArtifact
from bentoml.handlers import FastaiImageHandler

@env(pip_dependencies=['fastai'])
@artifacts([FastaiModelArtifact('regression_demoer')])
class RegressionDemo(BentoService):
    
    @api(FastaiImageHandler)
    def predict(self, image):
        result = self.artifacts.regression_demoer.predict(image)
        return str(result)
