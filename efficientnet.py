from fastai.vision.all import (
    CSVLogger,
    Learner,
    MSELossFlat,
    SaveModelCallback,
    apply_init,
    create_head,
    default_split,
    get_c,
    has_pool_type,
    load_learner,
    num_features_model,
    ranger,
)
from fastai.vision.learner import _update_first_layer
from fastcore.utils import remove_patches_path
from timm import create_model
from torch.nn import Sequential, init
from typer import Typer

from pet_regression import PetRegressionService
from utils import create_dataloaders

app = Typer()


def create_timm_body(arch: str, pretrained=True, cut=None, n_in=3):
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool="")
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i, o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int):
        return Sequential(*list(model.children())[:cut])
    elif callable(cut):
        return cut(model)
    else:
        raise NameError("cut must be either integer or function")


def create_timm_model(
    arch: str,
    n_out,
    pretrained=True,
    cut=None,
    n_in=3,
    init=init.kaiming_normal_,
    custom_head=None,
    concat_pool=True,
    **kwargs,
):
    body = create_timm_body(arch, pretrained, cut, n_in)
    if custom_head is None:
        nf = num_features_model(Sequential(*body.children())) * (
            2 if concat_pool else 1
        )
        head = create_head(nf, n_out, concat_pool=concat_pool, **kwargs)
    else:
        head = custom_head
    model = Sequential(body, head)
    if init is not None:
        apply_init(model[1], init)
    return model


def timm_learner(
    dls,
    arch: str,
    loss_func=MSELossFlat(),
    opt_func=ranger,
    pretrained=True,
    cut=None,
    splitter=default_split,
    n_out=None,
):
    if n_out is None:
        n_out = get_c(dls)
    model = create_timm_model(arch, n_out, pretrained, cut)
    learn = Learner(dls, model, loss_func, opt_func, splitter=splitter)
    if pretrained:
        learn.freeze()
    return learn


@app.command()
def train(
    images_path: str = "data/resized_and_user_images",
    labels_path: str = "data/resized_and_user_labels.csv",
    model_type: str = "efficientnet_b3a",
    batch_size: int = 64,
    frozen_epochs: int = 5,
    unfrozen_epochs: int = 0,
    frozen_lr: str = "1e-2",
    unfrozen_lr: str = "1e-4",
):
    dls = create_dataloaders(images_path, labels_path, batch_size)
    learn = timm_learner(dls, model_type)

    if frozen_epochs > 0:
        model_name = (
            f"frozen_{model_type}_{batch_size}_{frozen_epochs}_{unfrozen_epochs}"
            f"_{unfrozen_epochs}_{frozen_lr}"
        )

        learn.fit_flat_cos(
            frozen_epochs,
            float(frozen_lr),
            wd=0.1,
            cbs=[
                SaveModelCallback(fname=model_name),
                CSVLogger(fname=f"logs/{model_name}.csv"),
            ],
        )

        learn.export(fname=f"models/{model_name}.pkl")

    if unfrozen_epochs > 0:
        model_name = (
            f"unfrozen_{model_type}_{batch_size}_{frozen_epochs}_{unfrozen_epochs}"
            f"_{unfrozen_epochs}_{frozen_lr}"
        )

        learn.unfreeze()

        learn.fit_flat_cos(
            unfrozen_epochs,
            float(unfrozen_lr),
            wd=0.1,
            cbs=[
                SaveModelCallback(fname=model_name),
                CSVLogger(fname=f"logs/{model_name}.csv"),
            ],
        )

        learn.export(fname=f"models/{model_name}.pkl")


@app.command()
def bentoise(weights_path: str):
    learn = load_learner(fname=weights_path)
    svc = PetRegressionService()
    svc.pack("learner", learn)
    with remove_patches_path():
        svc.save()


if __name__ == "__main__":
    app()
