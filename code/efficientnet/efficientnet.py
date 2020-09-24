import os
from pathlib import Path
from typing import Optional
import bentoml

from fastai.vision.all import (
    apply_init, aug_transforms, cnn_learner, create_head, CSVLogger, DataBlock,
    default_split, get_c, get_image_files, has_pool_type, ImageBlock, imagenet_stats,
    Learner, MSELossFlat, Normalize, num_features_model, partial, PointBlock,
    RandomSplitter, ranger, SaveModelCallback, tensor
)
from fastai.vision.learner import _update_first_layer
from fastcore.utils import remove_patches_path
import pandas as pd
import shutil
from timm import create_model
from torch.nn import Sequential
from torch.nn.init import kaiming_normal_
from typer import Typer

from bentoml_service import PetRegressionService


app = Typer()


def get_y_func(data, x):
    filename = str(x).split('/')[-1]
    zipped = zip(list(data.loc[filename])[0::2],
                 list(data.loc[filename])[1::2])
    return tensor(list(zipped))


def create_timm_body(arch: str, pretrained=True, cut=None, n_in=3):
    model = create_model(arch, pretrained=pretrained,
                         num_classes=0, global_pool='')
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


def create_timm_model(arch: str, n_out, cut=None, pretrained=True, n_in=3,
                      init=kaiming_normal_, custom_head=None, concat_pool=True,
                      **kwargs):
    body = create_timm_body(arch, pretrained, None, n_in)
    if custom_head is None:
        nf = num_features_model(Sequential(
            *body.children())) * (2 if concat_pool else 1)
        head = create_head(nf, n_out, concat_pool=concat_pool, **kwargs)
    else:
        head = custom_head
    model = Sequential(body, head)
    if init is not None:
        apply_init(model[1], init)
    return model


def timm_learner(dls, arch: str, loss_func=MSELossFlat(), opt_func=ranger,
                 pretrained=True, cut=None, splitter=None, y_range=None, config=None,
                 n_out=None, normalize=True, **kwargs):
    if config is None:
        config = {}
    if n_out is None:
        n_out = get_c(dls)
    if y_range is None and 'y_range' in config:
        y_range = config.pop('y_range')
    model = create_timm_model(
        arch, n_out, default_split, pretrained, y_range=y_range, **config)
    learn = Learner(dls, model, loss_func=loss_func,
                    opt_func=opt_func, splitter=default_split, **kwargs)
    if pretrained:
        learn.freeze()
    return learn


def create_dataloaders(images_path: str, labels_path: str, batch_size: int):
    data = pd.read_csv(labels_path, index_col='filename')[:40]

    dblock = DataBlock(
        get_items=get_image_files,
        blocks=(ImageBlock, PointBlock),
        splitter=RandomSplitter(valid_pct=0.15),
        get_y=partial(get_y_func, data),
        batch_tfms=[
            *aug_transforms(
                do_flip=False,
                size=(224, 224),
                max_rotate=15,
                max_lighting=0.5,
                max_warp=0.4
            ),
            Normalize.from_stats(*imagenet_stats)
        ]
    )

    dls = dblock.dataloaders(images_path, bs=batch_size)
    dls.c = dls.train.after_item.c

    return dls


@app.command()
def train(
    images_path: str = '../../data/subset_images',
    labels_path: str = '../../data/subset_labels.csv',
    model_type: str = 'efficientnet_b3a',
    batch_size: int = 64,
    frozen_epochs: int = 10,
    unfrozen_epochs: int = 0,
    frozen_lr: str = "3e-2",
    unfrozen_lr: str = "3e-4"
):
    dls = create_dataloaders(images_path, labels_path, batch_size)
    learn = timm_learner(dls, model_type)

    model_name = f"{model_type}_{batch_size}_{frozen_epochs}_{unfrozen_epochs}" \
        f"_{unfrozen_epochs}_{frozen_lr}"

    learn.fit_flat_cos(
        frozen_epochs,
        float(frozen_lr),
        wd=0.1,
        cbs=[
            SaveModelCallback(fname=model_name),
            CSVLogger(fname=f"{model_name}.csv")
        ]
    )


@app.command()
def get_results(model_path: str):
    pass


@app.command()
def bentoise(
    weights_path: str,
    images_path: str = '../../data/subset_images',
    labels_path: str = '../../data/subset_labels.csv',
    batch_size: int = 64,
    model_type: str = 'efficientnet_b3a'
):
    # bentoml_path = os.path.join(Path.home(), 'bentoml')
    # if os.path.isdir(bentoml_path):
    #     shutil.rmtree(bentoml_path)

    dls = create_dataloaders(images_path, labels_path, batch_size)
    learn = timm_learner(dls, model_type)

    learn.load(weights_path.split('/')[1].split('.')[0])

    svc = PetRegressionService()
    svc.pack('learner', learn)

    with remove_patches_path():
        saved_path = svc.save()

    print(f'{svc.name}:{svc.version}')
    return f'{svc.name}:{svc.version}'


if __name__ == "__main__":
    app()
