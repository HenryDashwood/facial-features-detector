from fastai.vision.all import (
    DataBlock,
    ImageBlock,
    Normalize,
    PointBlock,
    RandomSplitter,
    aug_transforms,
    get_image_files,
    imagenet_stats,
    partial,
    tensor,
)
from pandas import read_csv


def get_y_func(data, x):
    filename = str(x).split("/")[-1]
    zipped = zip(list(data.loc[filename])[0::2], list(data.loc[filename])[1::2])
    return tensor(list(zipped))


def create_dataloaders(images_path: str, labels_path: str, batch_size: int):
    data = read_csv(labels_path, index_col="filename")

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
                max_warp=0.4,
            ),
            Normalize.from_stats(*imagenet_stats),
        ],
    )

    dls = dblock.dataloaders(images_path, bs=batch_size)
    dls.c = dls.train.after_item.c

    return dls
