import os
import torch
from argparse import ArgumentParser

from src.download import get_dataset
from src.lightning import QQPLightning

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), '/data/'))
    parser.add_argument('--dataset_url', type=str, default='http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv')
    parser.add_argument('--download_if_not_exist', type=bool, default=True)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/')

    parser.add_argument('--project_name', type=str, default='QQPBert')
    parser.add_argument('--gpu', nargs='*', type=int, default=[0] if torch.cuda.is_available() else None)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_norm', type=float, default=3.)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--n_epochs', type=int, default=5)

    parser.add_argument('--last_n_acc', type=int, default=1000)

    parser.add_argument('--model_name', type=str, default='bert-base-cased')
    parser.add_argument('--comet_api_key', type=str, default='Ul3oe90oSUc9wRnxdv2YLzVYQ')

    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--precision', type=int, default=16)

    parser.add_argument('--val_check_interval', type=float, default=0.25)

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)

    for file in ['train.tsv', 'validation.tsv', 'test.tsv']:
        if not os.path.isfile(os.path.join(args.data_dir, file)) and args.download_if_not_exist:
            get_dataset(url=args.dataset_url, data_dir=args.data_dir)
            break

    model = QQPLightning(hparams=args)

    if args.comet_api_key:
        logger = CometLogger(
            api_key=args.comet_api_key,
            project_name=args.project_name
        )
    else:
        logger = TensorBoardLogger(save_dir=os.getcwd(), name=args.project_name)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), args.checkpoint_path),
        save_top_k=2,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    trainer = pl.Trainer(logger=logger,
                         use_amp=args.use_amp,
                         precision=args.precision,
                         gradient_clip=args.max_norm,
                         gpus=args.gpu,
                         val_check_interval=args.val_check_interval,
                         num_sanity_val_steps=0,
                         checkpoint_callback=checkpoint_callback)

    trainer.fit(model)
