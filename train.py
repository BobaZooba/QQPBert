import os
import torch
import logging
from argparse import ArgumentParser

from src.download import get_dataset
from src.lightning import QQPLightning

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def train():

    logger = logging.getLogger(__file__)

    parser = ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset_url', type=str, default='http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv')
    parser.add_argument('--download_if_not_exist', type=bool, default=True)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/')

    parser.add_argument('--project_name', type=str, default='QQPBert')
    parser.add_argument('--gpu', nargs='*', type=int, default=[0] if torch.cuda.is_available() else None)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_norm', type=float, default=2.)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--n_epochs', type=int, default=5)

    parser.add_argument('--last_n_acc', type=int, default=1000)

    parser.add_argument('--model_name', type=str, default='bert-base-cased')
    parser.add_argument('--from_nsp', type=bool, default=False)
    parser.add_argument('--comet_api_key', type=str, default='')

    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--precision', type=int, default=16)

    parser.add_argument('--val_check_interval', type=float, default=0.25)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)

    for file in ['train.tsv', 'validation.tsv', 'test.tsv']:
        if not os.path.isfile(os.path.join(args.data_dir, file)) and args.download_if_not_exist:
            logger.info('Start downloading')
            get_dataset(url=args.dataset_url, data_dir=args.data_dir)
            break

    logger.info('Model init')
    model = QQPLightning(hparams=args)
    logger.info('Model has %s parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.comet_api_key:
        pl_logger = CometLogger(
            api_key=args.comet_api_key,
            project_name=args.project_name
        )
        pl_logger.experiment.log_parameters(args.__dict__)
        logger.info('Use comet logger')
    else:
        pl_logger = TensorBoardLogger(save_dir=os.getcwd(), name=args.project_name)
        logger.info('Use tensorboard logger')

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), args.checkpoint_path),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    logger.info('Trainer init')

    trainer = pl.Trainer(logger=pl_logger,
                         max_epochs=args.n_epochs,
                         use_amp=args.use_amp,
                         precision=args.precision,
                         gradient_clip=args.max_norm,
                         gpus=args.gpu,
                         val_check_interval=args.val_check_interval,
                         num_sanity_val_steps=0,
                         checkpoint_callback=checkpoint_callback)

    logger.info('Start training')
    trainer.fit(model)


if __name__ == "__main__":
    train()
