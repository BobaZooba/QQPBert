import os
import torch
from torch.utils.data import DataLoader
import transformers
from src.data import Tokenizer, PairedData
import pytorch_lightning as pl


class QQPLightning(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.tokenizer = Tokenizer(self.hparams.model_name)
        if self.hparams.from_nsp:
            self.bert = transformers.BertForNextSentencePrediction.from_pretrained(self.hparams.model_name)
        else:
            self.bert = transformers.BertForSequenceClassification.from_pretrained(self.hparams.model_name)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.correct_predictions = list()
        self.all_predictions = list()

    def forward(self, token_ids, token_type_ids, pad_mask):
        prediction = self.bert.forward(input_ids=token_ids,
                                       attention_mask=pad_mask,
                                       token_type_ids=token_type_ids)[0]

        return prediction

    def training_step(self, batch, batch_idx):
        (token_ids, token_type_ids, pad_mask), target = batch

        prediction = self.forward(token_ids, token_type_ids, pad_mask)

        loss = self.criterion(prediction, target)

        log = {'train_loss': loss}

        prediction_prob = torch.exp(torch.log_softmax(prediction.detach().cpu(), 1))[:, 1]
        binary_predictions = (prediction_prob > 0.5).float()
        correct_predictions = (binary_predictions == target.detach().cpu()).sum().float()

        self.correct_predictions.append(correct_predictions.item())
        self.all_predictions.append(target.size(0))

        if self.global_step >= self.hparams.last_n_acc:

            self.correct_predictions = self.correct_predictions[-self.hparams.last_n_acc:]
            self.all_predictions = self.all_predictions[-self.hparams.last_n_acc:]

            accuracy = sum(self.correct_predictions) / sum(self.all_predictions)

            log['train_accuracy'] = accuracy

        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        (token_ids, token_type_ids, pad_mask), target = batch

        with torch.no_grad():
            prediction = self.forward(token_ids, token_type_ids, pad_mask)

        loss = self.criterion(prediction, target).detach().cpu()

        prediction_prob = torch.exp(torch.log_softmax(prediction.detach().cpu(), 1))[:, 1]

        return {'val_loss': loss, 'prediction': prediction_prob, 'targets': target.detach().cpu()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([batch['val_loss'] for batch in outputs]).mean()

        prediction_probs = torch.cat([batch['prediction'] for batch in outputs])
        targets = torch.cat([batch['targets'] for batch in outputs])
        binary_predictions = (prediction_probs > 0.5).float()

        accuracy = (binary_predictions == targets).sum().float() / targets.shape[0]

        log = {'val_loss': avg_loss, 'val_accuracy': accuracy}

        return {'val_loss': avg_loss, 'log': log}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.bert.parameters(),
                                      lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)

        return optimizer

    @staticmethod
    def collate(batch):
        return batch[0]

    def get_loader(self, data_path, shuffle=True):
        dataset = PairedData(data_path=data_path,
                             tokenizer=self.tokenizer,
                             batch_size=self.hparams.batch_size * len(self.hparams.gpu))

        loader = DataLoader(dataset=dataset,
                            collate_fn=self.collate,
                            shuffle=shuffle)

        return loader

    def train_dataloader(self):
        return self.get_loader(os.path.join(self.hparams.data_dir, 'train.tsv'), shuffle=True)

    def val_dataloader(self):
        return self.get_loader(os.path.join(self.hparams.data_dir, 'validation.tsv'), shuffle=False)

    def test_dataloader(self):
        return self.get_loader(os.path.join(self.hparams.data_dir, 'test.tsv'), shuffle=False)
