import torch
from tqdm import tqdm
from copy import deepcopy
from typing import Literal
from torch.utils.tensorboard import SummaryWriter
from .base_CIL import BaseCIL


class BaselineModel(BaseCIL):
    def __init__(self, baseline_type: Literal['FT', 'FT+', 'FZ', 'FZ+', 'FT_E', 'FZ_E'], n_epochs, device, lr, log_dir,
                 init_model=None):
        super(BaselineModel, self).__init__(device=device, init_model=init_model)
        self.baseline_type = baseline_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.optimizer = None
        self.best_model = None
        self.writer = SummaryWriter(log_dir)

    def _get_optimizer(self):
        if self.baseline_type in ['FT', 'FT_E']:
            params = list(self.feature_extractor.parameters()) + [p for head in self.heads for p in head.parameters()]

        elif self.baseline_type == 'FT+':
            if len(self.heads) > 1:
                params = list(self.feature_extractor.parameters()) + list(self.heads[-1].parameters())
            else:
                params = list(self.feature_extractor.parameters()) + [p for head in self.heads for p in
                                                                      head.parameters()]

        elif self.baseline_type in ['FZ', 'FZ_E']:
            if len(self.heads) <= 1:
                params = list(self.feature_extractor.parameters()) + [p for head in self.heads for p in
                                                                      head.parameters()]
            else:
                params = [p for head in self.heads for p in head.parameters()]

        elif self.baseline_type == 'FZ+':
            if len(self.heads) <= 1:
                params = list(self.feature_extractor.parameters()) + [p for head in self.heads for p in
                                                                      head.parameters()]
            else:
                params = self.heads[-1].parameters()
        else:
            raise ValueError("Invalid baseline_type value: {}".format(self.baseline_type))

        return torch.optim.AdamW(params, lr=self.lr)

    def criterion(self, session, predictions, targets):
        if self.baseline_type in ['FT', 'FZ', 'FT_E', 'FZ_E']:
            return torch.nn.functional.cross_entropy(torch.cat(predictions, dim=1), targets)
        return torch.nn.functional.cross_entropy(predictions[session], targets - self.author_offset[session])

    def train_session(self, session, train_loader, val_loader):
        self.best_model = None
        best_val_loss = 100000
        self.optimizer = self._get_optimizer()

        log_losses = {}

        # Loop epochs
        for e in tqdm(range(self.n_epochs)):
            train_loss = self.train_epoch(session, train_loader, e)
            val_loss = self.eval(session, val_loader, e)

            if val_loss < best_val_loss:
                self.best_model = self.get_copy()
                best_val_loss = val_loss

            log_losses[f'epoch_{e + 1}'] = {'train_loss': round(train_loss, 3), 'val_loss': round(val_loss, 3)}

        self.feature_extractor = deepcopy(self.best_model[0])  # take the best model for the next session
        self.heads = deepcopy(self.best_model[1])

        return self.best_model, log_losses

    def train_epoch(self, session, train_loader, epoch):
        total_loss = 0

        self.feature_extractor.train()
        [head.train() for head in self.heads]

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target']
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            targets = targets.to(self.device)

            predictions = self.forward(input_ids, attention_mask)
            loss = self.criterion(session, predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            self.writer.add_scalar(f'Train_loss/Session_{session}', loss.item(), epoch * len(train_loader) + batch_idx)

        total_loss /= len(train_loader)
        return total_loss

    def eval(self, session, val_loader, epoch):
        total_loss = 0

        with torch.no_grad():
            self.feature_extractor.eval()
            [head.eval() for head in self.heads]

            for batch_idx, batch in enumerate(val_loader):
                input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target']
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                targets = targets.to(self.device)

                predictions = self.forward(input_ids, attention_mask)
                loss = self.criterion(session, predictions, targets)
                total_loss += loss.item()

            self.writer.add_scalar(f'Val_loss/Session_{session}', loss.item(), epoch * len(val_loader) + batch_idx)

        total_loss /= len(val_loader)
        return total_loss

    def test(self, session, test_loader):
        all_inc_labels, all_inc_preds, all_true_author_ides = [], [], []

        with torch.no_grad():
            self.feature_extractor.eval()
            [head.eval() for head in self.heads]

            for batch in test_loader:
                input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target']
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                targets = targets.to(self.device)

                logit = self.forward(input_ids, attention_mask)

                pred = torch.argmax(torch.cat(logit, dim=1), dim=1).squeeze()

                pred = [pred.item()] if pred.dim() == 0 else pred.detach().tolist()
                targets = [targets.item()] if targets.dim() == 0 else targets.detach().tolist()
                author_ids = [batch['author_id'].item()] if batch['author_id'].dim() == 0 else batch['author_id'].detach().tolist()

                all_inc_preds += pred
                all_inc_labels += targets
                all_true_author_ides +=  author_ids

        return all_inc_preds, all_inc_labels, all_true_author_ides
