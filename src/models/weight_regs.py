import torch
from tqdm import tqdm
from copy import deepcopy
from typing import Literal
from torch.utils.tensorboard import SummaryWriter
from .base_CIL import BaseCIL


class WeightRegularize(BaseCIL):
    def __init__(self, model_type: Literal['EWC', 'MAS'], n_epochs, device, lr, log_dir, init_model=None):
        super(WeightRegularize, self).__init__(device=device, init_model=init_model)
        self.n_epochs = n_epochs
        self.lr = lr
        self.optimizer = None
        self.best_model = None
        self.model_type = model_type
        self.lamb = 5000
        self.alpha = 0.5
        self.writer = SummaryWriter(log_dir)

        self.teacher_params = {n: p.clone().detach() for n, p in self.feature_extractor.named_parameters() if
                               p.requires_grad}

        self.importance = {n: torch.zeros(p.shape).to(self.device) for n, p in self.feature_extractor.named_parameters()
                           if p.requires_grad}

    def _get_optimizer(self):
        if len(self.heads) > 1:
            params = list(self.feature_extractor.parameters()) + list(self.heads[-1].parameters())
        else:
            params = list(self.feature_extractor.parameters()) + [p for head in self.heads for p in head.parameters()]
        return torch.optim.AdamW(params, lr=self.lr)

    def reg_loss(self):
        reg_loss = 0
        for n, p in self.feature_extractor.named_parameters():
            if n in self.importance.keys():
                reg_loss += torch.sum(self.importance[n] * (p - self.teacher_params[n]).pow(2)) / 2
        return self.lamb * reg_loss

    def criterion(self, session, predictions, targets):
        reg_loss = 0
        if session > 0:
            reg_loss = self.reg_loss()
        return reg_loss + torch.nn.functional.cross_entropy(predictions[session], targets - self.author_offset[session])

    def compute_new_importance(self, train_loader):
        new_importance = {n: torch.zeros(p.shape).to(self.device) for n, p in self.feature_extractor.named_parameters()
                          if p.requires_grad}

        self.feature_extractor.train()
        [head.train() for head in self.heads]

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target']
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            targets = targets.to(self.device)

            if self.model_type == 'EWC':
                predictions = self.forward(input_ids, attention_mask)
                loss = torch.nn.functional.cross_entropy(torch.cat(predictions, dim=1), targets)
                self.optimizer.zero_grad()
                loss.backward()
                for n, p in self.feature_extractor.named_parameters():
                    if p.grad is not None:
                        new_importance[n] += p.grad.pow(2) * len(targets)

            elif self.model_type == 'MAS':
                predictions = self.forward(input_ids, attention_mask)
                loss = torch.norm(torch.cat(predictions, dim=1), p=2, dim=1).mean()
                self.optimizer.zero_grad()
                loss.backward()
                for n, p in self.feature_extractor.named_parameters():
                    if p.grad is not None:
                        new_importance[n] += p.grad.abs() * len(targets)

            else:
                raise ValueError('model type is not valid')

        return {n: (p / (len(train_loader) * train_loader.batch_size)) for n, p in new_importance.items()}

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

        self.teacher_params = {n: p.clone().detach() for n, p in self.feature_extractor.named_parameters() if
                               p.requires_grad}

        new_importance = self.compute_new_importance(train_loader)

        for n in self.importance.keys():
            self.importance[n] = (self.alpha * self.importance[n] + (1 - self.alpha) * new_importance[n])

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

    def custom_load_model(self, filename, train_loader):
        """
        Loads the feature extractor, and heads from a file.

        Args:
        - filename (str): The file path from which the model will be loaded.

        """
        checkpoint = torch.load(filename, map_location=self.device)

        # Reconstruct heads based on saved configuration
        heads_config = checkpoint['heads_config']
        self.heads = torch.nn.ModuleList()
        for head_config in heads_config:
            head = torch.nn.Linear(head_config['in_features'], head_config['out_features'])
            self.heads.append(head)

        # Load heads state dictionary
        self.heads.load_state_dict(checkpoint['heads_state_dict'])
        self.heads.to(self.device)

        # Load the feature extractor state dictionary
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'], strict=False)
        self.feature_extractor.to(self.device)

        self.teacher_params = {n: p.clone().detach() for n, p in self.feature_extractor.named_parameters() if
                               p.requires_grad}

        self.optimizer = self._get_optimizer()
        new_importance = self.compute_new_importance(train_loader)

        for n in self.importance.keys():
            self.importance[n] = (self.alpha * self.importance[n] + (1 - self.alpha) * new_importance[n])

        print(f"Model loaded from {filename}")