import torch
from tqdm import tqdm
from copy import deepcopy
from typing import Literal
from torch.utils.tensorboard import SummaryWriter
from .base_CIL import BaseCIL


class LearningWithoutForgetting(BaseCIL):
    def __init__(self, model_type: Literal['LWF', 'LWF_E'], n_epochs, device, lr, log_dir, init_model=None):
        super(LearningWithoutForgetting, self).__init__(device=device, init_model=init_model)
        self.model_type = model_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.optimizer = None
        self.best_model = None
        self.temperature = 1 / 2
        self.epsilon = 1e-5
        self.writer = SummaryWriter(log_dir)

    def _get_optimizer(self):
        if self.model_type == 'LWF':
            if len(self.heads) > 1:
                params = list(self.feature_extractor.parameters()) + list(self.heads[-1].parameters())
            else:
                params = list(self.feature_extractor.parameters()) + [p for head in self.heads for p in
                                                                      head.parameters()]

        elif self.model_type == 'LWF_E':
            params = self.parameters()

        else:
            raise ValueError("Invalid model_type value: {}".format(self.model_type))

        return torch.optim.AdamW(params, lr=self.lr)

    def criterion(self, session, student_logit, targets, teacher_logit=None):
        dis_loss = 0
        if session > 0:
            t_logit = torch.cat(teacher_logit[:session], dim=1)
            s_logit = torch.cat(student_logit[:session], dim=1)
            dis_loss = self.distillation_loss(t_logit, s_logit)
        if self.model_type == 'LWF_E':
            return dis_loss + torch.nn.functional.cross_entropy(torch.cat(student_logit, dim=1), targets)
        return dis_loss + torch.nn.functional.cross_entropy(student_logit[session],
                                                            targets - self.author_offset[session])

    def distillation_loss(self, teacher_logit, student_logit):
        """
        Calculates the distillation loss (lwf loss)

        Args:
            teacher_logit (torch.Tensor): Logit from the teacher network.
            student_logit (torch.Tensor): Logit from the student network.
        Returns:
            torch.Tensor: Mean cross-entropy loss value.
        """

        # Compute softmax probabilities for teacher and student logit
        teacher_probs = torch.nn.functional.softmax(teacher_logit, dim=1)
        student_probs = torch.nn.functional.softmax(student_logit, dim=1)

        teacher_probs = teacher_probs.pow(1 / self.temperature)
        student_probs = student_probs.pow(1 / self.temperature)

        # Re-normalize to ensure they sum to 1
        teacher_probs = teacher_probs / teacher_probs.sum(1, keepdim=True)
        student_probs = student_probs / student_probs.sum(1, keepdim=True)

        # Add epsilon for numerical stability and re-normalize
        student_probs = student_probs + self.epsilon / student_probs.size(1)
        student_probs = student_probs / student_probs.sum(1, keepdim=True)

        # Compute the loss
        cross_entropy_loss = -(teacher_probs * student_probs.log()).sum(1)

        # Compute the mean loss over the batch
        mean_loss = cross_entropy_loss.mean()

        return mean_loss

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
        self.teacher_feature_extractor = deepcopy(self.feature_extractor)
        self.teacher_heads = deepcopy(self.heads)
        self.freeze_teacher_model()

        return self.best_model, log_losses

    def train_epoch(self, session, train_loader, epoch):
        total_loss = 0

        self.feature_extractor.train()
        [head.train() for head in self.heads]

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            teacher_logit = None

            input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target']
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            targets = targets.to(self.device)

            if session >= 1:
                teacher_logit = self.teacher_forward(input_ids, attention_mask)

            student_logit = self.forward(input_ids, attention_mask)

            loss = self.criterion(session, student_logit, targets, teacher_logit=teacher_logit)

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
                teacher_logit = None

                input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target']
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                targets = targets.to(self.device)

                if session >= 1:
                    teacher_logit = self.teacher_forward(input_ids, attention_mask)

                student_logit = self.forward(input_ids, attention_mask)

                loss = self.criterion(session, student_logit, targets, teacher_logit=teacher_logit)

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
                author_ids = [batch['author_id'].item()] if batch['author_id'].dim() == 0 else batch[
                    'author_id'].detach().tolist()

                all_inc_preds += pred
                all_inc_labels += targets
                all_true_author_ides += author_ids

        return all_inc_preds, all_inc_labels, all_true_author_ides

    def teacher_forward(self, input_ids, attention_mask, return_features=False):
        """
            Applied the features generated from base model to all classifiers/heads. Note that picking
            the desire head is not the responsibility of the forward method
        """
        outputs = []
        features = self.teacher_feature_extractor(input_ids, attention_mask=attention_mask)
        cls_hs = features.last_hidden_state[:, 0, :]  # take the BERT [cls] token representation for classification
        for head in self.teacher_heads:
            outputs.append(head(cls_hs))

        if return_features:
            return outputs, cls_hs

        return outputs

    def freeze_teacher_model(self):
        for param in self.teacher_feature_extractor.parameters():
            param.requires_grad = False
        for head in self.heads:
            for param in head.parameters():
                param.requires_grad = False

    def load_model(self, filename):
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

        self.teacher_feature_extractor = deepcopy(self.feature_extractor)  # copy the model to teacher
        self.teacher_feature_extractor.to(self.device)
        self.teacher_heads = deepcopy(self.heads)
        self.teacher_heads.to(self.device)
        self.freeze_teacher_model()

        print(f"Model loaded from {filename}")
