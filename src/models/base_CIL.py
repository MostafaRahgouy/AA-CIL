import torch
from copy import deepcopy
from abc import ABC, abstractmethod
from transformers import BertModel, BertTokenizer, AdamW


class BaseCIL(torch.nn.Module, ABC):
    def __init__(self, device, init_model):
        super(BaseCIL, self).__init__()

        self.device = device
        self.author_cls = []
        self.author_offset = []

        # define the pre-trained model without any head at top of it (e.g. Bert base)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.feature_extractor = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(
            self.device)

        self.fe_out_size = self.feature_extractor.config.hidden_size  # Output size of the base model
        self.heads = torch.nn.ModuleList()  # classifiers for the sessions

        if init_model:
            self.load_model(filename=init_model)

    def add_head(self, num_new_authors):
        """
            Create a new classifier for each new session based on the number of new authors in the session
            and preserve the previous classifiers separately. These separated classifiers allow us to either update
            all of them or a sub set of them
        """
        self.heads.append(torch.nn.Linear(self.fe_out_size, num_new_authors).to(self.device))
        self.author_cls = torch.tensor([head.out_features for head in self.heads])
        self.author_offset = torch.cat([torch.LongTensor(1).zero_(), self.author_cls.cumsum(0)[:-1]])

    def forward(self, input_ids, attention_mask, return_features=False):
        """
            Applied the features generated from base model to all classifiers/heads. Note that picking
            the desire head is not the responsibility of the forward method
        """
        outputs = []
        features = self.feature_extractor(input_ids, attention_mask=attention_mask)
        cls_hs = features.last_hidden_state[:, 0, :]  # take the BERT [cls] token representation for classification
        for head in self.heads:
            outputs.append(head(cls_hs))

        if return_features:
            return outputs, cls_hs

        return outputs

    def get_copy(self):
        return [deepcopy(self.feature_extractor), deepcopy(self.heads)]

    def get_tokenizer(self):
        return self.tokenizer

    @abstractmethod
    def _get_optimizer(self):
        pass

    @abstractmethod
    def criterion(self, session, predictions, targets):
        pass

    @abstractmethod
    def train_session(self, session, train_loader, val_loader):
        pass

    @abstractmethod
    def train_epoch(self, session, train_loader, epoch):
        pass

    @abstractmethod
    def eval(self, session, val_loader, epoch):
        pass

    @abstractmethod
    def test(self, session, test_loader):
        pass

    def compute_dis_to_author_mean(self, author_data_loader, author_mean):
        with torch.no_grad():
            self.feature_extractor.eval()
            distances = []

            for batch_idx, batch in enumerate(author_data_loader):
                input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                features = self.feature_extractor(input_ids, attention_mask=attention_mask)
                cls_hs = features.last_hidden_state[:, 0, :]

                # Calculate the distances
                batch_distances = torch.norm(cls_hs - author_mean, dim=1)
                distances.extend(batch_distances)

            return distances

    def compute_author_mean_rep(self, author_data_loader):
        with torch.no_grad():
            self.feature_extractor.eval()
            total_reps = None
            total_count = 0

            for batch_idx, batch in enumerate(author_data_loader):
                input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                features = self.feature_extractor(input_ids, attention_mask=attention_mask)
                cls_hs = features.last_hidden_state[:, 0, :]

                if total_reps is None:
                    total_reps = cls_hs.sum(dim=0)
                else:
                    total_reps += cls_hs.sum(dim=0)

                total_count += cls_hs.size(0)

            mean_rep = total_reps / total_count
        return mean_rep

    def save_model(self, filename):
        """
        Saves the feature extractor, and heads to a file.

        Args:
        - filename (str): The file path where the model will be saved.

        Returns:
        None
        """
        heads_config = [{
            'in_features': head.in_features,
            'out_features': head.out_features
        } for head in self.heads]

        torch.save({
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'heads_state_dict': self.heads.state_dict(),
            'heads_config': heads_config
        }, filename)
        print(f"Model saved to {filename}")

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

        print(f"Model loaded from {filename}")
