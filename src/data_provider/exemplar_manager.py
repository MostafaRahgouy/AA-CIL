import torch
import random
from .base_dataset import BaseDataProvider
from torch.utils.data import DataLoader, Dataset


def get_sample_candidates_per_author(sample_candidate):
    sample_candidates_per_author = {}
    for item in sample_candidate:
        if item['incremental_id'] not in sample_candidates_per_author:
            sample_candidates_per_author[item['incremental_id']] = [item]
        else:
            sample_candidates_per_author[item['incremental_id']] += [item]
    return sample_candidates_per_author


def get_exemplar_means(model, pooling_data):
    exemplar_means = {}
    pooling_data_per_author = get_sample_candidates_per_author(pooling_data)
    for author_inc_id, author_exemplars in pooling_data_per_author.items():
        author_dataset = BaseDataProvider(author_exemplars, model.get_tokenizer())
        author_loader = DataLoader(author_dataset, batch_size=16, shuffle=False)
        author_mean_rep = model.compute_author_mean_rep(author_loader)
        exemplar_means.update({author_inc_id: author_mean_rep})
    return exemplar_means


class RandomExemplarProvider:
    def get_random_exemplars_set(self, current_data_session, num_exp_per_author):
        train_sample_candidates = current_data_session['train']
        val_sample_candidates = current_data_session['val']

        train_candidate_samples_per_author = get_sample_candidates_per_author(train_sample_candidates)
        val_candidate_samples_per_author = get_sample_candidates_per_author(val_sample_candidates)

        selected_train_exemplars = self.select_exemplars(train_candidate_samples_per_author, num_exp_per_author)
        selected_val_exemplars = self.select_exemplars(val_candidate_samples_per_author, num_exp_per_author)

        return selected_train_exemplars, selected_val_exemplars

    @staticmethod
    def select_exemplars(sample_candidates_per_author, num_exp_per_author):
        selected_exemplars = []
        for author_inc_id, author_exemplars in sample_candidates_per_author.items():
            selected_exemplars += random.sample(author_exemplars, k=num_exp_per_author)
        return selected_exemplars


class HerdingExemplarProvider:
    def get_herding_exemplars(self, model, current_data_session, num_exp_per_author):
        train_sample_candidates = current_data_session['train']
        val_sample_candidates = current_data_session['val']

        train_candidate_samples_per_author = get_sample_candidates_per_author(train_sample_candidates)
        val_candidate_samples_per_author = get_sample_candidates_per_author(val_sample_candidates)

        selected_train_exemplars = self.get_herding_exemplars_per_author(model,
                                                                         train_candidate_samples_per_author,
                                                                         num_exp_per_author)

        selected_val_exemplars = self.get_herding_exemplars_per_author(model,
                                                                       val_candidate_samples_per_author,
                                                                       num_exp_per_author)

        return selected_train_exemplars, selected_val_exemplars

    @staticmethod
    def get_herding_exemplars_per_author(model, pooling_per_author, num_exp_per_author):
        selected_exemplars = []
        for author_inc_id, author_exemplars in pooling_per_author.items():
            author_dataset = BaseDataProvider(author_exemplars, model.get_tokenizer())
            author_loader = DataLoader(author_dataset, batch_size=16, shuffle=False)

            author_mean_rep = model.compute_author_mean_rep(author_loader)
            author_exemplars_dist = model.compute_dis_to_author_mean(author_loader, author_mean_rep)

            # Get the indices of the top k closest representations
            top_k_indices = torch.topk(torch.stack(author_exemplars_dist), num_exp_per_author, largest=False).indices

            # Select the top k closest representations
            selected_exemplars += [author_exemplars[idx] for idx in top_k_indices]

        return selected_exemplars


class HardExemplarProvider:
    def get_hard_exemplars(self, model, seen_data_sessions, num_exp_per_author):

        train_sample_candidates = seen_data_sessions['train']
        val_sample_candidates = seen_data_sessions['val']

        train_candidate_samples_per_author = get_sample_candidates_per_author(train_sample_candidates)
        val_candidate_samples_per_author = get_sample_candidates_per_author(val_sample_candidates)

        selected_train_exp = self.compute_hard_exemplars(model, train_candidate_samples_per_author, num_exp_per_author)
        selected_val_exp = self.compute_hard_exemplars(model, val_candidate_samples_per_author, num_exp_per_author)

        return selected_train_exp, selected_val_exp

    @staticmethod
    def compute_mean_for_all_authors(model, pooling_per_author):
        all_author_mean = {}

        for author_inc_id, author_exemplars in pooling_per_author.items():
            author_dataset = BaseDataProvider(author_exemplars, model.get_tokenizer())
            author_loader = DataLoader(author_dataset, batch_size=16, shuffle=False)

            author_mean_rep = model.compute_author_mean_rep(author_loader)

            all_author_mean.update({author_inc_id: author_mean_rep})

        return all_author_mean

    def compute_hard_exemplars(self, model, pooling_per_author, num_exp_per_author):
        all_authors_mean = self.compute_mean_for_all_authors(model, pooling_per_author)

        selected_exemplars = []
        for author_inc_id, author_exemplars in pooling_per_author.items():
            author_dataset = BaseDataProvider(author_exemplars, model.get_tokenizer())
            author_loader = DataLoader(author_dataset, batch_size=16, shuffle=False)

            internal_dist = model.compute_dis_to_author_mean(author_loader, all_authors_mean[author_inc_id])

            all_other_mean_dist = []
            for mean_inc_id, author_mean in all_authors_mean.items():
                if mean_inc_id == author_inc_id:
                    continue

                other_mean_dist = model.compute_dis_to_author_mean(author_loader, author_mean)
                all_other_mean_dist.append(torch.stack(other_mean_dist))

            min_author_dist_to_other_centers = torch.min(torch.stack(all_other_mean_dist), dim=0).values
            author_hard_examples = torch.stack(internal_dist) / min_author_dist_to_other_centers
            top_k_indices = torch.topk(author_hard_examples, num_exp_per_author, largest=True).indices
            selected_exemplars += [author_exemplars[idx] for idx in top_k_indices]

        return selected_exemplars
