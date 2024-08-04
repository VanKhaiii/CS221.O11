#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import json
from math import log
from scipy.optimize import fmin_l_bfgs_b
from itertools import chain
from collections import defaultdict, Counter


BOS_TOKEN, BOS_IDX = '<bos>', 0


def read_corpus(filename):
    with open(filename) as file:
        lines = [line.strip().split() for line in file]

    data, X, Y = [], [], []
    for words in lines:
        if not words:
            if X: data.append((X, Y))
            X, Y = [], []
        else:
            X.append(words[:-1])
            Y.append(words[-1])

    if X: data.append((X, Y))
    return data


def extract_text_features_at_position(text_tokens, position):
    """
    Extracts text features at a specified position in a list of text tokens.

    Args:
    text_tokens (list): A list of text tokens, where each token is represented as a list containing
                        the token itself and its part-of-speech tag.
    position (int): The position in the list to extract features for.

    Returns:
    list: A list of extracted features, including unigrams and bigrams based on the specified position.

    Examples:

    For text_tokens:

    [['Confidence', 'NN'], ['in', 'IN'], ['the', 'DT'], ['pound', 'NN'], ['is', 'VBZ'], ['widely', 'RB'],
     ['expected', 'VBN'], ['to', 'TO'], ['take', 'VB'], ['another', 'DT'], ['sharp', 'JJ'], ['dive', 'NN'],
     ['if', 'IN'], ['trade', 'NN'], ['figures', 'NNS'], ['for', 'IN'], ['September', 'NNP'], [',', ','],
     ['due', 'JJ'], ['for', 'IN'], ['release', 'NN'], ['tomorrow', 'NN'], [',', ','], ['fail', 'VB'],
     ['to', 'TO'], ['show', 'VB'], ['a', 'DT'], ['substantial', 'JJ'], ['improvement', 'NN'], ['from', 'IN'],
     ['July', 'NNP'], ['and', 'CC'], ['August', 'NNP'], ["'s", 'POS'], ['near-record', 'JJ'], ['deficits', 'NNS'],
     ['.', '.']]

    """

    # Sau khi thêm việc sử dụng các đặc trưng khác (Tức là thêm các nhãn từ loại sử dụng như là một đặc trưng
    # về mặt nghĩa nghĩa). Thì mô hình đã có sự tăng trưởng về hiệu suất (đạt khoảng 89%).
    # Ví dụ ở vị trí position = 0  thì:
    # features = ['unigram[0]:Confidence','pos_unigram[0]:NN', 'unigram[+1]:in', 'pos_unigram[+1]:IN', 'bigram[0]:Confidence in',
    #             'pos_bigram[0]:NN IN',  'unigram[+2]:the', 'pos_unigram[+2]:DT', 'pos_bigram[+1]:IN DT']

    ### YOUR CODE HERE
    length = len(text_tokens)

    features = list()
    features.append('unigram[0]:' + text_tokens[position][0])
    features.append('pos_unigram[0]:' + text_tokens[position][1])

    if position < length-1:
        features.append('unigram[+1]:' + (text_tokens[position+1][0]))
        features.append('pos_unigram[+1]:' + text_tokens[position+1][1])
        features.append('bigram[0]:{} {}'.format(text_tokens[position][0], text_tokens[position+1][0]))
        features.append('pos_bigram[0]:{} {}'.format(text_tokens[position][1], text_tokens[position+1][1]))

        if position < length-2:
            features.append('unigram[+2]:' + (text_tokens[position+2][0]))
            features.append('pos_unigram[+2]:' + (text_tokens[position+2][1]))
            features.append('pos_bigram[+1]:{} {}'.format(text_tokens[position+1][1], text_tokens[position+2][1]))

    if position > 0:
        features.append('unigram[-1]:' + (text_tokens[position-1][0]))
        features.append('pos_unigram[-1]:' + (text_tokens[position-1][1]))
        features.append('bigram[-1]:{} {}'.format(text_tokens[position-1][0], text_tokens[position][0]))
        features.append('pos_bigram[-1]:{} {}'.format(text_tokens[position-1][1], text_tokens[position][1]))

        if position > 1:
            features.append('unigram[-2]:' + (text_tokens[position-2][0]))
            features.append('pos_unigram[-2]:' + (text_tokens[position-2][1]))
            features.append('pos_bigram[-2]:{} {}'.format(text_tokens[position-2][1], text_tokens[position-1][1]))

    return features


class FeatureSet():
    feature_dict, observation_set, empirical_counts = dict(), set(), Counter()
    num_features, label_dict, label_array = 0, {BOS_TOKEN: BOS_IDX}, [BOS_TOKEN]

    def __init__(self):
        pass

    def process_corpus(self, data):
        for X, Y in data:
            prev_y = BOS_IDX
            for t, char in enumerate(X):
                y = self.label_dict.get(Y[t], len(self.label_dict))
                if Y[t] not in self.label_dict:
                    self.label_dict[Y[t]] = y
                    self.label_array.append(Y[t])
                self._add(prev_y, y, X, t)
                prev_y = y

    def load(self, feature_dict, num_features, label_array):
        self.num_features = num_features
        self.label_array = label_array
        self.label_dict = {label: i for label, i in enumerate(label_array)}
        self.feature_dict = self.deserialize_feature_dict(feature_dict)

    def __len__(self):
        return self.num_features

    def _add(self, prev_y, y, X, t):
        for feature_string in extract_text_features_at_position(X, t):
            if feature_string not in self.feature_dict:
                self.feature_dict[feature_string] = {}

            for pair in [(prev_y, y), (-1, y)]:
                if pair not in self.feature_dict[feature_string]:
                    self.feature_dict[feature_string][pair] = self.num_features
                    self.num_features += 1

                feature_id = self.feature_dict[feature_string][pair]
                self.empirical_counts[feature_id] += 1

    def get_feature_vector(self, prev_y, y, X, t):
        return [self.feature_dict[feature_string][(prev_y, y)] for feature_string in extract_text_features_at_position(X, t) if (prev_y, y) in self.feature_dict[feature_string]]

    def get_labels(self):
        return self.label_dict, self.label_array

    def calc_inner_products(self, params, X, t):
        inner_products = defaultdict(float)
        features = chain.from_iterable(self.feature_dict.get(feature_string, {}).items() for feature_string in extract_text_features_at_position(X, t))
        for (prev_y, y), feature_id in features:
            inner_products[(prev_y, y)] += params[feature_id]
        return [((prev_y, y), score) for (prev_y, y), score in inner_products.items()]

    def get_empirical_counts(self):
        return np.array([self.empirical_counts.get(feature_id, 0) for feature_id in range(self.num_features)])

    def get_feature_list(self, X, t):
        feature_list_dict = defaultdict(set)
        [feature_list_dict[(prev_y, y)].add(feature_id) for feature_string in extract_text_features_at_position(X, t) for (prev_y, y), feature_id in self.feature_dict[feature_string].items()]
        return list(feature_list_dict.items())

    def serialize_feature_dict(self):
        return {feature_string: {'%d_%d' % (prev_y, y): feature_id for (prev_y, y), feature_id in features.items()}
                for feature_string, features in self.feature_dict.items()}

    def deserialize_feature_dict(self, serialized):
        return {feature_string: {(int(prev_y), int(y)): feature_id
                for transition_string, feature_id in features.items()
                for prev_y, y in [transition_string.split('_')]}
                for feature_string, features in serialized.items()}


SCALE_THRES = 1e250
ITER_NUM = 0
SUB_ITER_NUM = 0
TOTAL_SUB_ITERS = 0
GRAD = None


def _gen_trans_prob_tables(params, num_labels, feature_set, X, inference=True):
    tables = []
    for t, x in enumerate(X):
        table = np.zeros((num_labels, num_labels))
        pairs = feature_set.calc_inner_products(params, X, t) if inference else x
        for pair, score in pairs:
            prev_y, y = pair
            score = sum(params[fid] for fid in score) if not inference else score
            table[prev_y if prev_y != -1 else slice(None), y] += score

        table = np.exp(table)
        if t == 0:
            table[BOS_IDX+1:] = 0
        else:
            table[:, BOS_IDX] = 0
            table[BOS_IDX, :] = 0
        tables.append(table)

    return tables


def _forward_backward(num_labels, time_length, trans_prob_tables):
    alpha, beta = np.zeros((time_length, num_labels)), np.zeros((time_length, num_labels))
    scaling_dict = {}

    alpha[0, :] = trans_prob_tables[0][BOS_IDX, :]
    for t in range(1, time_length):
        alpha[t] = np.dot(alpha[t-1], trans_prob_tables[t])
        if alpha[t].max() > SCALE_THRES:
            scaling_dict[t-1] = SCALE_THRES
            alpha[t-1] /= SCALE_THRES
            alpha[t] = 0
            break

    beta[-1] = 1.0
    for t in range(time_length - 2, -1, -1):
        beta[t] = np.dot(beta[t+1], trans_prob_tables[t+1].T)
        if t in scaling_dict:
            beta[t] /= scaling_dict[t]

    Z = alpha[-1].sum()

    return alpha, beta, Z, scaling_dict


def _log_likelihood(params, *args):
    training_data, feature_set, training_feature_data, empirical_counts, label_dict, squared_sigma = args
    expected_counts = np.zeros(len(feature_set))
    total_logZ = 0

    for X_features in training_feature_data:
        trans_prob_tables = _gen_trans_prob_tables(params, len(label_dict), feature_set, X_features, inference=False)
        alpha, beta, Z, scaling_dict = _forward_backward(len(label_dict), len(X_features), trans_prob_tables)
        total_logZ += log(Z) + sum(log(s) for s in scaling_dict.values())

        for t, X_feature in enumerate(X_features):
            for (prev_y, y), feature_ids in X_feature:
                if prev_y == -1:
                    prob = (alpha[t, y] * beta[t, y] * scaling_dict.get(t, 1)) / Z
                elif t == 0 and prev_y == BOS_IDX:
                    prob = (trans_prob_tables[t][BOS_IDX, y] * beta[t, y]) / Z
                elif prev_y != BOS_IDX and y != BOS_IDX:
                    prob = (alpha[t-1, prev_y] * trans_prob_tables[t][prev_y, y] * beta[t, y]) / Z
                else:
                    continue
                for fid in feature_ids:
                    expected_counts[fid] += prob

    likelihood = np.dot(empirical_counts, params) - total_logZ - np.sum(params**2) / (2 * squared_sigma)
    gradients = empirical_counts - expected_counts - params / squared_sigma
    global GRAD
    GRAD = gradients

    global SUB_ITER_NUM
    print(f"  {ITER_NUM:03d} {f'({SUB_ITER_NUM:02d})' if SUB_ITER_NUM > 0 else '    '}: {-likelihood}")
    SUB_ITER_NUM += 1

    return -likelihood


def _gradient(params, *args):
    return GRAD * -1


class LinearChainCRF():
    training_data = feature_set = label_dict = label_array = num_labels = params = None
    squared_sigma = 10.0

    def __init__(self):
        pass

    def _read_corpus(self, filename):
        return read_corpus(filename)

    def _get_training_feature_data(self):
        return [[self.feature_set.get_feature_list(X, t) for t in range(len(X))]
                for X, _ in self.training_data]

    def _estimate_parameters(self):
        def _callback(params):
            global ITER_NUM, SUB_ITER_NUM, TOTAL_SUB_ITERS
            ITER_NUM += 1
            TOTAL_SUB_ITERS += SUB_ITER_NUM
            SUB_ITER_NUM = 0

        # Get training feature data using a method _get_training_feature_data()
        train_feat_data = self._get_training_feature_data()

        # Perform L-BFGS-B optimization using the fmin_l_bfgs_b function
        self.params, log_likelihood, info = fmin_l_bfgs_b(
            func=_log_likelihood,  # Objective function to minimize
            fprime=_gradient,       # Gradient of the objective function
            x0=np.zeros(len(self.feature_set)),  # Initial guess for parameters
            args=(
                self.training_data,  # Training data
                self.feature_set,    # Feature set
                train_feat_data,     # Training feature data
                self.feature_set.get_empirical_counts(),  # Empirical feature counts
                self.label_dict,      # Label dictionary
                self.squared_sigma   # Squared sigma value
            ),
            callback=_callback  # Callback function to be called during optimization
        )

    def train(self, corpus_filename, model_filename):
        self.training_data = self._read_corpus(corpus_filename)
        self.feature_set = FeatureSet()
        self.feature_set.process_corpus(self.training_data)
        self.label_dict, self.label_array = self.feature_set.get_labels()
        self.num_labels = len(self.label_array)
        self._estimate_parameters()
        self.save_model(model_filename)

    def test(self, test_corpus_filename):
        test_data = self._read_corpus(test_corpus_filename)
        total_count, correct_count = 0, 0
        for X, Y in test_data:
            total_count += len(Y)
            correct_count += sum(y == yp for y, yp in zip(Y, self.inference(X)))
        print(f'Accuracy: {correct_count/total_count}')

    def inference(self, X):
        return self.viterbi(X, _gen_trans_prob_tables(self.params, self.num_labels, self.feature_set, X, inference=True))

    def viterbi(self, observations, trans_prob_tables):
        seq_len = len(observations)
        max_prob_table = np.zeros((seq_len, self.num_labels))
        backpointer_table = np.zeros((seq_len, self.num_labels), dtype='int64')

        max_prob_table[0, :] = trans_prob_tables[0][BOS_IDX, :]

        for t in range(1, seq_len):
            for cur_label_id in range(self.num_labels):
                probabilities = [max_prob_table[t-1, prev_label_id] *
                                trans_prob_tables[t][prev_label_id, cur_label_id]
                                for prev_label_id in range(self.num_labels)]

                best_prev_label = np.argmax(probabilities)
                highest_prob = probabilities[best_prev_label]
                max_prob_table[t, cur_label_id] = highest_prob
                backpointer_table[t, cur_label_id] = best_prev_label

        decoded_sequence = []
        last_label = np.argmax(max_prob_table[-1])
        decoded_sequence.append(last_label)

        for t in range(seq_len - 1, 0, -1):
            last_label = backpointer_table[t, last_label]
            decoded_sequence.append(last_label)

        return [self.label_dict[label_id] for label_id in decoded_sequence[::-1]]

    def save_model(self, model_filename):
        with open(model_filename, 'w') as f:
            json.dump({
                "feature_dict": self.feature_set.serialize_feature_dict(),
                "num_features": self.feature_set.num_features,
                "labels": self.feature_set.label_array,
                "params": list(self.params)
            }, f, ensure_ascii=False, indent=2, separators=(',', ':'))

    def load(self, model_filename):
        with open(model_filename) as f:
            model = json.load(f)

        self.feature_set = FeatureSet()
        self.feature_set.load(model['feature_dict'], model['num_features'], model['labels'])
        self.label_dict, self.label_array = self.feature_set.get_labels()
        self.num_labels, self.params = len(self.label_array), np.asarray(model['params'])


crf = LinearChainCRF()
crf.train('data/train.txt', 'data/model.json')
crf.load('data/model.json')
crf.test('data/test.txt')