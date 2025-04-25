#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==========================================
# CELL 1: IMPORTS AND SETUP
# ==========================================
import os
import argparse
import time
import pickle
import numpy as np
import tensorflow as tf
from collections import Counter
from functools import reduce
from math import ceil

# Disable eager execution for TensorFlow 2.x
tf.compat.v1.disable_eager_execution()

# Define the argument parser for use in both script mode and notebook mode
parser = argparse.ArgumentParser(description="Experiment setup")
# misc
parser.add_argument('--seed', default=33, type=int)
parser.add_argument('--gpu', default="", type=str)
parser.add_argument('--no_train', default=False, action="store_true")
parser.add_argument('--from_model_ckpt', default=None, type=str)
parser.add_argument('--no_rules', default=False, action="store_true")
parser.add_argument('--rule_thr', default=1e-2, type=float)    
parser.add_argument('--no_preds', default=False, action="store_true")
parser.add_argument('--get_vocab_embed', default=False, action="store_true")
parser.add_argument('--exps_dir', default=None, type=str)
parser.add_argument('--exp_name', default=None, type=str)
# data property
parser.add_argument('--datadir', default=None, type=str)
parser.add_argument('--resplit', default=False, action="store_true")
parser.add_argument('--no_link_percent', default=0., type=float)
parser.add_argument('--type_check', default=False, action="store_true")
parser.add_argument('--domain_size', default=128, type=int)
parser.add_argument('--no_extra_facts', default=False, action="store_true")
parser.add_argument('--query_is_language', default=False, action="store_true")
parser.add_argument('--vocab_embed_size', default=128, type=int)
# model architecture
parser.add_argument('--num_step', default=3, type=int)
parser.add_argument('--num_layer', default=1, type=int)
parser.add_argument('--rnn_state_size', default=128, type=int)
parser.add_argument('--query_embed_size', default=128, type=int)
# optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--print_per_batch', default=3, type=int)
parser.add_argument('--max_epoch', default=10, type=int)
parser.add_argument('--min_epoch', default=5, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--no_norm', default=False, action="store_true")
parser.add_argument('--thr', default=1e-20, type=float)
parser.add_argument('--dropout', default=0., type=float)
# evaluation
parser.add_argument('--get_phead', default=False, action="store_true")
parser.add_argument('--adv_rank', default=False, action="store_true")
parser.add_argument('--rand_break', default=False, action="store_true")
parser.add_argument('--accuracy', default=False, action="store_true")
parser.add_argument('--top_k', default=10, type=int)

# Create a compatibility class for SparseTensorValue
class SparseTensorValue:
    def __init__(self, indices, values, dense_shape):
        self.indices = indices
        self.values = values
        self.dense_shape = dense_shape

# ==========================================
# CELL 2: DATA PROCESSING
# ==========================================
def resplit(train, facts, no_link_percent):
    num_train = len(train)
    num_facts = len(facts)
    all = train + facts
    
    if no_link_percent == 0.:
        np.random.shuffle(all)
        new_train = all[:num_train]
        new_facts = all[num_train:]
    else:
        link_cntr = Counter()
        for tri in all:
            link_cntr[(tri[1], tri[2])] += 1
        tmp_train = []
        tmp_facts = []
        for tri in all:
            if link_cntr[(tri[1], tri[2])] + link_cntr[(tri[2], tri[1])] > 1:
                if np.random.random() < no_link_percent:
                    tmp_facts.append(tri)
                else:
                    tmp_train.append(tri)
            else:
                tmp_train.append(tri)
        
        if len(tmp_train) > num_train:
            np.random.shuffle(tmp_train)
            new_train = tmp_train[:num_train]
            new_facts = tmp_train[num_train:] + tmp_facts
        else:
            np.random.shuffle(tmp_facts)
            num_to_fill = num_train - len(tmp_train)
            new_train = tmp_train + tmp_facts[:num_to_fill]
            new_facts = tmp_facts[num_to_fill:]
    
    assert(len(new_train) == num_train)
    assert(len(new_facts) == num_facts)

    return new_train, new_facts


class Data(object):
    def __init__(self, folder, seed, type_check=False, domain_size=128, no_extra_facts=False, option=None):
        np.random.seed(seed)
        self.seed = seed
        self.type_check = type_check
        self.domain_size = domain_size
        self.use_extra_facts = not no_extra_facts
        self.query_include_reverse = True
        
        # Store option for dynamic access
        self.option = option

        self.relation_file = os.path.join(folder, "relations.txt")
        self.entity_file = os.path.join(folder, "entities.txt")
        
        self.relation_to_number, self.entity_to_number = self._numerical_encode()
        self.number_to_entity = {v: k for k, v in self.entity_to_number.items()}
        self.num_relation = len(self.relation_to_number)
        self.num_query = self.num_relation * 2
        self.num_entity = len(self.entity_to_number)
                
        self.test_file = os.path.join(folder, "test.txt")
        self.train_file = os.path.join(folder, "train.txt")
        self.valid_file = os.path.join(folder, "valid.txt")
        
        if os.path.isfile(os.path.join(folder, "facts.txt")):
            self.facts_file = os.path.join(folder, "facts.txt")
            self.share_db = True
        else:
            self.train_facts_file = os.path.join(folder, "train_facts.txt")
            self.test_facts_file = os.path.join(folder, "test_facts.txt")
            self.share_db = False

        self.test, self.num_test = self._parse_triplets(self.test_file)
        self.train, self.num_train = self._parse_triplets(self.train_file)        
        if os.path.isfile(self.valid_file):
            self.valid, self.num_valid = self._parse_triplets(self.valid_file)
        else:
            self.valid, self.train = self._split_valid_from_train()
            self.num_valid = len(self.valid)
            self.num_train = len(self.train)

        if self.share_db: 
            self.facts, self.num_fact = self._parse_triplets(self.facts_file)
            self.matrix_db = self._db_to_matrix_db(self.facts)
            self.matrix_db_train = self.matrix_db
            self.matrix_db_test = self.matrix_db
            self.matrix_db_valid = self.matrix_db
            if self.use_extra_facts:
                extra_mdb = self._db_to_matrix_db(self.train)
                self.augmented_mdb = self._combine_two_mdbs(self.matrix_db, extra_mdb)
                self.augmented_mdb_valid = self.augmented_mdb
                self.augmented_mdb_test = self.augmented_mdb
        else:
            self.train_facts, self.num_train_fact \
                = self._parse_triplets(self.train_facts_file)
            self.test_facts, self.num_test_fact \
                = self._parse_triplets(self.test_facts_file)
            self.matrix_db_train = self._db_to_matrix_db(self.train_facts)
            self.matrix_db_test = self._db_to_matrix_db(self.test_facts)
            self.matrix_db_valid = self._db_to_matrix_db(self.train_facts)
        
        if self.type_check:
            self.domains_file = os.path.join(folder, "stats/domains.txt")
            self.domains = self._parse_domains_file(self.domains_file)
            self.train = sorted(self.train, key=lambda x: x[0])
            self.test = sorted(self.test, key=lambda x: x[0])
            self.valid = sorted(self.valid, key=lambda x: x[0])
            self.num_operator = 2 * self.domain_size
        else:
            self.domains = None
            self.num_operator = 2 * self.num_relation

        # get rules for queries and their inverses appeared in train and test
        self.query_for_rules = list(set(list(zip(*self.train))[0]) | set(list(zip(*self.test))[0]) | 
                               set(list(zip(*self._augment_with_reverse(self.train)))[0]) | 
                               set(list(zip(*self._augment_with_reverse(self.test)))[0]))
        self.parser = self._create_parser()

    def _create_parser(self):
        """Create a parser that maps numbers to queries and operators given queries"""
        assert(self.num_query==2*len(self.relation_to_number)==2*self.num_relation)
        parser = {"query":{}, "operator":{}}
        number_to_relation = {value: key for key, value 
                                         in self.relation_to_number.items()}
        for key, value in self.relation_to_number.items():
            parser["query"][value] = key
            parser["query"][value + self.num_relation] = "inv_" + key
        for query in range(self.num_relation):
            d = {}
            if self.type_check:
                for i, o in enumerate(self.domains[query]):
                    d[i] = number_to_relation[o]
                    d[i + self.domain_size] = "inv_" + number_to_relation[o]
            else:
                for k, v in number_to_relation.items():
                    d[k] = v
                    d[k + self.num_relation] = "inv_" + v
            parser["operator"][query] = d
            parser["operator"][query + self.num_relation] = d
        return parser
        
    def _parse_domains_file(self, file_name):
        result = {}
        with open(file_name, "r") as f:
            for line in f:
                l = line.strip().split(",")
                l = [self.relation_to_number[i] for i in l]
                relation = l[0]
                this_domain = l[1:1+self.domain_size]
                if len(this_domain) == self.domain_size:
                    pass
                else:
                    # fill in blanks
                    num_remain = self.domain_size - len(this_domain)
                    remains = [i for i in range(self.num_relation) 
                                 if i not in this_domain]
                    pads = np.random.choice(remains, num_remain, replace=False)
                    this_domain += list(pads)
                this_domain.sort()
                assert(len(set(this_domain)) == self.domain_size)
                assert(len(this_domain) == self.domain_size)
                result[relation] = this_domain
        for r in range(self.num_relation):
            if r not in result.keys():
                result[r] = np.random.choice(range(self.num_relation), 
                                             self.domain_size, 
                                             replace=False)
        return result
    
    def _numerical_encode(self):
        relation_to_number = {}
        with open(self.relation_file) as f:
            for line in f:
                l = line.strip().split()
                assert(len(l) == 1)
                relation_to_number[l[0]] = len(relation_to_number)
        
        entity_to_number = {}
        with open(self.entity_file) as f:
            for line in f:
                l = line.strip().split()
                assert(len(l) == 1)
                entity_to_number[l[0]] = len(entity_to_number)
        return relation_to_number, entity_to_number

    def _parse_triplets(self, file):
        """Convert (head, relation, tail) to (relation, head, tail)"""
        output = []
        with open(file) as f:
            for line in f:
                l = line.strip().split("\t")
                assert(len(l) == 3)
                output.append((self.relation_to_number[l[1]], 
                               self.entity_to_number[l[0]], 
                               self.entity_to_number[l[2]]))
        return output, len(output)

    def _split_valid_from_train(self):
        np.random.shuffle(self.train)
        num_valid = max(1, int(0.1 * self.num_train))
        valid = self.train[:num_valid]
        train = self.train[num_valid:]
        return valid, train

    def _db_to_matrix_db(self, db):
        matrix_db = {}
        for r in range(self.num_relation):
            indices = []
            data = []
            db_r = list(filter(lambda x: x[0]==r, db))
            for _, h, t in db_r:
                indices.append([h, t])
                data.append(1.)
            if len(indices) > 0:
                indices = np.array(indices)
                data = np.array(data)
                matrix_db[r] = SparseTensorValue(indices, data,
                                              dense_shape=[self.num_entity, self.num_entity])
            else:
                matrix_db[r] = SparseTensorValue(
                                 np.empty((0,2), dtype=np.int64), 
                                 np.array([], dtype=np.float32),
                                 dense_shape=[self.num_entity, self.num_entity]) 
        return matrix_db
    
    def _combine_two_mdbs(self, mdbA, mdbB):
        combined = {}
        for r in range(self.num_relation):
            indices_A = mdbA[r].indices
            values_A = mdbA[r].values
            indices_B = mdbB[r].indices
            values_B = mdbB[r].values
            
            indices = np.concatenate([indices_A, indices_B], axis=0)
            values = np.concatenate([values_A, values_B], axis=0)
            if len(indices) > 0:
                combined[r] = SparseTensorValue(indices, values,
                                             dense_shape=[self.num_entity, self.num_entity])
            else:
                combined[r] = SparseTensorValue(
                                 np.empty((0,2), dtype=np.int64), 
                                 np.array([], dtype=np.float32),
                                 dense_shape=[self.num_entity, self.num_entity])
        return combined
    
    def _count_batch(self, samples, batch_size):
        return int(ceil(len(samples) / float(batch_size)))

    def reset(self, batch_size):
        self.batch_size = batch_size
        self.train_batch_index = 0        
        self.test_batch_index = 0
        self.valid_batch_index = 0
        self.num_batch_train = self._count_batch(self.train, batch_size)
        self.num_batch_test = self._count_batch(self.test, batch_size)
        self.num_batch_valid = self._count_batch(self.valid, batch_size)
        np.random.shuffle(self.train)
        np.random.shuffle(self.test)
        np.random.shuffle(self.valid)
        
    def train_resplit(self, no_link_percent):
        if self.share_db:
            self.train, self.facts = resplit(self.train, self.facts, no_link_percent)
            self.matrix_db = self._db_to_matrix_db(self.facts)
            self.matrix_db_train = self.matrix_db
            self.matrix_db_test = self.matrix_db
            self.matrix_db_valid = self.matrix_db

    def _subset_of_matrix_db(self, matrix_db, domain):
        """Pick a subset of domain from matrix_db."""
        subset = {}
        for r in domain:
            subset[r] = matrix_db[r]
        return subset
    
    def _augment_with_reverse(self, triplets):
        """Add reverse triplets with query = query + num_query / 2."""
        new_triplets = []
        for q, h, t in triplets:
            new_triplets.append((q + self.num_relation, t, h))
        return triplets + new_triplets
            
    def _next_batch(self, start, size, samples):
        '''Generate a minibatch.'''
        end = min(start + size, len(samples))
        sample_batch = samples[start:end]
        for i in range(end, start + size):
            j = np.random.randint(len(samples))
            sample_batch.append(samples[j])
        return sample_batch, end % len(samples)
    
    def _triplet_to_feed(self, triplets):
        """Convert triplets (query, head, tail) to (query, head, tail) for model inputs."""
        batch_size = len(triplets)
        
        # Adapt to the num_step from options
        num_step = 3  # Default
        if hasattr(self, 'option') and hasattr(self.option, 'num_step'):
            num_step = self.option.num_step
        
        queries = np.zeros((batch_size, num_step), dtype=np.int32)
        # padded with <END> = num_query
        queries.fill(self.num_query)
        for i, (q, h, t) in enumerate(triplets):
            queries[i, 0] = q
        
        heads = np.array([h for _, h, _ in triplets])
        tails = np.array([t for _, _, t in triplets])
        
        return (queries, heads, tails)

    def next_test(self):
        raw_sample_batch, self.test_batch_index = self._next_batch(
                                             self.test_batch_index, 
                                             self.batch_size, 
                                             self.test)
        if self.share_db:
            matrix_db = self.matrix_db_test
        else:
            sub_domain = range(self.num_relation)
            matrix_db = self._subset_of_matrix_db(self.matrix_db_test, sub_domain)
        
        return self._triplet_to_feed(raw_sample_batch), matrix_db
        
    def next_valid(self):
        raw_sample_batch, self.valid_batch_index = self._next_batch(
                                             self.valid_batch_index, 
                                             self.batch_size, 
                                             self.valid)
        if self.share_db:
            matrix_db = self.matrix_db_valid
        else:
            sub_domain = range(self.num_relation)
            matrix_db = self._subset_of_matrix_db(self.matrix_db_valid, sub_domain)
        
        return self._triplet_to_feed(raw_sample_batch), matrix_db

    def next_train(self):
        raw_sample_batch, self.train_batch_index = self._next_batch(
                                             self.train_batch_index, 
                                             self.batch_size, 
                                             self.train)
        if self.share_db:
            if self.use_extra_facts:
                matrix_db = self.augmented_mdb
            else:
                matrix_db = self.matrix_db_train
        else:
            sub_domain = range(self.num_relation)
            matrix_db = self._subset_of_matrix_db(self.matrix_db_train, sub_domain)
            
        return self._triplet_to_feed(raw_sample_batch), matrix_db


class DataPlus(Data):
    def __init__(self, folder, seed):
        np.random.seed(seed)
        self.seed = seed
        self.type_check = False
        self.use_extra_facts = True
        self.query_include_reverse = True

        self.entity_file = os.path.join(folder, "entities.txt")
        self.relation_file = os.path.join(folder, "relations.txt")
        self.vocab_file = os.path.join(folder, "vocab.txt")
        self.train_file = os.path.join(folder, "train.txt")
        self.valid_file = os.path.join(folder, "valid.txt")
        self.test_file = os.path.join(folder, "test.txt")

        self.entity_to_number = self._numerical_encode(self.entity_file)
        self.relation_to_number = self._numerical_encode(self.relation_file)
        self.vocab_to_number = self._numerical_encode(self.vocab_file)
        
        self.number_to_entity = {v: k for k, v in self.entity_to_number.items()}
        self.number_to_relation = {v: k for k, v in self.relation_to_number.items()}
        
        self.num_entity = len(self.entity_to_number)
        self.num_relation = len(self.relation_to_number)
        self.num_vocab = len(self.vocab_to_number)
        self.num_word = None
        
        # we are working with language, each query is treated as a list of words. 
        # convert query to a uniform length
        self.train, self.num_train = self._parse_examples(self.train_file)
        self.test, self.num_test = self._parse_examples(self.test_file)
        if os.path.isfile(self.valid_file):
            self.valid, self.num_valid = self._parse_examples(self.valid_file)
        else:
            self.valid, self.train = self._split_valid_from_train()
            self.num_valid = len(self.valid)
            self.num_train = len(self.train)
            
        # get facts from train
        self.facts = []
        for q, h, t in self.train:
            self.facts.append((h, q, t))
            
        self.matrix_db = self._db_to_matrix_db(self.facts)
        self.matrix_db_train = self.matrix_db
        self.matrix_db_test = self.matrix_db
        self.matrix_db_valid = self.matrix_db

        if self.use_extra_facts:
            self.augmented_mdb = self.matrix_db
            self.augmented_mdb_test = self.matrix_db
            self.augmented_mdb_valid = self.matrix_db
            
        self.num_operator = 2 * self.num_relation
        self.parser = self._create_parser()

    def _numerical_encode(self, file_name):
        mapping = {}
        with open(file_name) as f:
            for line in f:
                l = line.strip().split("\t")
                assert(len(l) == 1)
                mapping[l[0]] = len(mapping)
        return mapping
        
    def _parse_examples(self, file_name):
        examples = []
        max_num_word = 0
        with open(file_name) as f:
            for line in f:
                l = line.strip().split("\t")
                assert(len(l) >= 3)
                query = [self.vocab_to_number[w] for w in l[1].split()]
                if len(query) > max_num_word:
                    max_num_word = len(query)
                examples.append((query, 
                                 self.entity_to_number[l[0]], 
                                 self.entity_to_number[l[2]]))
        self.num_word = max_num_word
        return examples, len(examples)
        
    def _parse_facts(self, file_name, relation_field=1):
        facts = []
        with open(file_name) as f:
            for line in f:
                l = line.strip().split("\t")
                assert(len(l) >= 3)
                facts.append((self.relation_to_number[l[relation_field]], 
                              self.entity_to_number[l[0]], 
                              self.entity_to_number[l[2]]))
        return facts, len(facts)

    def _create_parser(self):
        """Create a parser for language query."""
        parser = {}
        parser["query"] = lambda q: ", ".join(
                            [self.number_to_relation.get(w) for w in q])
        
        d = {}
        for k, v in self.number_to_relation.items():
            d[k] = v
            d[k + self.num_relation] = "inv_" + v
        parser["operator"] = d
        return parser
        
    def is_true(self, q, h, t):
        return False

# ==========================================
# CELL 3: UTILITY FUNCTIONS
# ==========================================
def list_rules(attn_ops, attn_mems, the):
    """
    Given attentions over operators and memories, 
    enumerate all rules and compute the weights for each.
    
    Args:
        attn_ops: a list of num_step vectors, 
                  each vector of length num_operator.
        attn_mems: a list of num_step vectors,
                   with length from 1 to num_step.
        the: early prune by keeping rules with weights > the
    
    Returns:
        a list of (rules, weight) tuples.
        rules is a list of operator ids. 
    
    """
    
    num_step = len(attn_ops)
    paths = {t+1: [] for t in range(num_step)}
    paths[0] = [([], 1.)]
    for t in range(num_step):
        for m, attn_mem in enumerate(attn_mems[t]):
            for p, w in paths[m]:
                paths[t+1].append((p, w * attn_mem))
        if t < num_step - 1:
            new_paths = []           
            for o, attn_op in enumerate(attn_ops[t]):
                for p, w in paths[t+1]:
                    if w * attn_op > the:
                        new_paths.append((p + [o], w * attn_op))
            paths[t+1] = new_paths
    
    # Handle case where paths[num_step] might be empty
    if not paths[num_step]:
        return []
    
    # Find the max weight for thresholding
    max_weight = 0.0
    for _, w in paths[num_step]:
        if w > max_weight:
            max_weight = w
    
    # Use the threshold or the max weight, whichever is smaller
    this_the = min(the, max_weight) if max_weight > 0 else the
    
    # Filter paths based on threshold
    final_paths = [(p, w) for p, w in paths[num_step] if w >= this_the]
    
    # Sort by weight in descending order
    final_paths.sort(key=lambda x: x[1], reverse=True)
    
    return final_paths


def print_rules(q_id, rules, parser, query_is_language):
    """
    Print rules by replacing operator ids with operator names
    and formatting as logic rules.
    
    Args:
        q_id: the query id (the head)
        rules: a list of ([operator ids], weight) (the body)
        parser: a dictionary that convert q_id and operator_id to 
                corresponding names
    
    Returns:
        a list of strings, each string is a printed rule
    """
    
    if len(rules) == 0:
        return []
    
    if not query_is_language: 
        query = parser["query"][q_id]
    else:
        query = parser["query"](q_id)
        
    # assume rules are sorted from high to low
    if not rules:
        return []
    
    max_w = rules[0][1]
    if max_w == 0:
        return []
        
    # compute normalized weights also    
    rules = [[rule[0], rule[1], rule[1]/max_w] for rule in rules]

    printed_rules = [] 
    for rule, w, w_normalized in rules:
        if len(rule) == 0:
            printed_rules.append(
                "%0.3f (%0.3f)\t%s(B, A) <-- equal(B, A)" 
                % (w, w_normalized, query))
        else:
            lvars = [chr(i + 65) for i in range(1 + len(rule))]
            printed_rule = "%0.3f (%0.3f)\t%s(%c, %c) <-- " \
                            % (w, w_normalized, query, lvars[-1], lvars[0]) 
            try:
                for i, literal in enumerate(rule):
                    if not query_is_language:
                        literal_name = parser["operator"][q_id][literal]
                    else:
                        literal_name = parser["operator"][literal]
                    printed_rule += "%s(%c, %c), " \
                                   % (literal_name, lvars[i+1], lvars[i])
                printed_rules.append(printed_rule[0: -2])
            except (KeyError, IndexError) as e:
                # Skip rules with invalid operators
                print(f"Skipping rule with error: {e}")
                continue
    
    return printed_rules


class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))


# ==========================================
# CELL 4: MODEL DEFINITION
# ==========================================
class Learner(object):
    """
    This class builds a computation graph that represents the 
    neural ILP model and handles related graph running acitivies, 
    including update, predict, and get_attentions for given queries. 

    Args:
        option: hyper-parameters
    """

    def __init__(self, option):
        self.seed = option.seed        
        self.num_step = option.num_step
        self.num_layer = option.num_layer
        self.rnn_state_size = option.rnn_state_size
        
        self.norm = not option.no_norm
        self.thr = option.thr
        self.dropout = option.dropout
        self.learning_rate = option.learning_rate
        self.accuracy = option.accuracy
        self.top_k = option.top_k
        
        self.num_entity = option.num_entity
        self.num_operator = option.num_operator
        self.query_is_language = option.query_is_language
        
        if not option.query_is_language:
            self.num_query = option.num_query
            self.query_embed_size = option.query_embed_size       
        else:
            self.vocab_embed_size = option.vocab_embed_size
            self.query_embed_size = self.vocab_embed_size
            self.num_vocab = option.num_vocab
            self.num_word = option.num_word    
        
        np.random.seed(self.seed)
        self._build_graph()

    def _random_uniform_unit(self, r, c):
        """ Initialize random and unit row norm matrix of size (r, c). """
        bound = 6./ np.sqrt(c)
        init_matrix = np.random.uniform(-bound, bound, (r, c))
        init_matrix = np.array(list(map(lambda row: row / np.linalg.norm(row), init_matrix)))
        return init_matrix

    def _clip_if_not_None(self, g, v, low, high):
        """ Clip not-None gradients to (low, high). """
        """ Gradient of T is None if T not connected to the objective. """
        if g is not None:
            return (tf.clip_by_value(g, low, high), v)
        else:
            return (g, v)
    
    def _build_input(self):
        self.tails = tf.compat.v1.placeholder(tf.int32, [None])
        self.heads = tf.compat.v1.placeholder(tf.int32, [None])
        self.targets = tf.one_hot(indices=self.heads, depth=self.num_entity)
            
        if not self.query_is_language:
            self.queries = tf.compat.v1.placeholder(tf.int32, [None, self.num_step])
            self.query_embedding_params = tf.compat.v1.Variable(self._random_uniform_unit(
                                                          self.num_query + 1, # <END> token 
                                                          self.query_embed_size), 
                                                      dtype=tf.float32)
        
            rnn_inputs = tf.nn.embedding_lookup(self.query_embedding_params, 
                                                self.queries)
        else:
            self.queries = tf.compat.v1.placeholder(tf.int32, [None, self.num_step, self.num_word])
            self.vocab_embedding_params = tf.compat.v1.Variable(self._random_uniform_unit(
                                                          self.num_vocab + 1, # <END> token
                                                          self.vocab_embed_size),
                                                      dtype=tf.float32)
            embedded_query = tf.nn.embedding_lookup(self.vocab_embedding_params, 
                                                    self.queries)
            rnn_inputs = tf.reduce_mean(embedded_query, axis=2)

        return rnn_inputs


    def _build_graph(self):
        """ Build a computation graph that represents the model """
        rnn_inputs = self._build_input()                        
        # rnn_inputs: a list of num_step tensors,
        # each tensor of size (batch_size, query_embed_size).    
        self.rnn_inputs = [tf.reshape(q, [-1, self.query_embed_size]) 
                           for q in tf.split(rnn_inputs, 
                                             self.num_step, 
                                             axis=1)]
        
        # Create a simple RNN layer that will work with TF 2.x
        # Convert RNN inputs from a list to a single tensor
        stacked_rnn_inputs = tf.stack(self.rnn_inputs, axis=1)  # [batch_size, num_step, query_embed_size]
        
        # Define the RNN
        rnn = tf.keras.layers.SimpleRNN(
            units=self.rnn_state_size,
            return_sequences=True,
            name="rnn_layer"
        )
        
        # Run the RNN
        rnn_outputs = rnn(stacked_rnn_inputs)  # [batch_size, num_step, rnn_state_size]
        
        # Convert the outputs back to a list
        self.rnn_outputs = [rnn_outputs[:, i, :] for i in range(self.num_step)]
        self.final_state = self.rnn_outputs[-1]  # Just use the last output as the final state
        
        self.W = tf.compat.v1.Variable(np.random.randn(
                                self.rnn_state_size, 
                                self.num_operator), 
                            dtype=tf.float32)
        self.b = tf.compat.v1.Variable(np.zeros(
                                (1, self.num_operator)), 
                            dtype=tf.float32)

        # attention_operators: a list of num_step lists,
        # each inner list has num_operator tensors,
        # each tensor of size (batch_size, 1).
        # Each tensor represents the attention over an operator. 
        self.attention_operators = [tf.split(
                                    tf.nn.softmax(
                                      tf.matmul(rnn_output, self.W) + self.b), 
                                    self.num_operator, 
                                    axis=1) 
                                    for rnn_output in self.rnn_outputs]
        
        # attention_memories: (will be) a list of num_step tensors,
        # each of size (batch_size, t+1),
        # where t is the current step (zero indexed).
        # Each tensor represents the attention over currently populated memory cells. 
        self.attention_memories = []
        
        # memories: (will be) a tensor of size (batch_size, t+1, num_entity),
        # where t is the current step (zero indexed)
        # Then tensor represents currently populated memory cells.
        self.memories = tf.expand_dims(
                         tf.one_hot(
                                indices=self.tails, 
                                depth=self.num_entity), 1) 

        self.database = {r: tf.compat.v1.sparse_placeholder(
                            dtype=tf.float32, 
                            name="database_%d" % r)
                            for r in range(self.num_operator//2)}
        
        for t in range(self.num_step):
            self.attention_memories.append(
                            tf.nn.softmax(
                            tf.squeeze(
                                tf.matmul(
                                    tf.expand_dims(self.rnn_outputs[t], 1), 
                                    tf.stack(self.rnn_outputs[0:t+1], axis=2)), 
                            axis=[1])))
            
            # memory_read: tensor of size (batch_size, num_entity)
            memory_read = tf.squeeze(
                            tf.matmul(
                                tf.expand_dims(self.attention_memories[t], 1), 
                                self.memories),
                            axis=[1])
            
            if t < self.num_step - 1:
                # database_results: (will be) a list of num_operator tensors,
                # each of size (batch_size, num_entity).
                database_results = []    
                memory_read = tf.transpose(memory_read)
                for r in range(self.num_operator//2):
                    for op_matrix, op_attn in zip(
                                    [self.database[r], 
                                     tf.sparse.transpose(self.database[r])],
                                    [self.attention_operators[t][r], 
                                     self.attention_operators[t][r+self.num_operator//2]]):
                        product = tf.sparse.sparse_dense_matmul(op_matrix, memory_read)
                        database_results.append(tf.transpose(product) * op_attn)

                added_database_results = tf.add_n(database_results)
                if self.norm:
                    added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keepdims=True))                
                
                if self.dropout > 0.:
                  added_database_results = tf.nn.dropout(added_database_results, rate=self.dropout)

                # Populate a new cell in memory by concatenating.  
                self.memories = tf.concat( 
                    [self.memories, 
                    tf.expand_dims(added_database_results, 1)],
                    axis=1)
            else:
                self.predictions = memory_read
                           
        self.final_loss = - tf.reduce_sum(self.targets * tf.math.log(tf.maximum(self.predictions, self.thr)), 1)
        
        if not self.accuracy:
            self.in_top = tf.nn.in_top_k(
                predictions=self.predictions,
                targets=self.heads,
                k=self.top_k)
        else:
            self.in_top = tf.math.equal(
                tf.math.argmax(self.predictions, axis=1),
                tf.cast(self.heads, tf.int64))
        
        self.mean_loss = tf.reduce_mean(self.final_loss)
        
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = optimizer.compute_gradients(self.mean_loss)
        capped_gvs = list(map(lambda x: self._clip_if_not_None(x[0], x[1], -1., 1.), gvs))
        self.train_op = optimizer.apply_gradients(capped_gvs)
        
    def _run_graph(self, sess, qq, hh, tt, mdb, to_fetch):
        feed = {}
        feed[self.heads] = hh
        feed[self.tails] = tt
        feed[self.queries] = qq
        for r in range(self.num_operator//2):
            # Create feed dict entries for sparse tensor separately
            feed[self.database[r].indices] = mdb[r].indices
            feed[self.database[r].values] = mdb[r].values
            feed[self.database[r].dense_shape] = mdb[r].dense_shape
        
        return sess.run(to_fetch, feed)

    def update(self, sess, qq, hh, tt, mdb):
        _, mean_loss, in_top \
            = self._run_graph(sess, qq, hh, tt, mdb, 
                             [self.train_op, self.mean_loss, self.in_top])
        return mean_loss, in_top
    
    def predict(self, sess, qq, hh, tt, mdb):
        mean_loss, in_top \
            = self._run_graph(sess, qq, hh, tt, mdb, 
                             [self.mean_loss, self.in_top])
        return mean_loss, in_top

    def get_predictions_given_queries(self, sess, qq, hh, tt, mdb):
        in_top, predictions \
            = self._run_graph(sess, qq, hh, tt, mdb, 
                             [self.in_top, self.predictions])
        return in_top, predictions
    
    def get_attentions_given_queries(self, sess, queries):
        """
        Get attention values for operators.
        
        Args:
            sess: TensorFlow session
            queries: Query indices

        Returns:
            A list of attention values for each step and operator
        """
        feed = {}
        feed[self.queries] = queries
        
        # Get the attention operators directly
        attention_ops = []
        for i in range(len(self.attention_operators)):
            ops_i = []
            for j in range(len(self.attention_operators[i])):
                # Run each operator attention individually
                op_values = sess.run(self.attention_operators[i][j], feed)
                ops_i.append(op_values)
            attention_ops.append(ops_i)
        
        return attention_ops
    
    def get_vocab_embedding(self, sess):
        return sess.run(self.vocab_embedding_params)

# ==========================================
# CELL 5: EXPERIMENT CLASS
# ==========================================
class Experiment():
    """
    This class handles all experiments related activties, 
    including training, testing, early stop, and visualize
    results, such as get attentions and get rules. 

    Args:
        sess: a TensorFlow session 
        saver: a TensorFlow saver
        option: an Option object that contains hyper parameters
        learner: an inductive learner that can  
                 update its parameters and perform inference.
        data: a Data object that can be used to obtain 
              num_batch_train/valid/test,
              next_train/valid/test,
              and a parser for get rules.
    """
    
    def __init__(self, sess, saver, option, learner, data):
        self.sess = sess
        self.saver = saver
        self.option = option
        self.learner = learner
        self.data = data
        # helpers
        self.msg_with_time = lambda msg: \
                "%s Time elapsed %0.2f hrs (%0.1f mins)" \
                % (msg, (time.time() - self.start) / 3600., 
                        (time.time() - self.start) / 60.)

        self.start = time.time()
        self.epoch = 0
        self.best_valid_loss = np.inf
        self.best_valid_in_top = 0.
        self.train_stats = []
        self.valid_stats = []
        self.test_stats = []
        self.early_stopped = False
        self.log_file = open(os.path.join(self.option.this_expsdir, "log.txt"), "w")


    def one_epoch(self, mode, num_batch, next_fn):
        epoch_loss = []
        epoch_in_top = []
        for batch in range(num_batch):
            if (batch+1) % max(1, (num_batch // self.option.print_per_batch)) == 0:
                sys.stdout.write("%d/%d\t" % (batch+1, num_batch))
                sys.stdout.flush()
            
            (qq, hh, tt), mdb = next_fn()
            if mode == "train":
                run_fn = self.learner.update
            else:
                run_fn = self.learner.predict
            loss, in_top = run_fn(self.sess,
                                  qq, 
                                  hh, 
                                  tt, 
                                  mdb) 
            # Handle scalar loss value
            if isinstance(loss, (np.float32, np.float64, float)):
                epoch_loss.append(loss)
            else:
                epoch_loss.extend(list(loss))
                
            # Handle scalar or array in_top value
            if isinstance(in_top, (bool, np.bool_)):
                epoch_in_top.append(in_top)
            else:
                epoch_in_top.extend(list(in_top))
                                    
        msg = self.msg_with_time(
                "Epoch %d mode %s Loss %0.4f In top %0.4f." 
                % (self.epoch+1, mode, np.mean(epoch_loss), np.mean(epoch_in_top)))
        print(msg)
        self.log_file.write(msg + "\n")
        return epoch_loss, epoch_in_top

    def one_epoch_train(self):
        if self.epoch > 0 and self.option.resplit:
            self.data.train_resplit(self.option.no_link_percent)
        loss, in_top = self.one_epoch("train", 
                                      self.data.num_batch_train, 
                                      self.data.next_train)
        
        self.train_stats.append([loss, in_top])
        
    def one_epoch_valid(self):
        loss, in_top = self.one_epoch("valid", 
                                      self.data.num_batch_valid, 
                                      self.data.next_valid)
        self.valid_stats.append([loss, in_top])
        self.best_valid_loss = min(self.best_valid_loss, np.mean(loss))
        self.best_valid_in_top = max(self.best_valid_in_top, np.mean(in_top))

    def one_epoch_test(self):
        loss, in_top = self.one_epoch("test", 
                                      self.data.num_batch_test,
                                      self.data.next_test)
        self.test_stats.append([loss, in_top])
    
    def early_stop(self):
        loss_improve = self.best_valid_loss == np.mean(self.valid_stats[-1][0])
        in_top_improve = self.best_valid_in_top == np.mean(self.valid_stats[-1][1])
        if loss_improve or in_top_improve:
            return False
        else:
            if self.epoch < self.option.min_epoch:
                return False
            else:
                return True

    def train(self):
        while (self.epoch < self.option.max_epoch and not self.early_stopped):
            self.one_epoch_train()
            self.one_epoch_valid()
            self.one_epoch_test()
            self.epoch += 1
            model_path = self.saver.save(self.sess, 
                                         self.option.model_path,
                                         global_step=self.epoch)
            print("Model saved at %s" % model_path)
            
            if self.early_stop():
                self.early_stopped = True
                print("Early stopped at epoch %d" % (self.epoch))
        
        all_test_in_top = [np.mean(x[1]) for x in self.test_stats]
        best_test_epoch = np.argmax(all_test_in_top)
        best_test = all_test_in_top[best_test_epoch]
        
        msg = "Best test in top: %0.4f at epoch %d." % (best_test, best_test_epoch + 1)       
        print(msg)
        self.log_file.write(msg + "\n")
        pickle.dump([self.train_stats, self.valid_stats, self.test_stats],
                    open(os.path.join(self.option.this_expsdir, "results.pckl"), "wb"))

    def get_predictions(self):
        if self.option.query_is_language:
            all_accu = []
            all_num_preds = []
            all_num_preds_no_mistake = []

        f = open(os.path.join(self.option.this_expsdir, "test_predictions.txt"), "w")
        if self.option.get_phead:
            f_p = open(os.path.join(self.option.this_expsdir, "test_preds_and_probs.txt"), "w")
        all_in_top = []
        for batch in range(self.data.num_batch_test):
            if (batch+1) % max(1, (self.data.num_batch_test // self.option.print_per_batch)) == 0:
                sys.stdout.write("%d/%d\t" % (batch+1, self.data.num_batch_test))
                sys.stdout.flush()
            (qq, hh, tt), mdb = self.data.next_test()
            in_top, predictions_this_batch \
                    = self.learner.get_predictions_given_queries(self.sess, qq, hh, tt, mdb)
            all_in_top += list(in_top)

            for i, (q_array, h, t) in enumerate(zip(qq, hh, tt)):
                # Get the first element of the query array (containing the query ID)
                q = q_array[0]
                p_head = predictions_this_batch[i, h]
                if self.option.adv_rank:
                    eval_fn = lambda j_p: j_p[1] >= p_head and (j_p[0] != h)
                elif self.option.rand_break:
                    eval_fn = lambda j_p: (j_p[1] > p_head) or ((j_p[1] == p_head) and (j_p[0] != h) and (np.random.uniform() < 0.5))
                else:
                    eval_fn = lambda j_p: (j_p[1] > p_head)
                this_predictions = filter(eval_fn, enumerate(predictions_this_batch[i, :]))
                this_predictions = sorted(this_predictions, key=lambda x: x[1], reverse=True)

                if self.option.query_is_language:
                    all_num_preds.append(len(this_predictions))
                    mistake = False
                    for k, _ in this_predictions:
                        assert(k != h)
                        if not self.data.is_true(q, k, t):
                            mistake = True
                            break
                    all_accu.append(not mistake)
                    if not mistake:
                        all_num_preds_no_mistake.append(len(this_predictions))
                else:
                    this_predictions.append((h, p_head))
                    this_predictions = [self.data.number_to_entity[j] for j, _ in this_predictions]
                    q_string = self.data.parser["query"][q]
                    h_string = self.data.number_to_entity[h]
                    t_string = self.data.number_to_entity[t]
                    to_write = [q_string, h_string, t_string] + this_predictions
                    f.write(",".join(to_write) + "\n")
                    if self.option.get_phead:
                        f_p.write(",".join(to_write + [str(p_head)]) + "\n")
        f.close()
        if self.option.get_phead:
            f_p.close()
        
        if self.option.query_is_language:
            print("Averaged num of preds", np.mean(all_num_preds))
            print("Averaged num of preds for no mistake", np.mean(all_num_preds_no_mistake))
            msg = "Accuracy %0.4f" % np.mean(all_accu)
            print(msg)
            self.log_file.write(msg + "\n")
        
    def get_attentions(self):
        """
        Get attentions for each rule and store in a file.
        """
        f = open(os.path.join(self.option.this_expsdir, "attention.txt"), "w")
        
        if not self.option.query_is_language:
            for query_idx in range(self.data.num_query):
                query_string = self.data.parser["query"][query_idx]
                if not query_string.startswith("inv_"):
                    # shape (1, num_step)
                    queries = np.zeros((1, self.option.num_step), dtype=np.int32)
                    # padded with <END> = num_query
                    queries.fill(self.data.num_query)
                    queries[0, 0] = query_idx
                    attention_ops = self.learner.get_attentions_given_queries(self.sess, queries)
                    
                    # attention_ops is a list of num_step lists
                    # where each inner list has num_operator tensors of shape (1,)
                    # each inner list represents the attention over an operator
                    ops = []
                    for ops_each_step in attention_ops:
                        ops_for_head = []
                        for values in ops_each_step:
                            ops_for_head.append(float(values[0]))
                        ops.append(ops_for_head)
                    
                    f.write("%s\n" % query_string)
                    for ops_each_step in ops:
                        f.write("%s\n" % str(ops_each_step))
        f.close()

    def get_rules(self):
        """
        Extract rules for each query and store in a file.
        """
        f_all = open(os.path.join(self.option.this_expsdir, "rules.txt"), "w")
        f_filtered = open(os.path.join(self.option.this_expsdir, "rules_filtered.txt"), "w")

        if not self.option.query_is_language:
            for query_idx in self.data.query_for_rules:
                query_string = self.data.parser["query"][query_idx]
                
                # shape (1, num_step)
                queries = np.zeros((1, self.option.num_step), dtype=np.int32)
                # padded with <END> = num_query
                queries.fill(self.data.num_query)
                queries[0, 0] = query_idx
                
                attention_ops = self.learner.get_attentions_given_queries(self.sess, queries)
                
                # attention_ops is a list of num_step lists
                # where each inner list has num_operator tensors of shape (batch_size, 1)
                ops = []
                for ops_each_step in attention_ops:
                    ops_for_head = []
                    for values in ops_each_step:
                        # Extract the value from the tensor ensuring it's a scalar
                        if hasattr(values, 'shape') and len(values.shape) > 0:
                            value = float(values[0][0])  # Access batch 0, dim 0
                        else:
                            value = float(values)
                        ops_for_head.append(value)
                    ops.append(ops_for_head)
                
                # Not using attention_memories (self.learner.attention_memories)
                # Because it's a placeholder for memory attention
                # which depends on input, here we define some dummy ones
                mems = [[1.0] for _ in range(self.option.num_step)]
                
                # Handle possible empty paths
                try:
                    rules = list_rules(ops, mems, self.option.rule_thr)
                    if rules:
                        printed_rules = print_rules(query_idx, rules, self.data.parser, self.option.query_is_language)
                        
                        f_all.write("%s\n" % query_string)
                        for printed_rule in printed_rules:
                            f_all.write("%s\n" % printed_rule)
                        
                        # Write only interesting rules to filtered file
                        if len(printed_rules) > 0 and rules[0][0]:  # Check if rules exist and first rule body is not empty
                            f_filtered.write("%s\n" % query_string)
                            for printed_rule in printed_rules:
                                f_filtered.write("%s\n" % printed_rule)
                except Exception as e:
                    print(f"Error extracting rules for query {query_string}: {e}")
                
        f_all.close()
        f_filtered.close()
        
    def get_vocab_embedding(self):
        """
        Get embeddings for vocabulary
        """
        vocab_embedding = self.learner.get_vocab_embedding(self.sess)
        with open(os.path.join(self.option.this_expsdir, "vocab_embed.txt"), "w") as f:
            for i in range(len(vocab_embedding) - 1):
                f.write("%s %s\n" 
                        % (self.data.number_to_entity[i],
                           " ".join(map(str, list(vocab_embedding[i, :])))))

    def close_log_file(self):
        self.log_file.close()


# ==========================================
# CELL 6: MAIN FUNCTION
# ==========================================
def main():
    # Use the global parser instead of redefining it
    d = vars(parser.parse_args())
    option = Option(d)
    if option.exp_name is None:
      option.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
      option.tag = option.exp_name  
    if option.resplit:
      assert not option.no_extra_facts
    if option.accuracy:
      assert option.top_k == 1
    
    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Set default data directory if none provided
    if option.datadir is None:
        option.datadir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "family")
        print(f"No datadir specified. Using default: {option.datadir}")
    
    # Set default experiments directory if none provided
    if option.exps_dir is None:
        option.exps_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
        if not os.path.exists(option.exps_dir):
            os.makedirs(option.exps_dir)
        print(f"No exps_dir specified. Using default: {option.exps_dir}")
       
    if not option.query_is_language:
        data = Data(option.datadir, option.seed, option.type_check, option.domain_size, option.no_extra_facts, option)
    else:
        data = DataPlus(option.datadir, option.seed)
    print("Data prepared.")

    option.num_entity = data.num_entity
    option.num_operator = data.num_operator
    if not option.query_is_language:
        option.num_query = data.num_query
    else:
        option.num_vocab = data.num_vocab 
        option.num_word = data.num_word # the number of words in each query

    option.this_expsdir = os.path.join(option.exps_dir, option.tag)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)
    option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
    if not os.path.exists(option.ckpt_dir):
        os.makedirs(option.ckpt_dir)
    option.model_path = os.path.join(option.ckpt_dir, "model")
    
    option.save()
    print("Option saved.")

    learner = Learner(option)
    print("Learner built.")

    saver = tf.compat.v1.train.Saver(max_to_keep=option.max_epoch)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False
    config.allow_soft_placement = True
    with tf.compat.v1.Session(config=config) as sess:
        tf.compat.v1.set_random_seed(option.seed)
        sess.run(tf.compat.v1.global_variables_initializer())
        print("Session initialized.")

        if option.from_model_ckpt is not None:
            saver.restore(sess, option.from_model_ckpt)
            print("Checkpoint restored from model %s" % option.from_model_ckpt)

        data.reset(option.batch_size)
        experiment = Experiment(sess, saver, option, learner, data)
        print("Experiment created.")

        if not option.no_train:
            print("Start training...")
            experiment.train()
        
        if not option.no_preds:
            print("Start getting test predictions...")
            experiment.get_predictions()
        
        if not option.no_rules:
            print("Start getting rules...")
            experiment.get_rules()

        if option.get_vocab_embed:
            print("Start getting vocabulary embedding...")
            experiment.get_vocab_embedding()
            
    experiment.close_log_file()
    print("="*36 + "Finish" + "="*36)


if __name__ == "__main__":
    # Import sys here because it was missing
    import sys
    
    # ==========================================
    # CELL 7: NOTEBOOK ENTRY POINT
    # ==========================================
    # When running in a notebook environment, use this section instead of main()
    # Comment out this section if running as a script
    
    
    # Set arguments for the experiment
    args = [
        "--seed", "42",
        "--datadir", "../datasets/family",
        "--batch_size", "32",
        "--max_epoch", "5",
        "--num_step", "3",
        "--rnn_state_size", "128",
        "--learning_rate", "0.001",
        "--top_k", "10"
    ]
    
    # Parse the arguments manually
    parsed_args = parser.parse_args(args)
    d = vars(parsed_args)
    option = Option(d)
    
    # Set experiment tag
    option.tag = time.strftime("%y-%m-%d-%H-%M")
    
    # Set CUDA visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Set default data directory if none provided
    if option.datadir is None:
        option.datadir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "family")
        print(f"No datadir specified. Using default: {option.datadir}")

    # Set default experiments directory
    option.exps_dir = "../notebook_experiments"
    if not os.path.exists(option.exps_dir):
        os.makedirs(option.exps_dir)

    # Load data
    data = Data(option.datadir, option.seed, option.type_check, option.domain_size, option.no_extra_facts, option)
    print("Data prepared.")

    # Set options based on data
    option.num_entity = data.num_entity
    option.num_operator = data.num_operator
    option.num_query = data.num_query

    # Create experiment directory
    option.this_expsdir = os.path.join(option.exps_dir, option.tag)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)
    option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
    if not os.path.exists(option.ckpt_dir):
        os.makedirs(option.ckpt_dir)
    option.model_path = os.path.join(option.ckpt_dir, "model")

    option.save()
    print("Option saved.")

    # Build learner
    learner = Learner(option)
    print("Learner built.")

    # Create TensorFlow session
    saver = tf.compat.v1.train.Saver(max_to_keep=option.max_epoch)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    config.allow_soft_placement = True
    
    with tf.compat.v1.Session(config=config) as sess:
        tf.compat.v1.set_random_seed(option.seed)
        sess.run(tf.compat.v1.global_variables_initializer())
        print("Session initialized.")
        
        data.reset(option.batch_size)
        experiment = Experiment(sess, saver, option, learner, data)
        print("Experiment created.")
        
        # Training
        print("Start training...")
        experiment.train()
        
        # Get predictions
        print("Start getting test predictions...")
        experiment.get_predictions()
        
        # Get rules
        print("Start getting rules...")
        experiment.get_rules()
        
    experiment.close_log_file()
    print("="*36 + "Finish" + "="*36)
    
    # Run the main function when executing as a script
    main()
