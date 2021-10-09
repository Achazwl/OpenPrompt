# MIT License
# Copyright (c) 2021 THUDM
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This file contains the logic for loading data for LAMA tasks.
"""

import os
import re
import json, csv

from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import *
import tokenizers
import sys

from transformers.tokenization_utils import PreTrainedTokenizer

from openprompt.utils.logging import logger

from openprompt.data_utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor


from transformers import RobertaTokenizer

class LAMAProcessor(DataProcessor):
    """This dataset is a variant of the original `LAMA <https://github.com/facebookresearch/LAMA>`_ dataset, which adds train and dev split, and was created by `AutoPrompt <https://github.com/ucinlp/autoprompt>`_ .

    The code of this Processor refers to `the data processing phase in P-tuning <https://github.com/THUDM/P-tuning/tree/main/LAMA>`_

    Args:
        model_name (str): PLM model name.
        tokenizer (PreTrainedTokenizer): tokenizer of the corresponding PLM
        vocab_strategy (str): ["original", "share", "lama"]. "original" use the vocab of PLM; "share" use the vocab of LAMA-29k; "lama" use the vocab of LAMA-34k.
        relation_id (str, optional): [description]. Defaults to "P1001".

    Examples: # TODO test needed
    """
    def __init__(self,
                 base_path: str = "/home/hx/OpenPrompt/datasets/LAMA",
                 model_name: str = "roberta-large",
                 tokenizer: PreTrainedTokenizer = RobertaTokenizer.from_pretrained("roberta-large"),
                 vocab_strategy: str = "share",
                 single_relation_id: Optional[str] = None # e.g. "P1001"
                ):
        super().__init__()
        self.relations = {
            "P47": { "template": "[X] shares border with [Y] .", "label": "shares border with", "description": "countries or administrative subdivisions, of equal level, that this item borders, either by land or water", "type": "N-M"},
            "P138": { "template": "[X] is named after [Y] .", "label": "named after", "description": "entity or event that inspired the subject's name, or namesake (in at least one language)", "type": "N-1"},
            "P364": { "template": "The original language of [X] is [Y] .", "label": "original language of film or TV show", "description": "language in which a film or a performance work was originally created. Deprecated for written works; use P407 (\"language of work or name\") instead.", "type": "N-1"},
            "P54": { "template": "[X] plays with [Y] .", "label": "member of sports team", "description": "sports teams or clubs that the subject currently represents or formerly represented", "type": "N-1"},
            "P463": { "template": "[X] is a member of [Y] .", "label": "member of", "description": "organization or club to which the subject belongs. Do not use for membership in ethnic or social groups, nor for holding a position such as a member of parliament (use P39 for that).", "type": "N-M"},
            "P101": { "template": "[X] works in the field of [Y] .", "label": "field of work", "description": "specialization of a person or organization; see P106 for the occupation", "type": "N-M"},
            "P1923": { "template": "[Y] participated in the [X] .", "label": "participating team", "description": "Like 'Participant' (P710) but for teams. For an event like a cycle race or a football match you can use this property to list the teams and P710 to list the individuals (with 'member of sports team' (P54)' as a qualifier for the individuals)", "type": "N-M"},
            "P106": { "template": "[X] is a [Y] by profession .", "label": "occupation", "description": "occupation of a person; see also \"field of work\" (Property:P101), \"position held\" (Property:P39)", "type": "N-M"},
            "P527": { "template": "[X] consists of [Y] .", "label": "has part", "description": "part of this subject; inverse property of \"part of\" (P361). See also \"has parts of the class\" (P2670).", "type": "N-M"},
            "P102": { "template": "[X] is a member of the [Y] political party .", "label": "member of political party", "description": "the political party of which this politician is or has been a member", "type": "N-1"},
            "P530": { "template": "[X] maintains diplomatic relations with [Y] .", "label": "diplomatic relation", "description": "diplomatic relations of the country", "type": "N-M"},
            "P176": { "template": "[X] is produced by [Y] .", "label": "manufacturer", "description": "manufacturer or producer of this product", "type": "N-1"},
            "P27": { "template": "[X] is [Y] citizen .", "label": "country of citizenship", "description": "the object is a country that recognizes the subject as its citizen", "type": "N-M"},
            "P407": { "template": "[X] was written in [Y] .", "label": "language of work or name", "description": "language associated with this creative work (such as books, shows, songs, or websites) or a name (for persons use P103 and P1412)", "type": "N-1"},
            "P30": { "template": "[X] is located in [Y] .", "label": "continent", "description": "continent of which the subject is a part", "type": "N-1"},
            "P178": { "template": "[X] is developed by [Y] .", "label": "developer", "description": "organisation or person that developed the item", "type": "N-M"},
            "P1376": { "template": "[X] is the capital of [Y] .", "label": "capital of", "description": "country, state, department, canton or other administrative division of which the municipality is the governmental seat", "type": "1-1"},
            "P131": { "template": "[X] is located in [Y] .", "label": "located in the administrative territorial entity", "description": "the item is located on the territory of the following administrative entity. Use P276 (location) for specifying the location of non-administrative places and for items about events", "type": "N-1"},
            "P1412": { "template": "[X] used to communicate in [Y] .", "label": "languages spoken, written or signed", "description": "language(s) that a person speaks or writes, including the native language(s)", "type": "N-M"},
            "P108": { "template": "[X] works for [Y] .", "label": "employer", "description": "person or organization for which the subject works or worked", "type": "N-M"},
            "P136": { "template": "[X] plays [Y] music .", "label": "genre", "description": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic", "type": "N-1"},
            "P17": { "template": "[X] is located in [Y] .", "label": "country", "description": "sovereign state of this item; don't use on humans", "type": "N-1"},
            "P39": { "template": "[X] has the position of [Y] .", "label": "position held", "description": "subject currently or formerly holds the object position or public office", "type": "N-M"},
            "P264": { "template": "[X] is represented by music label [Y] .", "label": "record label", "description": "brand and trademark associated with the marketing of subject music recordings and music videos", "type": "N-1"},
            "P276": { "template": "[X] is located in [Y] .", "label": "location", "description": "location of the item, physical object or event is within. In case of an administrative entity use P131. In case of a distinct terrain feature use P706.", "type": "N-1"},
            "P937": { "template": "[X] used to work in [Y] .", "label": "work location", "description": "location where persons were active", "type": "N-M"},
            "P140": { "template": "[X] is affiliated with the [Y] religion .", "label": "religion", "description": "religion of a person, organization or religious building, or associated with this subject", "type": "N-1"},
            "P1303": { "template": "[X] plays [Y] .", "label": "instrument", "description": "musical instrument that a person plays", "type": "N-M"},
            "P127": { "template": "[X] is owned by [Y] .", "label": "owned by", "description": "owner of the subject", "type": "N-1"},
            "P103": { "template": "The native language of [X] is [Y] .", "label": "native language", "description": "language or languages a person has learned from early childhood", "type": "N-1"},
            "P190": { "template": "[X] and [Y] are twin cities .", "label": "twinned administrative body", "description": "twin towns, sister cities, twinned municipalities and other localities that have a partnership or cooperative agreement, either legally or informally acknowledged by their governments", "type": "N-M"},
            "P1001": { "template": "[X] is a legal term in [Y] .", "label": "applies to jurisdiction", "description": "the item (an institution, law, public office ...) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, ...)", "type": "N-M"},
            "P31": { "template": "[X] is a [Y] .", "label": "instance of", "description": "that class of which this subject is a particular example and member (subject typically an individual member with a proper name label); different from P279; using this property as a qualifier is deprecated\u2014use P2868 or P3831 instead", "type": "N-M"},
            "P495": { "template": "[X] was created in [Y] .", "label": "country of origin", "description": "country of origin of this item (creative work, food, phrase, product, etc.)", "type": "N-1"},
            "P159": { "template": "The headquarter of [X] is in [Y] .", "label": "headquarters location", "description": "specific location where an organization's headquarters is or has been situated. Inverse property of \"occupant\" (P466).", "type": "N-1"},
            "P36": { "template": "The capital of [X] is [Y] .", "label": "capital", "description": "primary city of a country, state or other type of administrative territorial entity", "type": "1-1"},
            "P740": { "template": "[X] was founded in [Y] .", "label": "location of formation", "description": "location where a group or organization was formed", "type": "N-1"},
            "P361": { "template": "[X] is part of [Y] .", "label": "part of", "description": "object of which the subject is a part (it's not useful to link objects which are themselves parts of other objects already listed as parts of the subject). Inverse property of \"has part\" (P527, see also \"has parts of the class\" (P2670)).", "type": "N-1"},
        }

        self.tokenizer = tokenizer
        self.model_name = model_name
        self.label_mapping = tokenizer.get_vocab()
        self.allowed_vocab_ids = [self.label_mapping[vocab] for vocab in self._get_allowed_vocab(model_name, vocab_strategy, base_path)]
        self.single_relation_id = single_relation_id

    def _get_allowed_vocab(self, model_name, strategy, base_path):
        if strategy == "original":
            return self.labels
        elif strategy == "share":
            with open(os.path.join(base_path, '29k-vocab.json')) as f:
                shared_vocab = json.load(f)
                if 'gpt' in model_name:
                    return shared_vocab['gpt2-xl']
                elif 'roberta' in model_name or 'megatron' in model_name:
                    return shared_vocab['roberta-large']
                else:
                    assert model_name in shared_vocab
                    return shared_vocab[model_name]
        elif strategy == "lama":
            with open(os.path.join(base_path, '34k-vocab.json')) as f:
                lama_vocab = json.load(f)
                if 'gpt' in model_name:
                    return lama_vocab['gpt2-xl']
                elif 'roberta' in model_name or 'megatron' in model_name:
                    return lama_vocab['roberta-large']
                else:
                    assert model_name in lama_vocab
                    return lama_vocab[model_name]
        else:
            raise ValueError('vocab_strategy must be "original", "share" or "lama"')

    def get_examples(self, data_dir, split):
        examples = []
        relation_ids = [self.single_relation_id] if self.single_relation_id else self.relations.keys()
        for relation_id in relation_ids:
            path = os.path.join(data_dir, "fact-retrieval/original/{}/{}.jsonl".format(relation_id, split))
            try:
                with open(path, encoding='utf8') as f:
                    for choicex, line in enumerate(f):
                        example_json = json.loads(line)
                        token_ids = self.tokenizer(" "+example_json["obj_label"], add_special_tokens=False)["input_ids"]
                        if len(token_ids) != 1 or token_ids[0] not in self.allowed_vocab_ids:
                            continue
                        template = self.relations[relation_id]["template"]
                        template = template.replace('[X]', example_json["sub_label"])

                        example = InputExample(guid=str(choicex), text_a=template.split('[Y]')[0], text_b=template.split('[Y]')[1], label=token_ids[0])
                        examples.append(example)
            except:
                pass
        return examples

PROCESSORS = {
    "lama": LAMAProcessor,
}
