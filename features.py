from collections import Callable
import itertools
import numpy as np
from corenlp import TimeoutException

class CallableAnnotate(Callable):
    def __init__(self, client):
        self._client = client

    def __call__(self, doc):
        try:
            ann = self._client.annotate(doc, annotators="ner".split())
        except TimeoutException as e:
            print "TimeoutException: ", doc
            return []
        return itertools.chain.from_iterable([[t.lemma for t in s.token if t.pos != "CD"] for s in ann.sentence])

def get_text_length(docs):
    return np.array([len(doc) for doc in docs]).reshape(-1, 1)

class CallableCountPronouns(Callable):
    def __init__(self, client):
        self._client = client

    def __call__(self, docs):
        result = []
        for doc in docs:
            ann = self._client.annotate(doc, annotators="ner".split())
            result.append(sum(itertools.chain.from_iterable([[1 for t in s.token if (t.pos == "PRP" or t.pos == "PRP$")] for s in ann.sentence])))
        return np.asarray(result).reshape(-1, 1)

