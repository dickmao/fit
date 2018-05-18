from collections import Callable
import itertools
import numpy as np
from corenlp import TimeoutException

class CallableAnnotate(Callable):
    def __init__(self, client):
        self._client = client
        self._foo = []

    def __call__(self, doc):
        if self._foo:
            return self._foo
        try:
            ann = self._client.annotate(doc, annotators="ner".split())
        except TimeoutException as e:
            print "TimeoutException: ", doc.encode('utf-8')
            return []
        self._foo = itertools.chain.from_iterable([[t.lemma for t in s.token if t.pos != "CD"] for s in ann.sentence])
        return self._foo

def get_text_length(docs):
    return np.array([len(doc) for doc in docs]).reshape(-1, 1)

class CallableCountPronouns(Callable):
    def __init__(self, client):
        self._client = client

    def __call__(self, docs):
        result = []
        for doc in docs:
            try:
                ann = self._client.annotate(doc, annotators="ner".split())
                result.append(sum(itertools.chain.from_iterable([[1 for t in s.token if (t.pos == "PRP" or t.pos == "PRP$")] for s in ann.sentence])))
            except TimeoutException as e:
                print "TimeoutException: ", doc.encode('utf-8')
                result.append(0)
        return np.asarray(result).reshape(-1, 1)
