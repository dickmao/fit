#!/usr/bin/python

from __future__ import division
import os, errno, operator, re, sys, subprocess, signal, redis
import random
import numpy as np
import enchant
from nltk import bigrams,ConditionalFreqDist,FreqDist,pos_tag,pos_tag_sents
from gensim import parsing, matutils, interfaces, corpora, models, similarities, summarization
from gensim.utils import lemmatize
from nltk import collocations, association, text, tree
from gensim.corpora.mmcorpus import MmCorpus
from gensim.matutils import corpus2csc
from gensim.similarities.docsim import SparseMatrixSimilarity
from reader import Json100CorpusReader
import itertools, shutil, requests
from collections import Counter
from bisect import bisect_left
from tempfile import mkstemp

import re
from lxml import etree
from os import listdir
import json
import cPickle
from collections import defaultdict
from math import radians, sin, cos, sqrt, asin
from nltk.corpus.util import LazyCorpusLoader

from os import listdir
from os.path import getmtime, join, realpath
import dateutil.parser
from pytz import utc
from datetime import datetime
from dateutil.tz import tzlocal
from time import time
import argparse

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def reflexive(x):
    # for CountVectorizer 'analyzer' which cannot accept a lambda due to pickle caching
    return x

def get_text_length(x):
    import numpy
    return numpy.array([len(t) for t in x]).reshape(-1, 1)

def argparse_dirtype(astring):
    if not os.path.isdir(astring):
        raise argparse.ArgumentError
    return astring

def datetime_parser(json_dict):
    for k,v in json_dict.iteritems():
        try:
            json_dict[k] = dateutil.parser.parse(v)
        except (ValueError, AttributeError):
            pass
    return json_dict

def determine_payfor_fencepost(dt1, thresh):
    Markers = sorted([f for f in os.listdir(args.odir) if re.search(r'Marker\..*\.json$', f)], reverse=True)
    jsons = []
    for m in Markers:
        within = False
        with open(join(args.odir, m), 'r') as fp:
            url2dt = json.load(fp, object_hook=datetime_parser)
            for url,dt0 in url2dt.iteritems():
                if (dt1 - dt0).days < thresh:
                    within = True
                    break
        if within:
            jsons.append("{}.{}.json".format(spider, m.split('.')[1]))
        else:
            break
    return jsons

def CorpusDedupe(cr):
    # dict.doc2bow makes:
    #   corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
    #             [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
    #             [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],      ]
    try:
        unduped = joblib.load(join(args.odir, 'unduped.pkl'), mmap_mode='r')
        duped = joblib.load(join(args.odir, 'duped.pkl'), mmap_mode='r')
    except IOError as e:
        unduped = set()
        duped = set()

    dictionary = corpora.Dictionary()
    corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in list(cr)]
    tfidf = models.TfidfModel(corpus)
    
    tempf  = mkstemp()[1]
    corpora.MmCorpus.serialize(tempf, tfidf[corpus], id2word=dictionary.id2token)
    mmcorpus = corpora.MmCorpus(tempf)
    unduped = unduped.intersection(ids)
    duped = duped.intersection(ids)
    new_ids = set(ids) - unduped.union(duped)
    new_indices = [ids.index(i) for i in new_ids]
    new_sim = SparseMatrixSimilarity(mmcorpus)[mmcorpus[new_indices]]

    t0 = time()
    # ConcatenatedCorpusView cannot seem to random access; must iterate sequentially lest
    # block reader get ahead of itself
    assert(len(mmcorpus) == len(ids))
    for i,z in enumerate(zip(new_ids, new_indices)):
        for dj in np.where((new_sim[i] > 0.61) & [j!=z[1] for j in range(len(ids))])[0]:
            duped.update([z[0], ids[dj]])
    print("(n-1) + ... (n-k) = k(n - (k+1)/2) took %0.3fs" % (time() - t0))

    unduped.update(new_ids - duped)
    joblib.dump(unduped, join(args.odir, 'unduped.pkl'))
    joblib.dump(duped, join(args.odir, 'duped.pkl'))
    return unduped, duped

def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8 # Earth radius in kilometers
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))
 
    return R * c

def within(coords):
    if coords[0] is None or coords[1] is None:
	return False
    if spider == "dmoz":
	return coords[0] < 40.796126
    # que
    elif spider == "que":
	km = haversine(40.743924, -73.912388, float(coords[0]), float(coords[1]))
	return km < 4
    # sf
    elif spider == "sfc":
	km = haversine(37.779076, -122.397501, coords[0], coords[1])
	return kms < 1.5
    # berkeley
    elif spider == "eby":
	km = haversine(37.871454, -122.298115, coords[0], coords[1])
	return km < 3
    # milbrae
    #    km = haversine(37.600122, -122.386914, coords[0], coords[1])
    # daly city
    #    km2 = haversine(37.687915, -122.472452, coords[0], coords[1]))
    return True


def qPronouns(vOfv):
    pronouns = re.compile("^(i|me|mine|our|he|she|they|their|we|my|his|her|myself|himself|herself|themselves)$", re.IGNORECASE)
    for w in [w for sent in vOfv for w in sent]:
	if pronouns.search(w):
	    return True
    return False


def nPara(raw):
    lines = raw.split('\n')
    wstart = next( (i for (i,l) in enumerate(lines) if re.search(r"\S", l) ), 0)
    wend = len(lines) - next( (i for (i,l) in enumerate(reversed(lines)) if re.search(r"\S", l) ), 0)
    result = 0
    inPara = False
    for l in lines[wstart:wend]:
        if re.search(r"\S", l):
            if not inPara:
                inPara = True
                result += 1
        elif inPara:
            inPara = False
    return result
    
def numSents(vOfv):
    return len(vOfv)

def numRecurs(vOfv):
    return sum([1 for v in vOfv if len(v) > 13])

def numYell(vOfv):
    return sum([1 for v in vOfv for w in v if re.search("[A-Z]{3}", w) and enchant.Dict("en_US").check(w)])

def numWords(vOfv):
    return sum([len(v) for v in vOfv])

def numGraphs(vOfv):
    return sum([1 for v in vOfv for w in v if re.match(r'([^0-9a-zA-Z])\1\1\1', w)])
    
def numNonAscii(vOfv):
    return sum([1 for v in vOfv for w in v if any(ord(char) > 127 and ord(char) != 8226 for char in w)])

parser = argparse.ArgumentParser()
parser.add_argument('--redis-host', default='localhost')
parser.add_argument("odir", type=argparse_dirtype, help="required json directory")
args = parser.parse_args()
args.odir = args.odir.rstrip("/")
tla = ['abo', 'sub', 'apa', 'cto']
spider = os.path.basename(os.path.realpath(args.odir))
wdir = os.path.dirname(os.path.realpath(__file__))

dt_marker1 = dateutil.parser.parse(os.path.basename(os.path.realpath(join(args.odir, 'marker1')).split(".")[1][::-1].replace("-", ":", 2)[::-1]).replace(tzinfo=utc)
payfor = 9
jsons = determine_payfor_fencepost(dt_marker1, payfor)
craigcr = Json100CorpusReader(args.odir, sorted(jsons), dedupe="id")
coords = list(craigcr.coords())
links = list(craigcr.field('link'))
titles = list(craigcr.field('title'))
ids = list(craigcr.field('id'))
posted = [dateutil.parser.parse(t) for t in craigcr.field('posted')]
bedrooms = []

grid_svm = joblib.load(join(wdir, 'best.pkl'), mmap_mode='r')
unduped, duped = CorpusDedupe(craigcr)
for i, z in enumerate(zip(craigcr.attrs_matching(r'[0-9][bB][rR]'), titles, craigcr.raw())):
    if z[0] is not None:
        bedrooms.append(int(re.findall(r"[0-9]", z[0])[0]))
    else:
        m = re.search(r'(1|one|2|two|3|three|4|four).{0,9}(b[rd]\b|bed)', z[1] + z[2], re.IGNORECASE)
        if m:
            if re.search(m.group(1), "one", re.IGNORECASE):
                bedrooms.append(1)
            elif re.search(m.group(1), "two", re.IGNORECASE):
                bedrooms.append(2)
            elif re.search(m.group(1), "three", re.IGNORECASE):
                bedrooms.append(3)
            elif re.search(m.group(1), "four", re.IGNORECASE):
                bedrooms.append(4)
            else:
                bedrooms.append(int(m.group(1)))
        else:
            bedrooms.append(0)

firstnames = set()
with open(join(wdir, 'firstnames'), 'r') as f:
    for name in f.readlines():
        name = re.sub('\n', '', name)
        firstnames.add(name)

listedby = []
re_suspect = r"deal|no fee|contact|apartments|apts|for all|llc|to view|([^a-zA-Z0-9]|x)\1\1|rentals|real|estate"
for lister in [re.split(r':\s*', i, 1).pop() if i else None for i in craigcr.attrs_matching(r'[lL]isted')]:
    if lister is not None and not re.search(re_suspect, lister):
        tokens = re.split(r'[^0-9a-zA-Z-]+', lister)
        if len(tokens) >= 2 and len(tokens) < 8:
            for i,z in list(enumerate(tokens))[:-1]:
                if z.lower().split('-')[0] in firstnames:
                    lister = "{} {}".format(tokens[i], tokens[i+1])
                    break
    listedby.append(lister)

oklistedby = set()
for pair in Counter(listedby).iteritems():
    if pair[1] == 1:
	if not re.search(re_suspect, pair[0], re.IGNORECASE):
	    oklistedby.add(pair[0])

odoms = craigcr.attrs_matching(r'[oO]dom')
odoms = [re.split(r':\s*', i, 1).pop() if i else None for i in odoms]

try:
    shutil.rmtree(join(args.odir, 'files'))
except OSError as e:
    if e.errno != errno.ENOENT:
        raise
try:
    os.makedirs(join(args.odir, 'files'))
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

scores = grid_svm.decision_function(list(itertools.chain(craigcr)))
filtered = []
with open(join(args.odir, 'digest'), 'w+') as good, open(join(args.odir, 'reject'), 'w+') as bad:
    for i,z in enumerate(zip(craigcr.docs(), craigcr.raw(newlines_are_periods=True))):
        listing = '%s %s %s' % ((' '.join([word for sent in z[0] for word in sent][0:50]),                                     links[i], re.sub(r'\s+', ' ', listedby[i]) if listedby[i] else "Actual Person?"))

        # filter in order of increasing time complexity
        if scores[i] < -0.5:
            bad.write(("spam %s" % listing).encode('utf-8') + '\n\n')
            continue
        if ids[i] in duped:
            bad.write(("dupe %s" % listing).encode('utf-8') + '\n\n')
            continue
        if (dt_marker1 - posted[i]).days >= payfor:
            bad.write(("payfor %s" % listing).encode('utf-8') + '\n\n')
            continue
        if listedby[i] is not None and listedby[i] not in oklistedby:
            bad.write(("listedby %s" % listing).encode('utf-8') + '\n\n')
            continue
        if odoms[i] and int(odoms[i]) > 160000:
            bad.write(("miles %s" % listing).encode('utf-8') + '\n\n')
            continue
        if not within(coords[i]):
            bad.write(("toofar %s" % listing).encode('utf-8') + '\n\n')
            continue
        if not listedby[i] and not qPronouns(z[0]):
            bad.write(("pronouns %s" % listing).encode('utf-8') + '\n\n')
            continue

#        if re.search(r'leasebreak', z[1]):
#            bad.write(("leasebreak %s" % listing).encode('utf-8') + '\n\n')
#            continue

        nw=numWords(z[0])
        ns=numSents(z[0])
        ng=numGraphs(z[0])
        wps=float(nw/ns) if ns else 0.0
        nr=numRecurs(z[0])
        np=nPara(z[1])
        spp=float(len(z[0])/np) if np else 0.0
        ny = numYell(z[0])
        yr=float(ny/nw) if nw else 0.0
        nna=numNonAscii(z[0])

        if nna > 3 or spp <= 1.0 or yr > 0.1 or ny > 20 or ng > 3:
            bad.write(("garbage %s" % listing).encode('utf-8') + '\n\n')
            continue
        good.write(listing.encode('utf-8') + '\n\n')
        filtered.append(i)

        tla_link = re.findall(r"({0})/(?:[^/]+/)*?(\d+).html".format('|'.join(tla)), links[i])[-1]
        with open(join(args.odir, "files", "{}-{}".format(tla_link[0], tla_link[1])), "w") as f:
            f.write(z[1].encode('utf-8'))

red = redis.StrictRedis(host=args.redis_host, port=6379, db=0)
prices = list(craigcr.numbers(['price']))
for i in sorted(filtered):
    if prices[i]['price'] is not None:
        red.hset('item.' + ids[i], 'price', prices[i]['price'])
        red.zadd('item.index.price', prices[i]['price'], ids[i])
    red.hmset('item.' + ids[i], {'link': links[i], 'title': titles[i], 'bedrooms': bedrooms[i], 'coords': coords[i], 'posted': posted[i].isoformat() })
    red.zadd('item.index.bedrooms', bedrooms[i], ids[i])
    if None not in coords[i]:
        red.geoadd('item.geohash.coords', *(tuple(reversed(coords[i])) + (ids[i],)))
    red.hset('item.' + ids[i], 'score', scores[i])
    red.zadd('item.index.score', scores[i], ids[i])
