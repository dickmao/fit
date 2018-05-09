#!/usr/bin/python

from __future__ import division
import os, re, redis, sys
import numpy as np
from corenlp import CoreNLPClient, TimeoutException
import itertools
import enchant
from gensim import corpora, models
from gensim.similarities.docsim import SparseMatrixSimilarity
from reader import Json100CorpusReader
from collections import Counter
from tempfile import mkstemp
import botocore.session
from get_data import download_s3

from math import radians, sin, cos, sqrt, asin

from os.path import join
import dateutil.parser
from datetime import datetime, timedelta
from time import time
import argparse

from sklearn.externals import joblib

vernum = 'ad0001'

def fill_year(b, posted):
    cands = [b.replace(year=posted.year-1), b.replace(year=posted.year), b.replace(year=posted.year+1)]
    return min(cands, key=lambda x:abs(x - posted))

def days_of(timex):
    mo = re.search(r'(\d+)([DWMY])', timex)
    if mo:
        num = int(mo.group(1))
        unit = mo.group(2)
        if unit == "D":
            return num
        elif unit == "W":
            return num * 7
        elif unit == "M":
            return num * 30
        elif unit == "Y":
            return num * 365
    return 0

def dtOfString(md, fmt, posted):
    try:
        b = datetime.strptime(md, fmt)
    except ValueError as e:
        if str(e).startswith("time"):
            md = md.replace("X", str(1))
            try:
                b = datetime.strptime(md, fmt)
            except ValueError as e:
                print md, fmt, posted
                raise e
        else:
            print md, fmt, posted
            raise e
    b = b.replace(tzinfo=posted.tzinfo)
    if abs(b - posted).days > 360:
        b = fill_year(b, posted)
    return b

def argparse_dirtype(astring):
    if not os.path.isdir(astring):
        raise argparse.ArgumentError
    return astring

def revisionist(cr, ids):
    dictionary = corpora.Dictionary()
    corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in list(cr)]
    tfidf = models.TfidfModel(corpus)
    tempf  = mkstemp()[1]
    corpora.MmCorpus.serialize(tempf, tfidf[corpus], id2word=dictionary.id2token)
    mmcorpus = corpora.MmCorpus(tempf)
    # SparseMatrixSimilarity[query]
    sim = SparseMatrixSimilarity(mmcorpus)[mmcorpus[range(len(ids))]]

    t0 = time()
    # ConcatenatedCorpusView cannot seem to random access; must iterate sequentially lest
    # block reader get ahead of itself
    assert(len(mmcorpus) == len(ids))
    latest = set()
    for i,z in enumerate(ids):
        # where-clause literals are n-wide boolean arrays
        clique = np.where(sim[i] > 0.61)[0]
        latest.add(ids[min(clique)])
    print("n*(n-1)/2 took %0.3fs" % (time() - t0))
    return latest

def CorpusDedupe(odir, jsons, exclude):
    cr = Json100CorpusReader(odir, jsons, dedupe="id", exclude=exclude)
    ids = list(cr.field('id'))
    # dict.doc2bow makes:
    #   corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
    #             [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
    #             [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],      ]
    try:
        unduped = joblib.load(join(args.odir, 'unduped.pkl'), mmap_mode='r')
        duped = joblib.load(join(args.odir, 'duped.pkl'), mmap_mode='r')
    except IOError:
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
    # SparseMatrixSimilarity[query]
    new_sim = SparseMatrixSimilarity(mmcorpus)[mmcorpus[new_indices]]

    t0 = time()
    # ConcatenatedCorpusView cannot seem to random access; must iterate sequentially lest
    # block reader get ahead of itself
    assert(len(mmcorpus) == len(ids))
    for i,z in enumerate(zip(new_ids, new_indices)):
        # where-clause literals are n-wide boolean arrays
        for dj in np.where((new_sim[i] > 0.61) & [j!=z[1] for j in range(len(ids))])[0]:
            duped.update([z[0], ids[dj]])
    print("(n-1) + ... (n-k) = k(n - (k+1)/2) took %0.3fs" % (time() - t0))

    unduped.update(new_ids - duped)
    joblib.dump(unduped, join(args.odir, 'unduped.pkl'))
    joblib.dump(duped, join(args.odir, 'duped.pkl'))
    return unduped, duped

def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8  # Earth radius in kilometers
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
        return km < 1.5
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

# span = begin_end(client, available[i], posted[i], descs[i])
def begin_end(client, available, posted, doc):
    try:
        ann = client.annotate(doc, annotators="ner".split())
    except TimeoutException:
        print "TimeoutException: ", doc.encode('utf-8')
        return None, None
    left, right = available, None
    durs = [m.timex.value for m in itertools.chain(*[mgroup for mgroup in [s.mentions for s in ann.sentence] ]) if m.ner == "DURATION" and re.match(r'P\d+[DWMY]', m.timex.value)]
    if left and durs:
        right = left + timedelta(days=max([days_of(x) for x in durs]))
    dates = [m.timex.value.split('T')[0] for m in itertools.chain(*[mgroup for mgroup in [s.mentions for s in ann.sentence] ]) if m.ner == 'DATE' and m.timex.value and re.match(r'[X0-9]{4}-[X0-9]{2}-', m.timex.value) ]
    if dates:
        dates = [dtOfString(d, "%Y-%m-%d", posted) for d in dates]
        if not available:
            left = dates[0]
        for i, d in enumerate(dates):
            if abs((d - left).days) < 6 and i+1 < len(dates):
                right = dates[i+1]
                break
        if not right:
            rights = [d for d in dates if d >= left]
            if rights:
                right = rights[0]
    return left, right

parser = argparse.ArgumentParser()
parser.add_argument('--redis-host', type=str, default='localhost')
parser.add_argument('--redis-database', type=int, default=0)
parser.add_argument('--corenlp-uri', type=str, default='http://localhost:9005')
parser.add_argument('--payfor', type=int, default=9)
parser.add_argument('--revisionist', dest='revisionist', action='store_true')
parser.set_defaults(revisionist=False)
parser.add_argument("odir", type=argparse_dirtype, help="required json directory")

args = parser.parse_args()
args.odir = args.odir.rstrip("/")
spider = os.path.basename(os.path.realpath(args.odir))
srcdir = os.path.dirname(os.path.realpath(__file__))
bucket = "303634175659.{}".format(spider)
s3_client = botocore.session.get_session().create_client('s3')

jsons, latest = download_s3(s3_client, bucket, args.odir, args.payfor)
exclude = set()
if args.revisionist: # for noncraig sources
    allcr = Json100CorpusReader(args.odir, jsons, dedupe="id")
    ids = list(allcr.field('id'))
    exclude = set(ids) - revisionist(allcr, ids)
unduped, duped = CorpusDedupe(args.odir, jsons, exclude)
try:
    last_json_read = joblib.load(join(args.odir, 'last-json-read.pkl'), mmap_mode='r')
except IOError:
    last_json_read = ""
fns = [fn for fn in jsons if fn > last_json_read]
if not len(fns):
    print "No Data files beyond %s".format(last_json_read)
    sys.exit(0)

craigcr = Json100CorpusReader(args.odir, fns, dedupe="id", exclude=exclude)
coords = list(craigcr.coords())
links = list(craigcr.field('link'))
titles = list(craigcr.field('title'))
ids = list(craigcr.field('id'))
posted = [dateutil.parser.parse(t) for t in craigcr.field('posted')]
bedrooms = []

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
with open(join(srcdir, 'firstnames'), 'r') as f:
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

with CoreNLPClient(start_cmd="gradle -p {} server".format("../CoreNLP"), endpoint=args.corenlp_uri, timeout=15000) as client:
    response = s3_client.get_object(Bucket=bucket, Key="{}.pkl".format(vernum))
    with open(join(args.odir, 'svc.pkl'), 'w') as fp:
        fp.write(response['Body'].read())
    svc = joblib.load(join(args.odir, 'svc.pkl'), mmap_mode='r')
    # tried to do a hasattr thing ...
    # for tup in svc.named_steps['featureunion'].transformer_list:
    #     pipeline = tup[1]
    #     for obj in pipeline.named_steps.values():
    #         if type(obj) ==
    pipeline = next(x[1] for x in svc.named_steps['featureunion'].transformer_list \
                    if x[0] == 'text')
    pipeline.named_steps['vectorizer'].analyzer._client = client
    pipeline = next(x[1] for x in svc.named_steps['featureunion'].transformer_list \
                    if x[0] == 'pronouns')
    pipeline.named_steps['count'].func._client = client
    scores = svc.decision_function(list(craigcr.desc()))

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
        if (latest - posted[i]).days >= args.payfor:
            bad.write(("payfor %s" % listing).encode('utf-8') + '\n\n')
            continue
        if listedby[i] is not None and listedby[i] not in oklistedby:
            bad.write(("listedby %s" % listing).encode('utf-8') + '\n\n')
            continue
        if not within(coords[i]):
            bad.write(("toofar %s" % listing).encode('utf-8') + '\n\n')
            continue
        # if not listedby[i] and not qPronouns(z[0]):
        #     bad.write(("pronouns %s" % listing).encode('utf-8') + '\n\n')
        #     continue

        #        if re.search(r'leasebreak', z[1]):
        #            bad.write(("leasebreak %s" % listing).encode('utf-8') + '\n\n')
        #            continue

        nw=numWords(z[0])
        ns=numSents(z[0])
        ng=numGraphs(z[0])
        wps=float(nw/ns) if ns else 0.0
        nr=numRecurs(z[0])
        nump=nPara(z[1])
        spp=float(len(z[0])/nump) if nump else 0.0
        ny = numYell(z[0])
        yr=float(ny/nw) if nw else 0.0
        nna=numNonAscii(z[0])

        if nna > 3 or spp <= 1.0 or yr > 0.1 or ny > 20 or ng > 3:
            bad.write(("garbage %s" % listing).encode('utf-8') + '\n\n')
            continue
        good.write(listing.encode('utf-8') + '\n\n')
        filtered.append(i)

red = redis.StrictRedis(host=args.redis_host, port=6379, db=args.redis_database)
prices = list(craigcr.numbers(['price']))
descs = list(craigcr.field('desc'))
available = [(dtOfString(re.search(r'\S+ \d+', z).group(), "%b %d", posted[i]) if z else None) for i,z in enumerate(craigcr.attrs_matching(r'^[aA]vail'))]
begins = [(dateutil.parser.parse(t) if t else None) for t in craigcr.field('begin')]
ends = [(dateutil.parser.parse(t) if t else None) for t in craigcr.field('end')]
for i in sorted(filtered):
    if prices[i]['price'] is not None:
        red.hset('item.' + ids[i], 'price', prices[i]['price'])
        red.zadd('item.index.price', prices[i]['price'], ids[i])
    red.zadd('item.index.posted.{}'.format(args.payfor), int(posted[i].strftime("%s")), ids[i])
    red.hmset('item.' + ids[i], {'link': links[i], 'title': titles[i], 'desc': descs[i], 'bedrooms': bedrooms[i], 'coords': coords[i], 'posted': posted[i].isoformat()})
    red.zadd('item.index.bedrooms', bedrooms[i], ids[i])
    if None not in coords[i]:
        red.geoadd('item.geohash.coords', *(tuple(reversed(coords[i])) + (ids[i],)))
    red.hset('item.' + ids[i], 'score', scores[i])
    red.zadd('item.index.score', scores[i], ids[i])
    if begins[i] and ends[i]:
        red.hset('item.' + ids[i], 'begin', begins[i].isoformat())
        red.hset('item.' + ids[i], 'end', ends[i].isoformat())
    else:
        span = begin_end(client, available[i], posted[i], descs[i])
        span2 = begin_end(client, available[i], posted[i], titles[i])
        if span[0]:
            red.hset('item.' + ids[i], 'begin', span[0].isoformat())
        if span[1]:
            if span2[1] and span2[1] > span[1]:
                red.hset('item.' + ids[i], 'end', span2[1].isoformat())
            else:
                red.hset('item.' + ids[i], 'end', span[1].isoformat())

joblib.dump(max(jsons), join(args.odir, 'last-json-read.pkl'))
