#!/bin/bash -ex

cd $(dirname $0)
if [ -d ".dedupe" ]; then
  (cd .dedupe ; git pull )  
else
  git clone --depth=1 --single-branch git@github.com:dickmao/fit.git .dedupe
fi

if [ ! -z $(docker ps -aq --filter "name=dedupe") ]; then
  docker rm -f $(docker ps -aq --filter "name=dedupe")
fi

COPY=""
for file in $( cd .dedupe ; git ls-files ) ; do
  dir=$(dirname $file)
  COPY=$(printf "$COPY\nCOPY .dedupe/${file} /${dir}/")
done

cat > ./Dockerfile.tmp <<EOF
FROM python:2.7
MAINTAINER dick <noreply@shunyet.com>
RUN set -xe \
  && apt-get -yq update \
  && DEBIAN_FRONTEND=noninteractive apt-get -yq install libenchant1c2a \
  && apt-get clean \
  && curl -sSL https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh -o ./wait-for-it.sh \
  && chmod u+x ./wait-for-it.sh \
  && pip install redis nltk numpy pytz gensim python_dateutil pyenchant scikit_learn git+https://github.com/scrapy/scrapyd-client.git git+https://github.com/dickmao/python-stanford-corenlp.git \
  && python -m nltk.downloader punkt \
  && rm -rf /var/lib/apt/lists/*
$COPY
EOF

../ecr-build-and-push.sh ./Dockerfile.tmp dedupe:latest

rm ./Dockerfile.tmp
