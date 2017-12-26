#!/bin/bash -e

WDIR=$(dirname $0)
PROJECT=tutorial
ITEMDIR=/var/lib/scrapyd/items/${PROJECT}

while [ 1 ] ; do
    for spider in $(scrapyd-client -t http://scrapyd:6800 spiders -p ${PROJECT} | tail -n +2) ; do
        if [ -d ${ITEMDIR}/$spider ]; then
           if ( [ ! -e ${ITEMDIR}/${spider}/digest ] && [ ! -z "$(find -L ${ITEMDIR}/$spider -name 'Marker.*\.json')" ] ) || ( [ -e ${ITEMDIR}/${spider}/digest ] && [[ ! -z $(find -L ${ITEMDIR}/${spider} -name 'Marker.*\.json' -cnewer ${ITEMDIR}/${spider}/digest) ]] ); then
               ${WDIR}/dedupe.py --redis-host redis ${ITEMDIR}/${spider}
           fi
        fi
    done
    sleep 30
done
