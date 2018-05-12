#!/bin/bash -ex

declare -a spiders
while [[ $# -gt 0 ]] ; do
  key="$1"
  case "$key" in
      -o)
      ONCE=1
      shift
      ;;
      -s)
      spiders+=("$2")
      shift
      shift
      ;;
      *)
      break
      ;;    
  esac
done
if [[ "${spiders[@]}" == "" ]]; then
    spiders=("newyork" "listingsproject" "sfbay")
fi
WDIR=$(dirname $0)
PROJECT=tutorial
ITEMDIR=/var/lib/scrapyd/items/${PROJECT}

declare -A once
function once_yet {
    for spider in "${spiders[@]}" ; do
        if ! test "${once[$spider]+isset}"; then
            return 1
        fi
    done    
    return 0
}

while [ 1 ] ; do
#    for spider in $(scrapyd-client -t http://scrapyd:6800 spiders -p ${PROJECT} | tail -n +2) ; do
# FIXME
    for spider in "${spiders[@]}" ; do
        if [ -d ${ITEMDIR}/$spider ]; then
           if ( [ ! -e ${ITEMDIR}/${spider}/digest ] && [ ! -z "$(find -L ${ITEMDIR}/$spider -name 'Marker.*\.json')" ] ) || ( [ -e ${ITEMDIR}/${spider}/digest ] && [[ ! -z $(find -L ${ITEMDIR}/${spider} -name 'Marker.*\.json' -cnewer ${ITEMDIR}/${spider}/digest) ]] ); then
               options=""
               database=0
               if [ $spider == "listingsproject" ]; then
                   options="${options} --revisionist --payfor 28"
               elif [ "${GIT_BRANCH:-}" == "dev" ]; then
                   options="${options} --payfor 2"
               fi
               case "$spider" in
                   sfbay)
                       database=1
                       ;;
               esac
               python ${WDIR}/dedupe.py --redis-host redis --corenlp-uri http://corenlp:9005 --redis-database ${database} ${ITEMDIR}/${spider}$options
               once+=([$spider]=1)
           fi
        fi
    done
    if [ ! -z "${ONCE:-}" ] && once_yet ; then
      break
    fi
    sleep 30
done
