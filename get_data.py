import os, json, dateutil.parser
from os.path import join
from pytz import utc
import pickle
from datetime import datetime, timedelta

def datetime_parser(json_dict):
    for k,v in json_dict.iteritems():
        try:
            json_dict[k] = dateutil.parser.parse(v)
        except (ValueError, AttributeError):
            pass
    return json_dict

def get_datetime(filename):
    return dateutil.parser.parse(filename.split(".")[1][::-1].replace("-", ":", 2)[::-1]).replace(tzinfo=utc)

def download_s3(s3_client, bucket, dir=".", payfor=9):
    jsons = []
    start_after = 'Marker.{}.json'.format((datetime.now() - timedelta(days=payfor)).replace(microsecond=0).isoformat().replace(":", "-"))
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix="Marker.", StartAfter=start_after)

    Markers = []
    for content in response['Contents']:
        if 'Key' in content:
            Markers.append(content['Key'])
        else:
            print("download_s3: bucket={}, dir={}, start_after={}, ignoring content={}".format(bucket, dir, start_after, str(content)))
    Markers = sorted(Markers, reverse=True)

    latest = get_datetime(Markers[0])
    for m in Markers:
        if not os.path.exists(m):
            response = s3_client.get_object(Bucket=bucket, Key=m)
            with open(join(dir, m), 'w') as fp:
                fp.write(response['Body'].read())

        within = payfor == 0
        if not within:
            with open(join(dir, m), 'r') as fp:
                try:
                    url2dt = json.load(fp, object_hook=datetime_parser)
                    for url,dt0 in url2dt.iteritems():
                        if (latest - dt0).days < payfor:
                            within = True
                            break
                except ValueError:
                    fp.seek(0)
                    dates = pickle.load(fp)
                    if any((latest.date() - dt).days < payfor for dt in dates):
                        within = True

        if within:
            body = "Data.{}".format(m.split(".", 1)[1])
            jsons.append(body) # reader.py readds the parent dir
            if not os.path.exists(join(dir, body)):
                response = s3_client.get_object(Bucket=bucket, Key=body)
                with open(join(dir, body), 'w') as fp:
                    fp.write(response['Body'].read())
        else:
            break
    return jsons, latest
