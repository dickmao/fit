import os, json, dateutil.parser
from os.path import join
from pytz import utc

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
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix="Marker.")
    Markers = sorted([content['Key'] for content in response['Contents']], reverse=True)
    latest = get_datetime(Markers[0])
    for m in Markers:
        within = False
        if not os.path.exists(m):
            response = s3_client.get_object(Bucket=bucket, Key=m)
            with open(join(dir, m), 'w') as fp:
                fp.write(response['Body'].read())

        with open(join(dir, m), 'r') as fp:
            url2dt = json.load(fp, object_hook=datetime_parser)
            for url,dt0 in url2dt.iteritems():
                if (latest - dt0).days < payfor:
                    within = True
                    break
        if within:
            body = "Data.{}".format(m.split(".", 1)[1])
            jsons.append(join(".", body))
            if not os.path.exists(join(".", body)):
                response = s3_client.get_object(Bucket=bucket, Key=body)
                with open(join(dir, body), 'w') as fp:
                    fp.write(response['Body'].read())
        else:
            break
    return jsons, latest

