from __future__ import print_function
import os
import subprocess
import shlex
import boto3

s3 = boto3.resource('s3')


class S3(object):
    def __init__(self, bucket):
        self.s3 = boto3.resource('s3')
        self.bucket = s3.Bucket(bucket)
        for obj in self.bucket.objects.all():
            print(obj.key)

    def __enter__(self):
        sync_down()

    def __exit__(self):
        sync_up()

    def sync_down(self):
        bucket = self.bucket.split('/')[2]
        key = '/'.join(self.bucket.split('/')[3:])
        s3.Object(bucket, key).download_file(self.file)

    def sync_up(self):
        bucket = self.bucket.split('/')[2]
        key = '/'.join(self.bucket.split('/')[3:])
        s3.Object(bucket, key).upload_file(self.file, ExtraArgs={'ServerSideEncryption': 'AES256'})
