#!/usr/bin/env python

import argparse
import os
import shutil
from enum import Enum

import requests

from .utils import load_cofig


class DataTypes(Enum):
    http = 1


class DataWorker():
    def __init__(self, link, dtype, path):

        self.link = link
        self.type = dtype
        self.path = path

    def clean(self):
        for filename in os.listdir(self.path):
            file_path = os.path.join(self.path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete {}. Reason: {}'.format(file_path, e))

    def download(self):
        return {
            DataTypes.http: self.__download_http,
        }[self.type]()

    def __download_http(self):
        os.makedirs(self.path, exist_ok=True)
        local_filename = os.path.join(self.path, self.link.split('/')[-1])
        if os.path.isfile(local_filename):
            print('File {} exist'.format(local_filename))
            return local_filename
        with requests.get(self.link, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='command',
                                       help='sub-command help',
                                       dest='command',
                                       required=True)

    parser_clean = subparsers.add_parser('clean')
    parser_clean.set_defaults(run=lambda worker: worker.clean())

    parser_download = subparsers.add_parser('download')
    parser_download.set_defaults(run=lambda worker: worker.download())

    for name, subp in subparsers.choices.items():
        subp.add_argument('-c', '--config', default=None, help='Config file')

    args = parser.parse_args()

    config = load_cofig()
    for name, v in config['data'].items():
        print('Downloading {}...'.format(name))
        args.run(DataWorker(v['link'], DataTypes[v['type']], v['out-path']))
