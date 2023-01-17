#!/usr/bin/env python

import argparse
import os
import shutil
import tarfile
import zipfile
from enum import Enum

import requests
from tqdm import tqdm

from .utils import load_cofig


class DataTypes(Enum):
    http = 1


class DataWorker():
    def __init__(self,
                 link,
                 dtype,
                 path,
                 file_type=None,
                 raw_path='data/raw',
                 cleaned_path='data/cleaned'):
        if file_type is None:
            file_type = 'clean',

        self.link = link
        self.type = dtype
        self.path = path
        self.file_type = file_type
        self.raw_path = raw_path
        self.cleaned_path = cleaned_path

    def remove(self):
        out_dir = os.path.join(self.raw_path, self.path)
        for filename in os.listdir(out_dir):
            file_path = os.path.join(out_dir, filename)
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

    def cleaned(self):
        return {
            'cleaned': self.__link_cleaner,
            'zip': self.__zip_cleaner,
            'tar': self.__tar_cleaner,
        }[self.file_type]()

    def __link_cleaner(self):
        in_dir = os.path.join(self.raw_path, self.path)
        in_filename = os.path.abspath(
            os.path.join(in_dir,
                         self.link.split('/')[-1]))

        out_dir = os.path.join(self.cleaned_path, self.path)
        out_filename = os.path.abspath(
            os.path.join(out_dir,
                         self.link.split('/')[-1]))

        os.symlink(in_filename, out_filename)

    def __tar_cleaner(self):
        in_dir = os.path.join(self.raw_path, self.path)
        in_filename = os.path.abspath(
            os.path.join(in_dir,
                         self.link.split('/')[-1]))
        out_dir = os.path.join(self.cleaned_path, self.path)

        with tarfile.open(in_filename) as tar_ref:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_ref, out_dir)

    def __zip_cleaner(self):
        in_dir = os.path.join(self.raw_path, self.path)
        in_filename = os.path.abspath(
            os.path.join(in_dir,
                         self.link.split('/')[-1]))
        out_dir = os.path.join(self.cleaned_path, self.path)

        with zipfile.ZipFile(in_filename, 'r') as zip_ref:
            zip_ref.extractall(out_dir)

    def __download_http(self):
        out_dir = os.path.join(self.raw_path, self.path)
        os.makedirs(out_dir, exist_ok=True)

        local_filename = os.path.join(out_dir, self.link.split('/')[-1])
        if os.path.isfile(local_filename):
            print('File {} exist'.format(local_filename))
            return local_filename
        with requests.get(self.link, stream=True) as r:
            r.raise_for_status()
            pbar = tqdm(total=int(r.headers['Content-Length']),
                        initial=0,
                        unit='B',
                        unit_scale=True,
                        desc=self.link.split('/')[-1])
            chunk_size = 8192
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)
            pbar.close()
        return local_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='command',
                                       help='sub-command help',
                                       dest='command',
                                       required=True)

    parser_remove = subparsers.add_parser('remove')
    parser_remove.set_defaults(run=lambda worker: worker.remove(),
                               run_name="Removing")

    parser_download = subparsers.add_parser('download')
    parser_download.set_defaults(run=lambda worker: worker.download(),
                                 run_name="Downloading")
    parser_download.add_argument('--clean', default=None, help='Config file')

    parser_clean = subparsers.add_parser('clean-data')
    parser_clean.set_defaults(run=lambda worker: worker.cleaned(),
                              run_name="Cleaning")

    for name, subp in subparsers.choices.items():
        subp.add_argument('-c', '--config', default=None, help='Config file')

    args = parser.parse_args()

    config = load_cofig()
    for name, v in config['data'].items():
        print('{} {}...'.format(args.run_name, name))
        args.run(
            DataWorker(v['link'], DataTypes[v['type']], v['out-path'],
                       v.get('file-type', None)))
