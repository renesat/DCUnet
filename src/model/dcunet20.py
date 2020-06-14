#!/usr/bin/env python3

from .dcunet import DCUnet


class DCUnet20(DCUnet):
    def __init__(self, iscomplex: bool = False):
        super(DCUnet20, self).__init__([
            {
                'kernel': (7, 1),
                'stride': (1, 1),
                'chanels': 32 if iscomplex else 45
            },
            {
                'kernel': (1, 7),
                'stride': (1, 1),
                'chanels': 32 if iscomplex else 45
            },
            {
                'kernel': (7, 5),
                'stride': (2, 2),
                'chanels': 64 if iscomplex else 90
            },
            {
                'kernel': (7, 5),
                'stride': (2, 1),
                'chanels': 64 if iscomplex else 90
            },
            {
                'kernel': (5, 3),
                'stride': (2, 2),
                'chanels': 64 if iscomplex else 90
            },
            {
                'kernel': (5, 3),
                'stride': (2, 1),
                'chanels': 64 if iscomplex else 90
            },
            {
                'kernel': (5, 3),
                'stride': (2, 2),
                'chanels': 64 if iscomplex else 90
            },
            {
                'kernel': (5, 3),
                'stride': (2, 1),
                'chanels': 64 if iscomplex else 90
            },
            {
                'kernel': (5, 3),
                'stride': (2, 2),
                'chanels': 64 if iscomplex else 90
            },
            {
                'kernel': (5, 3),
                'stride': (2, 1),
                'chanels': 90 if iscomplex else 180
            },
            {
                'kernel': (5, 3),
                'stride': (2, 1),
                'chanels': 64 if iscomplex else 90
            },
            {
                'kernel': (5, 3),
                'stride': (2, 2),
                'chanels': 64 if iscomplex else 90
            },
            {
                'kernel': (5, 3),
                'stride': (2, 1),
                'chanels': 64 if iscomplex else 90
            },
            {
                'kernel': (5, 3),
                'stride': (2, 2),
                'chanels': 64 if iscomplex else 90
            },
            {
                'kernel': (5, 3),
                'stride': (2, 1),
                'chanels': 64 if iscomplex else 90
            },
            {
                'kernel': (5, 3),
                'stride': (2, 2),
                'chanels': 64 if iscomplex else 90
            },
            {
                'kernel': (7, 5),
                'stride': (2, 1),
                'chanels': 64 if iscomplex else 90
            },
            {
                'kernel': (7, 5),
                'stride': (2, 2),
                'chanels': 32 if iscomplex else 45
            },
            {
                'kernel': (1, 7),
                'stride': (1, 1),
                'chanels': 32 if iscomplex else 45
            },
            {
                'kernel': (7, 1),
                'stride': (1, 1),
                'chanels': 1
            },
        ],
                                       iscomplex=iscomplex)
