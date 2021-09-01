#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Manage logging across the experiments
"""
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class Log:
    @classmethod
    def info(cls, phase, source, msg):
        logging.info(f'*** {phase}: {source} - {msg}')

    @classmethod
    def debug(cls, phase, source, msg):
        logging.info(f'*** {phase}: {source} - {msg}')

    @classmethod
    def warning(cls, phase, source, msg):
        logging.info(f'*** {phase}: {source} - {msg}')

    @classmethod
    def error(cls, phase, source, msg):
        logging.info(f'*** {phase}: {source} - {msg}')