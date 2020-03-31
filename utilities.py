#!/usr/bin/python3

import chardet


def detect_encoding(filepath):
    result={}
    with open(filepath, 'rb') as rawdata:
        msg = rawdata.read(50000)
        result = chardet.detect(msg)
        rawdata.flush()

    if result==None:
        result={}
    if result['encoding']==None:
        return result

    if 'encoding' not in result.keys():
        result['encoding']='utf-8-sig'
        result['confidence']=0

    # https://github.com/buriy/python-readability/issues/42
    if result['encoding'].lower()=='big5':
        result['encoding']='big5hkscs'
    elif result['encoding'].lower()=='gb2312':
        result['encoding']='gb18030'
    elif result['encoding'].lower()=='ascii':
        result['encoding']='utf-8'
    elif result['encoding'].lower()=='iso-8859-1':
        result['encoding']='cp1252'

    return result


def load_target_words(target_path):
    target_words = []
    with open(target_path, 'r') as fd:
        for line in fd:
            target_words.extend([entry for entry in line.split() if entry and not entry.isspace()])
    return target_words