#!/usr/bin/python3

# Copyright 2020 Eric Munson

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

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