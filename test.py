# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/4
@Time    : 16:31
@File    : test.py
@Function: XX
@Other: XX
"""
import json
import requests
from tqdm import tqdm
sess_web = requests.Session()


def server_test(sen):
    # noinspection PyBroadException
    try:
        results = sess_web.post(url=url, data=sen.encode("utf-8")).text
    except Exception as e:
        results = str(e)
    return results


if __name__ == '__main__':
    url = 'http://10.17.107.66:10089/prediction'
    # text = '日用化学品混合分装生产项目'
    with open('财务附注测试用例.txt', 'r', encoding='utf-8') as f:
        texts = f.readlines()
    results = server_test(json.dumps(texts, ensure_ascii=False))
    print(results)
    # results = server_test(json.dumps([''], ensure_ascii=False))

