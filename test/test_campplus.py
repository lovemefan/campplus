# -*- coding:utf-8 -*-
# @FileName  :test_campplus.py
# @Time      :2023/6/26 12:08
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
from campplus.src.Campplus import Campplus


def campplus_text():
    audio_file1 = 'a_cn_16k.wav'
    audio_file2 = 'b_cn_16k.wav'

    # threshold 越高区分度越高，返回的说话人身份越多
    model = Campplus(threshold=0.7)
    index = model.recognize(audio_file1)
    print(index)
    index = model.recognize(audio_file1)
    print(index)
    index = model.recognize(audio_file2)
    print(index)
    index = model.recognize(audio_file2)
    print(index)


if __name__ == '__main__':
    campplus_text()