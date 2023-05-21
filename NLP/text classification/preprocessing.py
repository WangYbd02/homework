"""
@ 预处理
"""


import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


def get_text(dirname):
    """获取文本"""
    with open(dirname, 'r', encoding='utf-8', errors='ignore') as f:
        flag = 0
        text = ''
        while True:
            line = f.readline()
            if line == '':
                break
            if flag == 0:
                # if line == '\n':
                if 'Lines:' in line:  # 去除文本的标注信息(观察发现大多数文本标注信息都在包含‘Lines:’的行之前)
                    flag = 1
                    continue
            if flag == 1:
                text = text + line
        f.close()
    return text


def getStopWords():
    """
    获得停用词表和字母a~z在停用词表中的位置
    因为stopwords.txt文件中的停用词是按照首字母排序的，得知字母a~z在停用词表中的位置可以加快判断一个词是否在停用词表中的判断速度
    """
    stopWordsList = []  # 停用词列表
    with open('./stopwords_modified.txt', 'r') as f:
        while True:
            line = f.readline()
            word = line.split('\n')[0]
            if word == '':
                break
            stopWordsList.append(word)
        f.close()
    letter_loca = []  # 字母a~z在停用词表中的位置
    for indx in range(len(stopWordsList)):
        if len(stopWordsList[indx]) == 1:
            letter_loca.append(indx)
    letter_loca.append(len(stopWordsList))
    return stopWordsList, letter_loca

def getPunctuations():
    """获取数字标点符号表"""
    punctuationsList = []  # 数字标点符号表
    with open('./punctuations.txt', 'r') as f:
        while True:
            line = f.readline()
            punc = line.split('\n')[0]
            if punc == '':
                break
            punctuationsList.append(punc)
        f.close()
    return punctuationsList

def preprocess(dirname):
    """文本预处理"""
    """--------------------------------------------------------------------------"""
    """1.获取文本、数字标点符号表、停用词表、停用词表中的字母位置的索引列表"""
    text = get_text(dirname)  # 从给定路径中读取文本
    punctuationsList = getPunctuations()  # 获取数字标点符号表
    stopWordsList, letter_loca = getStopWords()  # 获取停用词表和停用词表中的字母位置的索引列表
    """--------------------------------------------------------------------------"""
    """2.正则化，将标点符号、数字、换行符、制表符替换为空格，所有字母变为小写，其它符号也替换为空格"""
    text1 = ""
    for ele in text:
        if ele == '\n' or ele == '\t':  # 换行符和制表符替换为空格
            text1 = text1 + " "
        elif ele not in punctuationsList:
            if (ord(ele) >= 97 and ord(ele) <= 122) or (ord(ele) >= 65 and ord(ele) <= 90):  # 字母保留
                text1 = text1 + ele
            else:  # 其它符号替换为空格
                text1 = text1 + " "
        else:  # 数字和标点符号替换为空格
            text1 = text1 + " "
    text1 = text1.lower() # 将所有字母变为小写
    """--------------------------------------------------------------------------"""
    """3.删除停用词"""
    text2 = text1.split(' ')  # 提取出单词
    text3 = []
    for ele in text2:
        if ele == '':
            continue
        else:
            init = ele[0]  # 提出单词的首字母
            loca = ord(init)-97
            # start～end是以init为首字母的所有停用词的索引范围
            start = letter_loca[loca]
            end = letter_loca[loca+1]
            if ele not in stopWordsList[start:end]:  # 删除停用词
                text3.append(ele)
    """--------------------------------------------------------------------------"""
    """4.词形还原"""
    lemmatizer = WordNetLemmatizer()
    # stemmer = PorterStemmer()  # 如果没有WordNet可以用词干提取
    text4 = []
    for word in text3:
        text4.append(lemmatizer.lemmatize(word))  # 词形还原
        # text4.append(stemmer.stem(word))
    """--------------------------------------------------------------------------"""
    """5.将预处理后的单词利用空格拼接成文本"""
    text5 = ''
    for word in text4:
        if len(word) > 1:  # 经过词形还原会出现一些字母，比如'cs'就会变成'c'
            text5 = text5 + word + ' '
    # if text5 == '':  # 可以输出处理完后为空字符串的文本的绝对路径，更多信息见classify.py第27行
    #     print(dirname)

    return text5
