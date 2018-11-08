from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
from sklearn import metrics

from Step1_Segment import *
from Step2_ToBunch import *
from Step3_TFIDFSpace import *

def predict(line):
    segment_Line