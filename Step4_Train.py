#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
from sklearn import metrics
from sklearn import svm
from Tools import readbunchobj
from Step1_Segment import segment_Line,Step1_Segment
from Step2_ToBunch import seg2Bunch,Step2_ToBunch
from Step3_TFIDFSpace import bunch2Space,Step3_TFIDFSpace
import numpy as np
def Train():
        Step1_Segment()
        Step2_ToBunch()
        Step3_TFIDFSpace()
        trainpath = "train_word_bag/tfdifspace.dat"
        train_set = readbunchobj(trainpath)
        vecLen =train_set.tdm.shape[1]


        clf = svm.SVC()
        # clf = MultinomialNB(alpha=0.001)
        clf.fit(train_set.tdm, train_set.label)

        return clf,vecLen

def Predict(clf,vecLen,text):
        if(text==''):
                return ''
        segs = segment_Line(text)
        bunch=seg2Bunch(segs)
        space = bunch2Space(bunch)
        data=space.tdm

        testData=np.zeros((1,vecLen))
        for i in range(data.shape[1]):
                testData[0,i]=data[0,i]
        
        predicted = clf.predict(testData)
        return predicted

def SaveText(fileName,text):
        if(text!=''):
                textFile = open(fileName, 'a')
                textFile.write(','+text)
                textFile.close()

def ReTrain(t0,t1,t2,t3):
        SaveText('train_corpus/0/0',t0)
        SaveText('train_corpus/1/1',t1)
        SaveText('train_corpus/2/2',t2)
        SaveText('train_corpus/3/3',t3)
        return Train()

def predictItem(clf,len,text):
        result=Predict(clf,vecLen,text)
        print('推荐类别：')
        print(result)

clf,vecLen=Train()

num=0
text=''
text0=''
text1=''
text2=''
text3=''
while(True):
        if(text=='Q' or text=='q'):
                break

        if(num==10):
                print("正在更新分类器，请耐心等待...")
                clf,vecLen=ReTrain(text0,text1,text2,text3)
                print("更新完成")
        
        text=input("请输入句子(输入Q退出): ")
        print(text)
        predictItem(clf,vecLen,text)  

        flag=False
        while(not flag):
                flag=True
                classIndex=input("请输入实际类别(0-3): ")
                if(classIndex=='0'):
                        text0=text0+','+text
                        num+=1                
                elif(classIndex=='1'):
                        text0=text1+','+text
                        num+=1
                elif(classIndex=='2'):
                        text0=text2+','+text
                        num+=1
                elif(classIndex=='3'):
                        text0=text3+','+text
                        num+=1
                else:
                        flag=False
                        print('请输入正确的类别')
                

