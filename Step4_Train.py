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
import cv2

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
        if(len(text)<2):
                text=text+text
        try:
                segs = segment_Line(text)
                bunch=seg2Bunch(segs)
                space = bunch2Space(bunch)
                data=space.tdm

                testData=np.zeros((1,vecLen))
                for i in range(data.shape[1]):
                        testData[0,i]=data[0,i]
                
                predicted = clf.predict(testData)
                return predicted
        except Exception as err:
                return '0'

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
def ShowImg(classIndex):
        fileName='Images/'+str(classIndex[0])+'.jpg'
        img=cv2.imread(fileName)
        cv2.imshow('Recommended',img)
        cv2.waitKey(5000)
        cv2.destroyWindow('Recommended')

def predictItem(clf,len,text):
        result=Predict(clf,vecLen,text)
        ShowImg(result)
        print('Recommended Class:')
        print(result)

clf,vecLen=Train()

num=0
text=''
text0=''
text1=''
text2=''
text3=''
while(True):
        text=input("Please enter sentence(Q for exit): ")
        if(text=='Q' or text=='q'):
                break

        if(num==10):
                print("Updating classifier, please wait patiently...")
                clf,vecLen=ReTrain(text0,text1,text2,text3)
                print("Update done")
        
        
        predictItem(clf,vecLen,text)  
        flag=False
        while(not flag):
                flag=True
                classIndex=input("Please enter the real class(0-3): ")
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
                        print('Please enter the right class index:')
                

