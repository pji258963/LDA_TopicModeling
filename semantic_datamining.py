# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 00:07:53 2018

@author: jerry.liu
"""
import matplotlib.pyplot as plt
from gensim import corpora, models, similarities
from itertools import chain
import json
from ConfigParser import RawConfigParser
import numpy as np

#num_lines = sum(1 for line in open('2018.05.13SS.json'))
k=w=q=s=r=z=0
Abstractset1=Abstractset2=Abstractset3=Abstractset4=a=t=[]
Topics_num=10
b=[0]*Topics_num
c1=[0]*Topics_num
c2=[0]*Topics_num
c3=[0]*Topics_num
c4=[0]*Topics_num
d=[0]*(Topics_num-1)
e=[0]*(Topics_num-2)
f=[0]*Topics_num
#print num_lines
file= open('2018.05.13SS.json' , 'r') 
for line in file.readlines():
        jf = json.loads(line)
        print [k,w,q,s,r,z]
        print line
        if "reinforcement learning" in jf["title"]:    
         if ("year" not in jf): 
          print "There is no year data!!!"
          k=k+1
         elif int(jf["year"])>=2000 and int(jf["year"])<=2005:
          Abstractset1.append(jf["title"])
          print "it's elif1"
          w=w+1
         elif int(jf["year"])>=2006 and int(jf["year"])<=2010:
          Abstractset2.append(jf["title"])
          print "it's elif2"
          q=q+1
         elif int(jf["year"])>=2011 and int(jf["year"])<=2015:
          Abstractset3.append(jf["title"])        
          print "it's elif3"
          s=s+1
         elif int(jf["year"])>=2016 and int(jf["year"])<=2018:
          Abstractset4.append(jf["title"])        
          print "it's elif4"
          t=t+1
         else:
          print "it's elses' working!"   
          r=r+1
        else:
          print "recurrent neural network is not in the string! "
        z=z+1
d[0]=[k,w,q,s,t,r,z]
print d[0]
#print Abstractset1
print "================="
#print Abstractset2
#print "================="
#print Abstractset3
print"===================="
# remove common words and tokenize
stoplist = set('for in by with on of at and to a an the this is was there are were I we us or'.split())
texts1 = [[word for word in document.lower().split() if word not in stoplist]
         for document in Abstractset1]
# remove words that appear only once
all_tokens = sum(texts1, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts1 = [[word for word in text if word not in tokens_once] for text in texts1]

# Create Dictionary.
id2word = corpora.Dictionary(texts1)
# Creates the Bag of Word corpus.
mm1 = [id2word.doc2bow(text) for text in texts1]

# Trains the LDA models.
lda = models.ldamodel.LdaModel(corpus=mm1, id2word=id2word, num_topics=Topics_num, \
                               update_every=1, chunksize=10000, passes=1)

# Prints the topics.
for top in lda.print_topics():
  print top
#got doc_topic
all_topics = lda.get_document_topics(mm1)

for doc_topics in all_topics:
    print('Document \n')
    print 'Document topics:', doc_topics
    for i in range(0,len(doc_topics)):
        a=doc_topics[i]
        b[a[0]]=a[1]
    print max(b)    
    for k in range(0,Topics_num):
        if float(b[k])==float(max(b)):
          c1[k]=c1[k]+1
    print c1
d[1]=c1
print d     
print a
######round2#######

# remove common words and tokenize
stoplist = set('for in by with on of at and to a an the this is was there are were I we us '.split())
texts2 = [[word for word in document.lower().split() if word not in stoplist]
         for document in Abstractset2]
# remove words that appear only once
all_tokens = sum(texts2, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts2 = [[word for word in text if word not in tokens_once] for text in texts2]

# Create Dictionary.
id2word = corpora.Dictionary(texts2)
# Creates the Bag of Word corpus.
mm2 = [id2word.doc2bow(text) for text in texts2]

# Trains the LDA models.
lda = models.ldamodel.LdaModel(corpus=mm2, id2word=id2word, num_topics=Topics_num, \
                               update_every=1, chunksize=10000, passes=1)

# Prints the topics.
for top in lda.print_topics():
  print top

#got doc_topic
all_topics = lda.get_document_topics(mm2)

for doc_topics in all_topics:
    print('Document \n')
    print 'Document topics:', doc_topics
    for i in range(0,len(doc_topics)):
        a=doc_topics[i]
        b[a[0]]=a[1]
    print max(b)    
    for k in range(0,Topics_num):
        if float(b[k])==float(max(b)):
          c2[k]=c2[k]+1
    print c2  
d[2]=c2
print d
######round3#######

# remove common words and tokenize
stoplist = set('for in by with on of at and to a an the this is was there are were I we us '.split())
texts3 = [[word for word in document.lower().split() if word not in stoplist]
         for document in Abstractset3]
# remove words that appear only once
all_tokens = sum(texts3, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts3 = [[word for word in text if word not in tokens_once] for text in texts3]

# Create Dictionary.
id2word = corpora.Dictionary(texts3)
# Creates the Bag of Word corpus.
mm3 = [id2word.doc2bow(text) for text in texts3]

# Trains the LDA models.
lda = models.ldamodel.LdaModel(corpus=mm3, id2word=id2word, num_topics=Topics_num, \
                               update_every=1, chunksize=10000, passes=1)

# Prints the topics.
for top in lda.print_topics():
  print top
 #got doc_topic
all_topics3 = lda.get_document_topics(mm3)
for doc_topics in all_topics3:
    print('Document \n')
    print 'Document topics:', doc_topics
    for i in range(0,len(doc_topics)):
        a=doc_topics[i]
        b[a[0]]=a[1]
    print max(b)    
    for k in range(0,Topics_num):
        if float(b[k])==float(max(b)):
          c3[k]=c3[k]+1
    print c3  
d[3]=c3
print d
######round4#######

# remove common words and tokenize
stoplist = set('for in by with on of at and to a an the this is was there are were I we us '.split())
texts4 = [[word for word in document.lower().split() if word not in stoplist]
         for document in Abstractset4]
# remove words that appear only once
all_tokens = sum(texts4, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts4 = [[word for word in text if word not in tokens_once] for text in texts4]

# Create Dictionary.
id2word = corpora.Dictionary(texts4)
# Creates the Bag of Word corpus.
mm4 = [id2word.doc2bow(text) for text in texts4]

# Trains the LDA models.
lda = models.ldamodel.LdaModel(corpus=mm4, id2word=id2word, num_topics=Topics_num, \
                               update_every=1, chunksize=10000, passes=1)

# Prints the topics.
for top in lda.print_topics():
  print top

#got doc_topic
all_topics = lda.get_document_topics(mm4)

for doc_topics in all_topics:
    print('Document \n')
    print 'Document topics:', doc_topics
    for i in range(0,len(doc_topics)):
        a=doc_topics[i]
        b[a[0]]=a[1]
    print max(b)    
    for k in range(0,Topics_num):
        if float(b[k])==float(max(b)):
          c4[k]=c4[k]+1
    print c4  
d[4]=c4
print d 

for i in range(1,len(d)):
    e=d[i]
    print e 
    for p in range(0,Topics_num):
     h=e[p]
     print h
     plt.plot(i,h,'bo')
plt.show()