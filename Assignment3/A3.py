import csv
import sys
import os

from csv import reader
from gensim.models import Word2Vec
# Reading tokenized data
def do(path):
    with open(path+'out_with_stopwords.csv', 'r') as ish:
        csv_reader = reader(ish)
        out_with_stopwords = list(csv_reader)  
    # print(out_with_stopwords[:4])
    a=[]
    b=[]
    # Separating labels from features
    for i in out_with_stopwords:
        a.append(i[0])
        b.append(i[1:])

    # train model
    my_model = Word2Vec(b,workers=6,iter=50,min_count=1,size=300)
    # summarize the loaded model
    print(my_model)
    # summarize vocabulary
    # words = list(model.wv.vocab)
    # print(words)
    # access vector for one word
    # print(model['sentence'])
    # save model
    my_model.save(path+'my_model.model')
    # load model
    # new_model = Word2Vec.load('model.bin')
    # print(new_model)
    # load model
    my_model = Word2Vec.load(path+'my_model.model')

    # Finding 20 most similar words  to good and bad along with its value 
    print("20 most similar words to 'good' are:")
    result1 = my_model.most_similar(['good'], topn=20)
    print(result1)
    print("\n")
    print("20 most similar words to 'bad' are:")
    
    result2 = my_model.most_similar(['bad'], topn=20)
    print(result2)
    
   
    


