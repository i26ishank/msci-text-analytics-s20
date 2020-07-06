import os
import pandas as pd
import numpy as np
import sys
import keras
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from a4 import do_it
def main(path,calssifier_to_use):
    # Loading the model depending upon the activation function
    if(calssifier_to_use.lower()=='relu'):
        model = keras.models.load_model('data/nn_relu.model')
    elif(calssifier_to_use.lower()=='sigmoid'):
        model = keras.models.load_model('data/nn_sigmoid.model')
    elif(calssifier_to_use.lower()=='tanh'):
        model = keras.models.load_model('data/nn_tanh.model')
    # Tokenizing sentence one by one from test file 
    btr=[]
    with open(str(path)) as ti:
        s=ti.read().split("\n")
        print(s)
    for i in s:
            btr.append(i.split(" "))
    
    print(btr)

    train_with_stopwords=pd.read_csv(os.path.join('data/','train_with_stopwords.csv'),sep="\n",names=['features'])
    val_with_stopwords=pd.read_csv(os.path.join('data/','val_with_stopwords.csv'),sep="\n",names=['features'])
    test_with_stopwords=pd.read_csv(os.path.join('data/','test_with_stopwords.csv'),sep="\n",names=['features'])

    Data=pd.concat((train_with_stopwords,val_with_stopwords,test_with_stopwords),axis=0)
    print(type(Data))
    print(Data.head())
    Data=[str(Data.iloc[i]).split(',') for i in range(len(Data))]
    print(Data[0:1])
    t=Tokenizer(num_words=27000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    
    t.fit_on_texts([' '.join(seq[:25]) for seq in Data])
    
    X_train= t.texts_to_sequences([' '.join(seq[:25]) for seq in btr])
    a=len(X_train)
    X_train = pad_sequences(X_train, maxlen=25, padding='post', truncating='post')
    # Prediction in categorical form
    yhat = model.predict(X_train, verbose=0)
    
    # Changing categorical form into normal labels for classification into positive or negative classes
    print("\n")
    for i in range(a):
        j=i
        t=np.argmax(yhat[i])
        if(t==1):
            print("The review ",j+1," is positive")
        else:
            print("The review ",j+1," is negative")
        print("\n")
if __name__=='__main__':
     main(sys.argv[1],sys.argv[2])