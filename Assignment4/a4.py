import random
import pandas as pd
import numpy as np
import io
import os
import sys
from gensim.models import Word2Vec
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, Dropout, Activation, Flatten
from keras import regularizers
from keras.layers.convolutional import Conv2D,Conv1D
import plotly
import plotly.graph_objects as table

# Loading the required data files
def do_it(path1):
  train_with_stopwords=pd.read_csv(os.path.join(path1,'train_with_stopwords.csv'),sep="\n",names=['features'])
  val_with_stopwords=pd.read_csv(os.path.join(path1,'val_with_stopwords.csv'),sep="\n",names=['features'])
  test_with_stopwords=pd.read_csv(os.path.join(path1,'test_with_stopwords.csv'),sep="\n",names=['features'])
  train_label_with_stopwords=pd.read_csv(os.path.join('data/train_labels.csv'),names=['labels'])
  val_label_with_stopwords=pd.read_csv(os.path.join('data/val_labels.csv'),names=['labels'])
  test_label_with_stopwords=pd.read_csv(os.path.join('data/test_labels.csv'),names=['labels'])

  # Loading the embeddings 
  embeddings = Word2Vec.load(os.path.join(path1,'my_model.model'))

  t=Tokenizer(num_words=27000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
  # merging training,validation and testing data with their labels
  train_data_with_stopwords=pd.concat([train_with_stopwords,train_label_with_stopwords], axis=1)
  val_data_with_stopwords=pd.concat([val_with_stopwords,val_label_with_stopwords],axis=1)
  test_data_with_stopwords=pd.concat([test_with_stopwords,test_label_with_stopwords],axis=1)

  # Forming training ,test and validation sets
  X_train_with_stopwords=train_data_with_stopwords['features']
  Y_train_with_stopwords=train_data_with_stopwords['labels']
  X_val_with_stopwords=val_data_with_stopwords['features']
  Y_val_with_stopwords=val_data_with_stopwords['labels']
  X_test_with_stopwords=test_data_with_stopwords['features']
  Y_test_with_stopwords=test_data_with_stopwords['labels']

  Data=pd.concat((X_train_with_stopwords,X_val_with_stopwords,X_test_with_stopwords),axis=0)
  # Forming tokenized lists
  X_train_with_stopwords=[(X_train_with_stopwords.iloc[i]).split(',') for i in range(len(X_train_with_stopwords))]
  X_val_with_stopwords=[(X_val_with_stopwords.iloc[i]).split(',') for i in range(len(X_val_with_stopwords))]
  X_test_with_stopwords=[(X_test_with_stopwords.iloc[i]).split(',') for i in range(len(X_test_with_stopwords))]
  Data=[(Data.iloc[i]).split(',') for i in range(len(Data))]


  len1 = int(np.percentile([len(seq) for seq in Data], 95))
  
  # Forming text to sequences
  t.fit_on_texts([' '.join(seq[:len1]) for seq in Data])
  X_train_with_stopwords = t.texts_to_sequences([' '.join(seq[:len1]) for seq in X_train_with_stopwords])
  X_val_with_stopwords = t.texts_to_sequences([' '.join(seq[:len1]) for seq in X_val_with_stopwords])
  X_test_with_stopwords = t.texts_to_sequences([' '.join(seq[:len1]) for seq in X_test_with_stopwords])
  
  # Using padding
  X_train_with_stopwords = pad_sequences(X_train_with_stopwords, maxlen=len1, padding='post', truncating='post')
  X_val_with_stopwords = pad_sequences(X_val_with_stopwords, maxlen=len1, padding='post', truncating='post')
  X_test_with_stopwords = pad_sequences(X_test_with_stopwords, maxlen=len1, padding='post', truncating='post')
  vocabulary_size=len(t.word_index)+1
  embedding_dimensions=embeddings.vector_size
 

  embedding_matrix=np.random.randn(vocabulary_size,embedding_dimensions)

  # Forming embedding matrix
  for g,i in t.word_index.items():
    if g in embeddings.wv.vocab:
      embedding_matrix[i]=embeddings[g]
    else:
      embedding_matrix[i]=np.random.randn(1,embedding_dimensions)


  # changing labels to categorical form
  Y_train_with_stopwords=np_utils.to_categorical(Y_train_with_stopwords)
  Y_val_with_stopwords=np_utils.to_categorical(Y_val_with_stopwords)
  Y_test_with_stopwords=np_utils.to_categorical(Y_test_with_stopwords)

  # Checking the input shape and details of the input to the neural network
  print(X_train_with_stopwords)
  print(X_train_with_stopwords.shape)

  # fully-connected feed-forward neural network
  my_classifier=Sequential()

  # embedding layer
  my_classifier.add(Embedding(input_dim=vocabulary_size,output_dim=embedding_dimensions,weights=[embedding_matrix], input_length=len1,trainable=False))
  my_classifier.add(Flatten())

  # including L2 
  my_classifier.add(Dense(140,activation='tanh',kernel_regularizer=regularizers.l2(0.0005)))

  # Using ReLU activation with l2
  # neurons=140,batch size 1024
  #Dropout=0.2,using only dense layer,relu activation, l2(not using)  ---76.59% accuracy
  #Dropout=0.2,using only dense layer,relu activation, l2=0.0005  ---77.63% accuracy
  #Dropout=0.2,trainable=True,using only dense layer,relu activation, l2=0.0005  ---81.03% accuracy
  # Dropout=0.2,using only dense layer,relu activation, l2=0.005  ---75.20% accuracy
  # Dropout=0.2,using only dense layer,relu activation, l2=0.05  --- 72.59% accuracy
  # Dropout=0.2,using only dense layer,relu activation, l2=0.5  ---  67.64% accuracy
   
  
  # Dropout=0.1,using only dense layer,relu activation, l2=0.0005  ---77.19% accuracy
  # Dropout=0.3,using only dense layer,relu activation, l2=0.0005  ---76.98% accuracy
  # Dropout=0.4,using only dense layer,relu activation, l2=0.0005  ---76.92% accuracy
  # Dropout=0.5,using only dense layer,relu activation, l2=0.0005  ---76.71% accuracy
  # Dropout=0.6,using only dense layer,relu activation, l2=0.0005  ---76.08% accuracy
  

  # Neurons =200 Dropout=0.2,using only dense layer,relu activation, l2=0.0005  ---75.44% accuracy
  # Neurons =170 Dropout=0.2,using only dense layer,relu activation, l2=0.0005  ---77.145% accuracy
  # Neurons=120 Dropout=0.2 , rusing only dense layer, relu activation, l2=0.0005----77.52% accuracy
  # Neurons=100 Dropout=0.2 , rusing only dense layer, relu activation, l2=0.0005----77.0925% accuracy

  # Using tanh activation with l2
  # neurons=140, Dropout=0.2, using only dense layer, l2=0.0005------80.95% accuracy
  # neurons=140, Dropout=0.2, using only dense layer, l2=0.0005------76.62% accuracy
  # neurons=140, Dropout=0.2, using only dense layer, l2(not used)------77.21% accuracy
  # neurons=140, Dropout=0.2, using only dense layer, l2=0.005------75.17% accuracy
  # neurons=140, Dropout=0.2, using only dense layer, l2=0.05------73.47% accuracy
  # neurons=140, Dropout=0.2, using only dense layer, l2=0.5------70.21% accuracy

  # neurons=140, Dropout=0.1, using only dense layer, l2=0.0005------70.96% accuracy
  # neurons=140, Dropout=0.3, using only dense layer, l2=0.0005------76.06% accuracy
  # neurons=140, Dropout=0.4, using only dense layer, l2=0.0005------76.11% accuracy
  # neurons=140, Dropout=0.5, using only dense layer, l2=0.0005------75.85% accuracy
  # neurons=140, Dropout=0.6, using only dense layer, l2=0.0005------75.82% accuracy


  # Using sigmoid activation with l2
  # neurons=140, Dropout=0.2,trainable=True, using only dense layer, l2=0.0005------81.10% accuracy
  # neurons=140, Dropout=0.2, using only dense layer, l2=0.0005------77.538% accuracy
  # neurons=140, Dropout=0.2, using only dense layer, l2(not used)------77.33% accuracy
  # neurons=140, Dropout=0.2, using only dense layer, l2=0.005------75.386% accuracy
  # neurons=140, Dropout=0.2, using only dense layer, l2=0.05------71.93% accuracy
  # neurons=140, Dropout=0.2, using only dense layer, l2=0.5------66.63% accuracy

  # neurons=140, Dropout=0.3, using only dense layer, l2=0.0005------77.00% accuracy
  # neurons=140, Dropout=0.4, using only dense layer, l2=0.0005------77.15% accuracy
  # neurons=140, Dropout=0.5, using only dense layer, l2=0.0005------76.0625% accuracy
  # neurons=140, Dropout=0.6, using only dense layer, l2=0.0005------76.18% accuracy

  #neurons=140, Dropout=0.3, trainable= True, using only dense layer, l2=0.0005------80.86% accuracy

  

  my_classifier.add(Dropout(rate=0.2))

  my_classifier.add(Dense(2,activation='softmax'))
  my_classifier.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

  my_classifier.fit(X_train_with_stopwords, Y_train_with_stopwords,batch_size=1024,epochs=20,validation_data=(X_val_with_stopwords, Y_val_with_stopwords))
  
  # evaluating testing set
  res=my_classifier.evaluate(X_test_with_stopwords,Y_test_with_stopwords)[1]
  print(f"Accuracy of model when test Test is taken  : {res*100} ")
  # Saving the model
  my_classifier.save('data/nn_tanh.model')

  # Classification accuracy result in a table
  Table_o= table.Figure(data=[table.Table(header=dict(values=
  ['ReLU(%)', 'Tanh(%)','Sigmoid(%)',' L2-norm(%)','Dropout','Trainable(Embedding Layer)']),
                  cells=dict(values=[
                  [77.63,81.03,75.20,72.59,67.64,77.19,76.98,76.92,76.71,76.08,76.59], 
                  [76.62,80.95,75.17,73.47,70.21,70.96,76.06,76.11,75.85,75.82,77.21],
                  [77.54,81.10,75.39,71.93,66.63,75.51,77.01,77.15,76.06,76.18,77.33],
                  [0.0005,0.0005,0.005,0.05,0.5,0.0005,0.0005,0.0005,0.0005,0.0005,0],
                  [0.2,0.2,0.2,0.2,0.2,0.1,0.3,0.4,0.5,0.6,0.2],
                  ['False','True','False','False','False','False','False','False','False','False','False']      ]))
                      ])
  Table_o.show()  

  return t
