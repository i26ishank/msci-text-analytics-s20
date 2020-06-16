
# ## Fetching train,validation and test data with stopwords from the csv files in form of lists of lists

# In[ ]:


import csv
from csv import reader
with open('data\\train_with_stopwords.csv', 'r') as ish:
    csv_reader = reader(ish)
    train_with_stopwords = list(csv_reader)  


# In[ ]:


with open('data\\val_with_stopwords.csv', 'r') as ish:
    csv_reader = reader(ish)
    val_with_stopwords = list(csv_reader)  


# In[ ]:


with open('data\\test_with_stopwords.csv', 'r') as ish:
    csv_reader = reader(ish)
    test_with_stopwords = list(csv_reader)  


# ## Fetching train,validation and test data without stopwords from the csv files in form of lists of lists

# In[ ]:


with open('data\\train_without_stopwords.csv', 'r') as ish:
    csv_reader = reader(ish)
    train_without_stopwords = list(csv_reader)  


# In[ ]:


with open('data\\val_without_stopwords.csv', 'r') as ish:
    csv_reader = reader(ish)
    val_without_stopwords = list(csv_reader)    


# In[ ]:


with open('data\\test_without_stopwords.csv', 'r') as ish:
    csv_reader = reader(ish)
    test_without_stopwords = list(csv_reader)  


# ## Making a function to separate labels from features in our dataset and store them in different variables

# In[ ]:


def split_labels(input):
    a=[]
    b=[]
    for i in input:
        a.append(i[0])
        b.append(i[1:])
    return a,b
    


# ## l_train represents the training labels and f_train represents training features (words from given corpus) with stopwords

# In[ ]:


l_train_with,f_train_with=split_labels(train_with_stopwords)


# ## l_val represents the validation labels and f_val represents validation features(words from given corpus) with stopwords

# In[ ]:


l_val_with,f_val_with=split_labels(val_with_stopwords)


# ## l_test represents the test labels and f_test represents test features(words from given corpus) with stopwords

# In[ ]:


l_test_with,f_test_with=split_labels(test_with_stopwords)


# ## l_train_without represents the training labels and f_train_without represents training features (words from given corpus) without stopwords

# In[ ]:


l_train_without,f_train_without=split_labels(train_without_stopwords)


# ## l_val_without represents the validation labels and f_val_without represents validation features(words from given corpus) without stopwords

# In[ ]:


l_val_without,f_val_without=split_labels(val_without_stopwords)


# ## l_test_without represents the test labels and f_test_without represents test features(words from given corpus) without stopwords

# In[ ]:


l_test_without,f_test_without=split_labels(test_without_stopwords)


# ## Making a function to tranform our training ,validation and test features into single list of string data where each string corresponds to a positive or negative review 

# In[ ]:


def corpus(input):
    p=[]
    for i in input:
        bh=" ".join(i)
        p.append(bh)
        bh=""
    return p


# ## Transforming training data with stopwords

# In[ ]:


data_train_with=corpus(f_train_with)


# ## Transforming validation data with stopwords

# In[ ]:


data_val_with=corpus(f_val_with)


# ## Transforming test data with stopwords

# In[ ]:


data_test_with=corpus(f_test_with)


# ## Transforming training data without stopwords

# In[ ]:


data_train_without=corpus(f_train_without)


# ## Transforming validation data without stopwords

# In[ ]:


data_val_without=corpus(f_val_without)


# ## Transforming test data without stopwords

# In[ ]:


data_test_without=corpus(f_test_without)


# ## Forming an unigram vectorizer using TfidfVectorizer

# In[ ]:


import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer 

unigram_vectorizer = TfidfVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b')


# ## Using the unigram vectorizer to fit and tranform our training data with stopwords into a sparse matrix 

# In[ ]:


X_train_with= unigram_vectorizer.fit_transform(data_train_with)


# ## Using the unigram vectorizer to tranform our validation data with stopwords into a sparse matrix 

# In[ ]:


X_val_with=unigram_vectorizer.transform(data_val_with)


# ## Using the unigram vectorizer to tranform our test data with stopwords into a sparse matrix 

# In[ ]:


X_test_with=unigram_vectorizer.transform(data_test_with)


# ## Using the unigram vectorizer to fit and tranform our training data without stopwords into a sparse matrix 

# In[ ]:


X_train_without= unigram_vectorizer.fit_transform(data_train_without)


# ## Using the unigram vectorizer to tranform our validation data without stopwords into a sparse matrix 

# In[ ]:


X_val_without= unigram_vectorizer.transform(data_val_without)


# ## Using the unigram vectorizer tranform our test data without stopwords into a sparse matrix 

# In[ ]:


X_test_without= unigram_vectorizer.transform(data_test_without)


# ## Multinomial Naïve Bayes (MNB) classifier

# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


from sklearn.metrics import accuracy_score


# ## Applying Multinomial Naïve Bayes (MNB) classifier to unigram training data with stopwords

# In[ ]:


list_accuracy=[]
a=[0, 0.1, 0.5, 1,1.5,1.55,2,2.5,5,10,15,20,25,50, 100]
for i in a:     
    ish = MultinomialNB(alpha = i)    
    ish.fit(X_train_with,l_train_with)     
    prediction = ish.predict(X_val_with)     
    accuracy =accuracy_score(l_val_with,prediction)
    print(f'The accuracy for unigram with stopwords at alpha = {i} is: {accuracy*100} ')
    list_accuracy.append(accuracy*100)
        


# In[ ]:



best_alpha=a[list_accuracy.index(max(list_accuracy))]


# ## Clearly we get maximum accuracy at alpha=1.5 although aplha=1,0.5,1.55,2,2.5  all lead to accuracy close to the maximum accuracy

# ## So running the unigram test data (with stopwords) with Multinomial Naïve Bayes (MNB) classifier 

# In[ ]:


ish = MultinomialNB(alpha=best_alpha)    
ish.fit(X_train_with,l_train_with)     
prediction = ish.predict(X_test_with)     
accuracy =accuracy_score(l_test_with,prediction)
print(f'The accuracy for unigram with stopwords at alpha = {best_alpha} is: {accuracy*100} ')


# ## Applying Multinomial Naïve Bayes (MNB) classifier to unigram training data without stopwords

# In[ ]:


a=[0, 0.1, 0.5, 1,1.5,1.55,2,2.5,5,10,15,20,25,50, 100]
list_accuracy=[]
for i in a:     
    ish = MultinomialNB(alpha = i)    
    ish.fit(X_train_without,l_train_without)     
    prediction = ish.predict(X_val_without)     
    accuracy =accuracy_score(l_val_without,prediction)
    print(f'The accuracy for unigram without stopwords at alpha = {i} is: {accuracy*100} ')
    list_accuracy.append(accuracy*100)


# In[ ]:


best_alpha=a[list_accuracy.index(max(list_accuracy))]


# ## Clearly we get maximum accuracy at alpha=1 although aplha=1.5,0.5,1.55,2,2.5  all lead to accuracy close to the maximum accuracy

# ## So running the unigram test data (without stopwords) with Multinomial Naïve Bayes (MNB) classifier 

# In[ ]:


ish = MultinomialNB(alpha = best_alpha)    
ish.fit(X_train_without,l_train_without)     
prediction = ish.predict(X_test_without)     
accuracy =accuracy_score(l_test_without,prediction)
print(f'The accuracy for unigram without stopwords at alpha = {best_alpha} is: {accuracy*100} ')


# ## Forming an bigram vectorizer using TfidfVectorizer

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer 

bigram_vectorizer = TfidfVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b')


# ## Using the bigram vectorizer to fit and tranform our training data with stopwords into a sparse matrix 

# In[ ]:


X_train_with= bigram_vectorizer.fit_transform(data_train_with)


# ## Using the bigram vectorizer to tranform our validation data with stopwords into a sparse matrix  

# In[ ]:


X_val_with=bigram_vectorizer.transform(data_val_with)


# ## Using the bigram vectorizer to tranform our test data with stopwords into a sparse matrix  

# In[ ]:


X_test_with=bigram_vectorizer.transform(data_test_with)


# ## Using the bigram vectorizer to fit and tranform our training data without stopwords into a sparse matrix 

# In[ ]:


X_train_without= bigram_vectorizer.fit_transform(data_train_without)


# ## Using the bigram vectorizer to tranform our validation data without stopwords into a sparse matrix 

# In[ ]:


X_val_without= bigram_vectorizer.transform(data_val_without)


# ## Using the bigram vectorizer to tranform our test data without stopwords into a sparse matrix 

# In[ ]:


X_test_without= bigram_vectorizer.transform(data_test_without)


# ## Applying Multinomial Naïve Bayes (MNB) classifier to bigram training data with stopwords

# In[ ]:


list_accuracy=[]
a=[0, 0.1, 0.5, 1,1.5,1.55,2,2.5,5,10,15,20,25,50, 100]
for i in a:     
    ish = MultinomialNB(alpha = i)    
    ish.fit(X_train_with,l_train_with)     
    prediction = ish.predict(X_val_with)     
    accuracy =accuracy_score(l_val_with,prediction)
    print(f'The accuracy for bigram with stopwords at alpha = {i} is: {accuracy*100} ')
    list_accuracy.append(accuracy*100)
    


# In[ ]:


best_alpha=a[list_accuracy.index(max(list_accuracy))]


# ## Clearly we get maximum accuracy at alpha=0.5 although aplha=0.1,1 all lead to accuracy close to the maximum accuracy

# ## So running the bigram test data (with stopwords) with Multinomial Naïve Bayes (MNB) classifier 

# In[ ]:


ish = MultinomialNB(alpha = best_alpha)    
ish.fit(X_train_with,l_train_with)     
prediction = ish.predict(X_test_with)     
accuracy =accuracy_score(l_test_with,prediction)
print(f'The accuracy for bigram with stopwords at alpha = {best_alpha} is: {accuracy*100} ')
    


# ## Applying Multinomial Naïve Bayes (MNB) classifier to bigram training data without stopwords

# In[ ]:


list_accuracy=[]
a=[0, 0.1, 0.5, 1,1.5,1.55,2,2.5,5,10,15,20,25,50, 100]
for i in a:     
    ish = MultinomialNB(alpha = i)    
    ish.fit(X_train_without,l_train_without)     
    prediction = ish.predict(X_val_without)     
    accuracy =accuracy_score(l_val_without,prediction)
    print(f'The accuracy for bigram without stopwords at alpha = {i} is: {accuracy*100} ')
    list_accuracy.append(accuracy*100)


# In[ ]:


best_alpha=a[list_accuracy.index(max(list_accuracy))]


# ## Clearly we get maximum accuracy at alpha=0.5 although aplha=1,1.5.1.55 all lead to accuracy close to the maximum accuracy

# ## So running the bigram test data (without stopwords) with Multinomial Naïve Bayes (MNB) classifier 

# In[ ]:


ish = MultinomialNB(alpha = best_alpha)    
ish.fit(X_train_without,l_train_without)     
prediction = ish.predict(X_test_without)     
accuracy =accuracy_score(l_test_without,prediction)
print(f'The accuracy for bigram without stopwords at alpha = {best_alpha} is: {accuracy*100} ')


# ## Forming an unigram+bigram vectorizer using TfidfVectorizer

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer 

unigram_bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b')


# ## Using the unigram+bigram vectorizer to fit and tranform our training data with stopwords into a sparse matrix 

# In[ ]:


X_train_with= unigram_bigram_vectorizer.fit_transform(data_train_with)


# ## Using the unigram+bigram vectorizer to tranform our validation data with stopwords into a sparse matrix 

# In[ ]:


X_val_with=unigram_bigram_vectorizer.transform(data_val_with)


# ## Using the unigram+bigram vectorizer to tranform our test data with stopwords into a sparse matrix 

# In[ ]:


X_test_with=unigram_bigram_vectorizer.transform(data_test_with)


# ## Using the unigram+bigram vectorizer to fit and tranform our training data without stopwords into a sparse matrix 

# In[ ]:


X_train_without= unigram_bigram_vectorizer.fit_transform(data_train_without)


# ## Using the unigram+bigram vectorizer to tranform our validation data without stopwords into a sparse matrix 

# In[ ]:


X_val_without= unigram_bigram_vectorizer.transform(data_val_without)


# ## Using the unigram+bigram vectorizer to tranform our test data without stopwords into a sparse matrix 

# In[ ]:


X_test_without= unigram_bigram_vectorizer.transform(data_test_without)


# ## Applying Multinomial Naïve Bayes (MNB) classifier to unigram+bigram training data with stopwords

# In[ ]:


list_accuracy=[]
a=[0, 0.1, 0.5, 1,1.5,1.55,2,2.5,5,10,15,20,25,50, 100]
for i in a:     
    ish = MultinomialNB(alpha = i)    
    ish.fit(X_train_with,l_train_with)     
    prediction = ish.predict(X_val_with)     
    accuracy =accuracy_score(l_val_with,prediction)
    print(f'The accuracy for unigram_bigram with stopwords at alpha  = {i} is: {accuracy*100} ')
    list_accuracy.append(accuracy*100)


# In[ ]:


best_alpha=a[list_accuracy.index(max(list_accuracy))]


# ## Clearly we get maximum accuracy at alpha=0.5 although aplha=0.1,1 all lead to accuracy close to the maximum accuracy

# ## So running the unigram+bigram test data (with stopwords) with Multinomial Naïve Bayes (MNB) classifier 

# In[ ]:


ish = MultinomialNB(alpha = best_alpha)    
ish.fit(X_train_with,l_train_with)     
prediction = ish.predict(X_test_with)     
accuracy =accuracy_score(l_test_with,prediction)
print(f'The accuracy for unigram_bigram with stopwords at alpha  = {best_alpha} is: {accuracy*100} ')


# ## Applying Multinomial Naïve Bayes (MNB) classifier to unigram+bigram training data without stopwords

# In[ ]:


list_accuracy=[]
a=[0, 0.1, 0.5, 1,1.5,1.55,2,2.5,5,10,15,20,25,50, 100]
for i in a:     
    ish = MultinomialNB(alpha = i)    
    ish.fit(X_train_without,l_train_without)     
    prediction = ish.predict(X_val_without)     
    accuracy =accuracy_score(l_val_without,prediction)
    print(f'The accuracy for unigram_bigram without stopwords at alpha = {i} is: {accuracy*100}')
    list_accuracy.append(accuracy*100)


# In[ ]:


best_alpha=a[list_accuracy.index(max(list_accuracy))]


# ## Clearly we get maximum accuracy at alpha=0.5 although aplha=1,1.5,1.55 all lead to accuracy close to the maximum accuracy

# ## So running the unigram+bigram test data (without stopwords) with Multinomial Naïve Bayes (MNB) classifier 

# In[ ]:


ish = MultinomialNB(alpha = best_alpha)    
ish.fit(X_train_without,l_train_without)     
prediction = ish.predict(X_test_without)     
accuracy =accuracy_score(l_test_without,prediction)
print(f'The accuracy for unigram_bigram without stopwords at alpha = {best_alpha} is: {accuracy*100}')


# In[ ]:


#pip install plotly


# In[ ]:

#Forming a table using plotly library depicting accuracies calculated at test set
import plotly
import plotly.graph_objects as table

Table_o= table.Figure(data=[table.Table(header=dict(values=['Stopwords removed', 'text features','Accuracy(test set)']),
                 cells=dict(values=[['yes','yes','yes','no','no','no'], ['unigrams','bigrams','unigrams+bigrams','unigrams','bigrams','unigrams+bigrams'],[80.581,79.068,82.528,80.834,82.575,83.605]]))
                     ])
Table_o.show()