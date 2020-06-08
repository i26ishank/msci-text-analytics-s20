

# ### Opening positive reviews and negative reviews files downloaded

# In[ ]:


ish=open("D:\\641\\pos.txt")


# In[ ]:


ish_neg=open("D:\\641\\neg.txt")


# In[ ]:


import random


# ### Reading data from both files and storing them in variables

# In[ ]:


p=ish.read()


# In[ ]:


n=ish_neg.read()


# ### Closing  both files

# In[ ]:


ish.close()
ish_neg.close()


# ### Creating duplicate files in case of any data loss

# In[ ]:


with open("D:\\641\\pos.txt") as f:
    with open("D:\\641\\positive.txt", "w") as f1:
        for line in f:
            f1.write(line)


# In[ ]:


with open("D:\\641\\neg.txt") as f:
    with open("D:\\641\\negative.txt", "w") as f1:
        for line in f:
            f1.write(line)


# ### Tokenizing   the corpus(positive reviews)

# In[ ]:


a=p.split("\n")


# In[ ]:


b_pos=[]
for i in a:
    b_pos.append(i.split(" "))


# In[ ]:


b_pos.pop()


# ### Removing the following special characters:  (!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n) from tokenized positive reviews data

# In[ ]:



cl_pos=[]
lo=[]
for i in range(0,len(b_pos)):
    d=b_pos[i]
    t=''
    for j in range(0,len(d)):
        e=d[j]
        
        
        bad_chars = ["'",',','!','"','#','$','&','(',')','*',"' '",'+','/',':',';','<','=','>','@','[','\\',']','^','`','{','|','}','~','\t','\n','%'] 
        
        for k in range(0,len(e)):
            if e[k] not in bad_chars:
                
                t+=e[k]
        if t=='' or t==' ' or t=="" or t==" ":
            continue
        else:
            lo.append(t)
        t=''
        
    cl_pos.append(lo)
    lo=[]


# ### Tokenizing   the corpus(negative reviews)

# In[ ]:


c=n.split("\n")


# In[ ]:


c_neg=[]
for j in c:
    c_neg.append(j.split(" "))


# In[ ]:


c_neg.pop()


# ### Removing the following special characters:  (!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n) from tokenized negative reviews data
# 

# In[ ]:


cl_neg=[]
lo=[]
for i in range(0,len(c_neg)):
    d=c_neg[i]
    t=''
    for j in range(0,len(d)):
        e=d[j]
        
        
        bad_chars = ["'",',','!','"','#','$','&','(',')','*',"' '",'+','/',':',';','<','=','>','@','[','\\',']','^','`','{','|','}','~','\t','\n','%'] 
        
        for k in range(0,len(e)):
            if e[k] not in bad_chars:
                
                t+=e[k]
        if t=='' or t==' ' or t=="" or t==" ":
            continue
        else:
            lo.append(t)
        t=''
        
    cl_neg.append(lo)
    lo=[]


# ### Creating customized stopwords list

# In[ ]:


stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


# In[ ]:


len(stop_words)


# In[ ]:


l2=["i've","you've","we've","they've","i'd","you'd","he'd","she'd","we'd","they'd","i'll","you'll","he'll","she'll","we'll","they'll","isn't","aren't","wasn't","weren't","hasn't","haven't","hadn't","doesn't","don't","didn't","won't","wouldn't","shan't","shouldn't","can't","cannot","couldn't","mustn't","let's","that's","who's","what's","here's","there's","when's","where's","why's","how's","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","ought","i'm","you're","he's","she's","it's","we're","they're","having","do","does","did","doing","would","should","could","was","were","be","been","being","have","has","had","whom","this","that","these","those","am","is","are","they","them","their","theirs","themselves","what","which","who","himself","she","her","hers","herself","it","its","itself","you","your","yours","yourself","yourselves","he","him","his","i","me","my","myself","we","our","ours","ourselves","i'm"]     


# In[ ]:


k=[]
for j in l2:
    if j not in stop_words:
        k.append(j)


# In[ ]:


for i in k:
    stop_words.append(i)


# In[ ]:


len(stop_words)


# ### Removing the stop words from tokenized positive reviews data

# In[ ]:



a=[]
cl_pos_without_stopwords=[]
for i in range(0,len(cl_pos)):
    d=cl_pos[i]
    t=d
    for j in range(0,len(t)):
        
        if d[j].lower() not in stop_words:
            
            a.append(d[j])
    cl_pos_without_stopwords.append(a)
    a=[]

       


# ### Removing the stopwords from tokenized negative reviews data

# In[ ]:


lp=[]
cl_neg_without_stopwords=[]
for i in range(0,len(cl_neg)):
    d=cl_neg[i]
    t=d
    for j in range(0,len(t)):
        
        if d[j].lower() not in stop_words:
            
            lp.append(d[j])
    cl_neg_without_stopwords.append(lp)
    lp=[]


# ### Adding a column of label(1 for positive reviews and 2 for negative reviews) in each of the following:
# ### (1)tokenized positive reviews with stopwords
# ### (2)tokenized negative reviews with stopwords
# ### (3)tokenized positive reviews without stopwords
# ### (4)tokenized negative reviews without stopwords

# In[ ]:


for i in range(0,(len(cl_pos))):
    cl_pos[i].insert(0,'postive')
    


# In[ ]:


for i in range(0,(len(cl_neg))):
    cl_neg[i].insert(0,'negative')
    


# In[ ]:


for i in range(0,(len(cl_pos_without_stopwords))):
    cl_pos_without_stopwords[i].insert(0,'positive')
    


# In[ ]:


for i in range(0,(len(cl_neg_without_stopwords))):
    cl_neg_without_stopwords[i].insert(0,'negative')
    


# ### Creating .csv files for:
# ### (1)tokenized positive reviews with stopwords
# ### (2)tokenized negative reviews with stopwords
# ### (3)tokenized positive reviews without stopwords
# ### (4)tokenized negative reviews without stopwords

# In[ ]:


# import csv

# with open('D:\\641\\out_positive_with_stopwords.csv', mode='w') as review:
#     review_writer = csv.writer(review, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
#     for i in cl_pos:
#         review_writer.writerow(i)
   


# In[ ]:


# with open('D:\\641\\out_positive_without_stopwords.csv', mode='w') as review:
#     review_writer = csv.writer(review, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
#     for i in cl_pos_without_stopwords:
#         review_writer.writerow(i)


# In[ ]:


# with open('D:\\641\\out_negative_with_stopwords.csv', mode='w') as review:
#     review_writer = csv.writer(review, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
#     for i in cl_neg:
#         review_writer.writerow(i)


# In[ ]:


# with open('D:\\641\\out_negative_without_stopwords.csv', mode='w') as review:
#     review_writer = csv.writer(review, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
#     for i in cl_neg_without_stopwords:
#         review_writer.writerow(i)


# ### Creating combined list of tokenized reviews with stopwords (both positive and negative)

# In[ ]:


for i in cl_neg:
    cl_pos.append(i)


# In[ ]:


len(cl_pos)


# ### Creating combined list of tokenized reviews without stopwords (both positive and negative)

# In[ ]:


for i in cl_neg_without_stopwords:
    cl_pos_without_stopwords.append(i)



# ###  Creating .csv file  having combined data(tokenized reviews with stopwords (both positive and negative))

# In[ ]:



csv_data=""
for line in cl_pos:
        csv_line = '{}\n'.format(','.join(line))
        csv_data += csv_line

with open('D:\\641\\out_with_stopwords.csv', 'w') as f:
        f.write(csv_data)


# ###  Creating .csv file  having combined data(tokenized reviews without stopwords (both positive and negative))

# In[ ]:



csv_data=""
for line in cl_pos_without_stopwords:
        csv_line = '{}\n'.format(','.join(line))
        csv_data += csv_line
with open('D:\\641\\out_without_stopwords.csv', 'w') as f:
        f.write(csv_data)



# ### Creating a dummy list having combined data (tokenized reviews with stopwords (both positive and negative))

# In[ ]:


data_with_stopwords=[]
for i in cl_pos:
     data_with_stopwords.append(i)


# ### Creating a list having all the indices 

# In[ ]:


all_indices=[]
for i in range(0,800000):
    all_indices.append(i)
    


# In[ ]:


len(all_indices)


# ### Using random module for randomly splitting indices for our training set(with stopwords)
# 
# ### Here we have used random.seed() to make sure split is always the same every time we run the code

# In[ ]:


random.seed(5000)
a=0.8*len(data_with_stopwords)
indices_train_with_stopwords=random.sample(range(len(data_with_stopwords)),round(a))


# ### Training set(with stopwords)

# In[ ]:


train_with_stopwords=[]
for i in indices_train_with_stopwords:
    train_with_stopwords.append(data_with_stopwords[i])


# ### Writing training set with stopwords to a .csv file

# In[ ]:



csv_data=""
for line in train_with_stopwords:
        csv_line = '{}\n'.format(','.join(line))
        csv_data += csv_line
with open('D:\\641\\train_with_stopwords.csv', 'w') as f:
        f.write(csv_data)


# In[ ]:


rest=list(set(all_indices) - set(indices_train_with_stopwords))


# In[ ]:


len(rest)


# In[ ]:


random.seed(5000)
a=0.5*len(rest)
indices_val=random.sample(rest,round(a))


# In[ ]:


indices_test=list(set(rest) - set(indices_val))


# In[ ]:


len(indices_val)


# In[ ]:


len(indices_test)


# ### Validation set(with stopwords)

# In[ ]:


val_with_stopwords=[]
for i in indices_val:
    val_with_stopwords.append(data_with_stopwords[i])


# ### Writing Validation set with stopwords to a .csv file

# In[ ]:


csv_data=""
for line in val_with_stopwords:
        csv_line = '{}\n'.format(','.join(line))
        csv_data += csv_line
with open('D:\\641\\val_with_stopwords.csv', 'w') as f:
        f.write(csv_data)


# In[ ]:


train_with_stopwords[10]==val_with_stopwords[10]


# ###  Test set(with stopwords)

# In[ ]:


test_with_stopwords=[]
for i in indices_test:
    test_with_stopwords.append(data_with_stopwords[i])


# ### Writing Test set with stopwords to a .csv file

# In[ ]:


csv_data=""
for line in test_with_stopwords:
        csv_line = '{}\n'.format(','.join(line))
        csv_data += csv_line
with open('D:\\641\\test_with_stopwords.csv', 'w') as f:
        f.write(csv_data)



# In[ ]:


train_with_stopwords[10]==test_with_stopwords[10]


# ### Now creating train,test and validation splits for our dataset without stopwords

# ### Creating a dummy list having combined data (tokenized reviews without stopwords (both positive and negative))

# In[ ]:


data_without_stopwords=[]
for i in cl_pos_without_stopwords:
     data_without_stopwords.append(i)


# In[ ]:


len(data_without_stopwords)


# ### Using random module for randomly splitting indices for our training set(without stopwords)

# In[ ]:


random.seed(5000)
a=0.8*len(data_without_stopwords)
indices_train_without_stopwords=random.sample(range(len(data_without_stopwords)),round(a))


# ### Training set(without stopwords)

# In[ ]:


train_without_stopwords=[]
for i in indices_train_without_stopwords:
    train_without_stopwords.append(data_without_stopwords[i])


# ### Writing Train set without stopwords to a .csv file

# In[ ]:


csv_data=""
for line in train_without_stopwords:
        csv_line = '{}\n'.format(','.join(line))
        csv_data += csv_line
with open('D:\\641\\train_without_stopwords.csv', 'w') as f:
        f.write(csv_data)



# In[ ]:


rest=list(set(all_indices) - set(indices_train_without_stopwords))


# In[ ]:


len(rest)


# In[ ]:


random.seed(5000)
a=0.5*len(rest)
indices_val_without_stopwords=random.sample(rest,round(a))


# In[ ]:


indices_test_without_stopwords=list(set(rest) - set(indices_val_without_stopwords))


# In[ ]:


len(indices_val_without_stopwords)


# ### Validation set(without stopwords)

# In[ ]:


val_without_stopwords=[]
for i in indices_val_without_stopwords:
    val_without_stopwords.append(data_without_stopwords[i])


# ### Writing Validation set without stopwords to a .csv file

# In[ ]:


csv_data=""
for line in val_without_stopwords:
        csv_line = '{}\n'.format(','.join(line))
        csv_data += csv_line
with open('D:\\641\\val_without_stopwords.csv', 'w') as f:
        f.write(csv_data)



# ### Test set(without stopwords)

# In[ ]:


test_without_stopwords=[]
for i in indices_test_without_stopwords:
    test_without_stopwords.append(data_without_stopwords[i])


# ### Writing Test set without stopwords to a .csv file

# In[ ]:


csv_data=""
for line in test_without_stopwords:
        csv_line = '{}\n'.format(','.join(line))
        csv_data += csv_line
with open('D:\\641\\test_without_stopwords.csv', 'w') as f:
        f.write(csv_data)


