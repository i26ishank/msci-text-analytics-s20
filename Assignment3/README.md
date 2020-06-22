## No, nor all the words similar to “good” are positive, neither are all the words close to “bad” are negative. 

### Reason: What word2vec model goes for is “syntactic similarity” and not the “semantic similarity”. Which means for word2vec, two words are similar if they happen in similar context. So, a word “to” is similar to word “too” if the words near them are similar to each other. This reasoning is not always true. For example, the words “good” and “bad” though semantically far away from each other, but they are bound to be found in similar context and in vector space antonyms are quite close to each other(syntactic distance difference is less). So, for word2vec model these words are a bit like each other.

you will be able to run A3.py from main file by command line arguement like:
 python a3.py data_folder path(say data/) . 
 Although you will be able to run "inference.py" seperately by command like(python inference.py path_of_text_file(like python inference.py data/try.txt) 

I have tried to be as close to the guidelines as possible. However if there is any mistake of my representation of guidelines, mention in feedback and i will take care of it in next assignment. Thanks again for your continued help.
