from gensim.models import Word2Vec
import sys
import os
# Function to calculate similar words to any word using word2vec model we trained
def main(input_path_txt_file):
    my_model = Word2Vec.load('my_model.model')
    s=[]
    with open(str(input_path_txt_file),'r') as q:
        s=q.read().split("\n")
    # al.append(s)
    print(s)
    for i in s:
    
        result_n = my_model.most_similar(str(i), topn=20)
        print(f"20 most similar words to {i} is:")
        print("\n")
        print(result_n)
        print("\n")
if __name__=='__main__':
    main(sys.argv[1])

    



