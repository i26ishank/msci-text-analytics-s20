from gensim.models import Word2Vec
import sys
import os
def main(input_txt_file_path):
    my_model = Word2Vec.load('my_model.model')
    al=[]
    with open(str(input_txt_file_path),'r') as q:
        s=q.read().split("\n")
    # al.append(s)
    print(s)
    for i in s:
    
        result_n = my_model.most_similar(str(i), topn=20)
        print(result_n)
        print("\n")
if __name__=='__main__':
    main(sys.argv[1])

    



