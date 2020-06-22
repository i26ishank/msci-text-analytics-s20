from A3 import do

import os
def main(data):
     # Calling the function in A3.py file tod o word2vec training and calculating 20 most similar words to good and bad
     do(data)
     

if __name__=='__main__':
     main(os.sys.argv[1])