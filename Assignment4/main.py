from a4 import do_it

import os
def main(data):
     # calling the function of a4.py to run and construct a neural model and find accuracy of classification of Amazon corpus 
     do_it(data)
     

if __name__=='__main__':
     main(os.sys.argv[1])