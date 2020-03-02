import Data as da
import tensorflow as tf


def create():
    config = {}
    config['train_path'] = './data/freebase15k/train.txt'
    config['test_path'] = './data/freebase15k/test.txt'
    config['validation_path'] = './data/freebase15k/valid.txt'
    return da.Data(config)

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    
@timing
def load_object(filename):
     with open(filename, 'rb') as input:
        return pickle.load(input)

@timing
def _main():
    d = create()
    _path = './data/freebase15k/data.pkl'
    save_object(d,_path )
    d = load_object(_path)
    print(d)
        

if __name__ == "__main__":
    _main()    