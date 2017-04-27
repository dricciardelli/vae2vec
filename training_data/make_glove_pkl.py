import pickle as pkl
import argparse
import numpy as np
import pandas as pd

annotation_path = 'results_20130124.token'
def_path='D_cbow_pdw_8B.pkl'
def get_captions(annotation_path):
    annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
    capts=annotations['caption'].values
    # function from Andre Karpathy's NeuralTalk
    # print('preprocessing %d word vocab' % (word_count_threshold, ))
    word_counts = {}
    nsents = 0
    for sent in capts:
      nsents += 1
      for w in sent.lower().split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

    return word_counts.keys()
def get_definitions(def_path):
    emb=pkl.load(open(def_path,'rb'))
    return emb.keys()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors_file', default='glove/glove.6B.300d.txt', type=str)
    args = parser.parse_args()
    task_words=set()
    task_words|=set(get_captions(annotation_path))
    task_words|=set(get_definitions(def_path))

    num_words_in=0
    num_words_out=0
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            if vals[0].lower() in task_words:
                vectors[vals[0]] = np.array([float(x) for x in vals[1:]])
                num_words_in+=1
            else:
                num_words_out+=1
    print (num_words_in,num_words_out)

    # vocab_size = len(words)
    # vocab = {w: idx for idx, w in enumerate(words)}
    # ivocab = {idx: w for idx, w in enumerate(words)}

    # vector_dim = len(vectors[ivocab[0]])
    # W = np.zeros((vocab_size, vector_dim))
    # for word, v in vectors.items():
    #     if word == '<unk>':
    #         continue
    #     W[vocab[word], :] = v

    pkl.dump(vectors,open('glove_embeddings.pkl','wb'))


if __name__ == "__main__":
    main()