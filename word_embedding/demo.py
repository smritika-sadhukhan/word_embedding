import logging
import numpy as np

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
  
def find_k_nearest(source, vectors, k):
    norm1 = np.linalg.norm(source)
    norm2 = np.linalg.norm(vectors, axis=1)
    cosine_similarity = np.sum(source * vectors, axis=1) / norm1 / norm2
    return np.argsort(cosine_similarity)[::-1][1:(k + 1)]

def get_nearest_vector(lang,list1,output):
    
    demo_nearest = list1
    # load word vector
    words = []
    vectors = []
    logging.info('Loading word vector')
    with open(output) as f:
        # skip first line
        f.readline()
        line = f.readline()
        while len(line) > 0:
            line = line.split(' ')
            words.append(line[0])
            vectors.append(np.array([float(x) for x in line[1:]]))
            line = f.readline()
    vectors = np.vstack(vectors)

    # demo word similarity
    k = 5
    for word in demo_nearest:
        word_index = words.index(word)
        k_nearest = find_k_nearest(vectors[word_index], vectors, k)
        logging.info('Nearest words of %s', word)
        for index in k_nearest:
            v1 = vectors[word_index, :]
            v2 = vectors[index, :]
            logging.info('word %s score %f', words[index], np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def get_vectors(lang,word,output):
    # load word vector
    words = []
    vectors = []
    logging.info('Loading word vector')
    with open(output) as f:
        # skip first line
        f.readline()
        line = f.readline()
        while len(line) > 0:
            line = line.split(' ')
            if word==line[0]:
                 vectors.append(np.array([float(x) for x in line[1:]]))
                 vectors = np.vstack(vectors)
            line = f.readline()

    return(vectors)