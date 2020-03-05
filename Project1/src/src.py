import argparse
import math
import nltk
import string
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')


def normalize_doc(term_weights_doc):
    norm = 0.0
    for t in term_weights_doc.keys():
        norm += term_weights_doc[t] * term_weights_doc[t]
    norm = math.sqrt(norm)
    for t in term_weights_doc.keys():
        term_weights_doc[t] = term_weights_doc[t] / norm


def doc_vector_dot(vec1, vec2):
    common = set(vec1.keys()).intersection(set(vec2.keys()))
    ret = 0.0
    for term in common:
        ret += vec1[term] * vec2[term]
    return ret


def custom_tokenize(s):
    s = s.lower()  # lower case
    s = re.sub(" \d+", " ", s)  # remove number
    tks = list(filter(lambda x: x not in set(stopwords.words('english')),
                      nltk.word_tokenize(s)))  # tokenize and filter stop words
    # PUNC = set(string.punctuation)
    # tks = list(filter(lambda x: x[0] not in PUNC, map(lamb    da x: list(x), nltk.pos_tag(tks))))  # pos tag
    # for tk in tks:
    #     tk[0] = lemmatizer.lemmatize(tk[0])  # lemmatization
    # tks = map(lambda x: x[0], tks)  # incorporate POS
    return tks


parser = argparse.ArgumentParser(description="Text Categorization Script")
parser.add_argument("TrainingSets", type=str)
parser.add_argument("TestSets", type=str)

args = parser.parse_args()

TEST_SET_PATH = args.TestSets
TRAIN_SET_PATH = args.TrainingSets

TEST_SET_CORPUS_PATH = "/".join(TEST_SET_PATH.split("/")[:-1])
TRAIN_SET_CORPUS_PATH = "/".join(TRAIN_SET_PATH.split("/")[:-1])

train_list = open(TRAIN_SET_PATH, "r")
test_list = open(TEST_SET_PATH, "r")
train_docs = list(map(lambda x: x.split(" "), map(lambda x: TRAIN_SET_CORPUS_PATH + x[1:],
                                                  filter(lambda x: x != "", train_list.read().split("\n")))))
test_docs = list(map(lambda x: TEST_SET_CORPUS_PATH + x[1:], filter(lambda x: x != "", test_list.read().split("\n"))))

N = float(len(train_docs))

ALPHA = 0.8
BETA = 0.2

tf_cnt = {}
df = {}
idf = {}
term_weights = {}
categories = {}
categories_cnt = {}
categories_inv = {}
categories_vec = {}

for doc_full in train_docs:
    doc = doc_full[0].split("/")[-1]
    sentence = open(doc_full[0], "r").read()
    doc_tokens = custom_tokenize(sentence)
    tf_cnt[doc] = {}
    for token in doc_tokens:
        if token not in tf_cnt[doc]:
            tf_cnt[doc][token] = 0.0
        tf_cnt[doc][token] = tf_cnt[doc][token] + 1
    for token in doc_tokens:
        if token not in df:
            df[token] = 0.0
        df[token] = df[token] + 1

for doc_full in train_docs:
    doc = doc_full[0].split("/")[-1]
    term_weights[doc] = {}
    for token in df.keys():
        if token in tf_cnt[doc] and token in df:
            term_weights[doc][token] = math.log(1 + tf_cnt[doc][token]) * math.log(N / df[token])
    normalize_doc(term_weights[doc])

for doc_full in train_docs:
    doc = doc_full[0].split("/")[-1]
    categories[doc] = doc_full[1]
    if doc_full[1] not in categories_cnt:
        categories_cnt[doc_full[1]] = 0.0
        categories_inv[doc_full[1]] = []
    categories_cnt[doc_full[1]] += 1
    categories_inv[doc_full[1]].append(doc)

# find category vector
for category in categories_cnt.keys():
    categories_vec[category] = {}
    for doc in categories_inv[category]:
        for token in term_weights[doc]:
            if token not in categories_vec[category]:
                categories_vec[category][token] = 0.0
            categories_vec[category][token] += term_weights[doc][token] / categories_cnt[category] * ALPHA
    for c in filter(lambda x: x != category, categories_cnt.keys()):
        for doc in categories_inv[c]:
            for token in term_weights[doc]:
                if token not in categories_vec[category]:
                    categories_vec[category][token] = 0.0
                categories_vec[category][token] -= term_weights[doc][token] / (N - categories_cnt[category]) / \
                                                   categories_cnt[c] * BETA
    normalize_doc(categories_vec[category])

f = open(f"{TEST_SET_CORPUS_PATH}/prediction.labels", "w")

for doc_full in test_docs:
    max_rocchio_dict = {}
    max_rocchio = -math.inf
    max_category = ""

    test_vec = {}
    test_vec_cnt = {}

    original_doc = "./" + "/".join(doc_full.split("/")[1:])
    doc = doc_full[0].split("/")[-1]
    sentence = open(doc_full, "r").read()
    doc_tokens = custom_tokenize(sentence)
    for token in doc_tokens:
        if token not in test_vec_cnt:
            test_vec_cnt[token] = 0.0
        test_vec_cnt[token] = test_vec_cnt[token] + 1
    for token in doc_tokens:
        test_vec[token] = math.log(1 + test_vec_cnt[token]) * math.log(N / (1 + df.get(token, 0)))
    normalize_doc(test_vec)

    for category in categories_cnt.keys():
        rocchio = doc_vector_dot(test_vec, categories_vec[category])
        if rocchio > max_rocchio:
            max_rocchio = rocchio
            max_category = category
    f.write(f"{original_doc} {max_category}\n")
f.close()
