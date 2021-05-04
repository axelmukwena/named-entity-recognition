from nltk.tag.perceptron import PerceptronTagger
from names_dataset import NameDatasetV1
import pickle
from tqdm import tqdm
import enchant
import nltk
import re


def get_features(sentence, fs, ws):
    """ Baseline Features """
    length = len(sentence)
    poses = tagger.tag(sentence)
    for i in range(length):
        features = {}
        current_word = sentence[i]
        features['has_(%s)' % current_word] = 1
        if i == 0:
            features['prev_label'] = "O"
        else:
            features['prev_label'] = get_class(fs, i - 1)

        pos = poses[i][1]
        if pos == 'NNP':
            features['proper_noun'] = 1

            if bool(re.search('[A-Z]+[a-z]+$', current_word)):
                features['first_uppercase'] = 1
            else:
                features['first_uppercase'] = 0

            if i + 1 < length and sentence[i + 1] == "'s":
                features['possession'] = 1
            else:
                features['possession'] = 0

            if m.search_first_name(current_word) or m.search_last_name(current_word):
                features['dataset'] = 1
            else:
                features['dataset'] = 0

            if not (us.check(current_word) or au.check(current_word) or gb.check(current_word) or ca.check(current_word)):
                features['foreign'] = 1
            else:
                features['foreign'] = 0
        else:
            features['proper_noun'] = 0
            features['first_uppercase'] = 0
            features['possession'] = 0
            features['dataset'] = 0
            features['foreign'] = 0

        honorifics = ['Mr', 'Ms', 'Miss', 'Mrs', 'Mx', 'Master', 'Sir', 'Madam', 'Dame', 'Lord', 'Lady', 'Dr',
                      'Prof', 'Br', 'Sr', 'Fr', 'Rev', 'Pr', 'Elder']
        if i > 0:
            previous_word = sentence[i - 1].replace('.', '')
            if previous_word in honorifics:
                features['honorific'] = 1
            else:
                features['honorific'] = 0
        else:
            features['honorific'] = 0

        fs.append(features)
        ws.append(current_word)


def get_class(features, i):
    c = classifier.classify_many(features)[i]
    p = classifier.prob_classify_many(features)[i]
    a, b, = p.prob('PERSON'), p.prob('O')
    return c


def get_sentences():
    ss = []
    with open("../data/one", "r", encoding="utf-8") as file:
        data = file.read()
        data = data.replace('\n', ' ')
        data = data.strip().split(".")
        for sentence in data:
            sentence = sentence + "."
            sentence = nltk.tokenize.word_tokenize(sentence)
            ss.append(sentence)
    return ss


us = enchant.Dict("en_US")
au = enchant.Dict("en_AU")
ca = enchant.Dict("en_CA")
gb = enchant.Dict("en_GB")

m = NameDatasetV1()
tagger = PerceptronTagger()
with open('../model.pkl', 'rb') as f:
    classifier = pickle.load(f)

sentences = get_sentences()
fts = []
wds = []
for j in tqdm(range(len(sentences))):
    get_features(sentences[j], fts, wds)

classes = classifier.classify_many(fts)
predictions = classifier.prob_classify_many(fts)

with open('../output3.txt', 'w') as f:
    f.write('  Words          P(PERSON)  P(O)   CLASS\n' + '-' * 40 + '\n')
    for (word, prediction, c) in list(zip(wds, predictions, classes)):
        fmt = '  %-15s %6.4f   %6.4f   %s'
        f.write(fmt % (word, prediction.prob('PERSON'), prediction.prob('O'), c) + "\n")
