from collections import Counter


def load_data(filename):
    words = []
    labels = []
    for line in open(filename, "r", encoding="utf-8"):
        doublet = line.strip().split("\t")
        if len(doublet) < 2:  # remove emtpy lines
            continue
        words.append(doublet[0])
        labels.append(doublet[1])
    return words, labels


path = "../data/train"
ws, ls = load_data(path)

before, before1, after, after1 = [], [], [], []
for i in range(len(ws)):
    if ls[i] == 'PERSON':
        if i > 0:
            before.append(ws[i - 1])
            print(ws[i - 1])
        # if i > 1:
        #   before1.append(ws[i - 2])
        if i + 1 < len(ws):
            after.append(ws[i + 1])
        # if i + 2 < len(ws):
        #    after1.append(ws[i + 2])

b = Counter(before).most_common(10)
# b1 = Counter(before1).most_common(10)
a = Counter(after).most_common(10)
# a1 = Counter(after1).most_common(10)
print("hey")
