from MEM import MEMM


def train():
    classifier = MEMM()
    BETA = 0.5
    MAX_ITER = 5
    BOUND = (0, 20)
    classifier.max_iter = MAX_ITER
    classifier.train()
    classifier.dump_model()


def test():
    classifier = MEMM()
    BETA = 0.5
    MAX_ITER = 5
    BOUND = (0, 20)
    try:
        classifier.load_model()
        classifier.beta = BETA
        text = classifier.test()
    except Exception as e:
        print(e)
    return text


def show():
    classifier = MEMM()
    BETA = 0.5
    MAX_ITER = 5
    BOUND = (0, 1000)
    try:
        classifier.load_model()
        text = classifier.show_samples(BOUND)
    except Exception as e:
        print(e)
    return text


def predict(name):
    classifier = MEMM()
    BETA = 0.5
    MAX_ITER = 5
    BOUND = (0, 1000)
    classifier.load_model()
    text = classifier.check(name)
    return text
