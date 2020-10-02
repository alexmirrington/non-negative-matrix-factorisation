from models import MultiplicativeUpdateNMF
from datasets import load_data

def test():
    # Load ORL dataset.
    X, Y = load_data(root='../data/ORL', reduce=2)
    print('ORL dataset: X.shape = {}, Y.shape = {}'.format(X.shape, Y.shape))

    mur = MultiplicativeUpdateNMF(X, n_components=len(set(Y)))
    mur.fit(max_iter=200)


if __name__ == "__main__":
    test()
