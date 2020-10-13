from models import StandardNMF, HypersurfaceNMF, L21NMF, L1RobustNMF
from datasets import load_data

def test():
    # Load ORL dataset.
    X, Y = load_data(root='../data/ORL', reduce=2)
    print('ORL dataset: X.shape = {}, Y.shape = {}'.format(X.shape, Y.shape))


    # print("Testing standard NMF")
    # standard = StandardNMF(X, n_components=len(set(Y)))
    # standard.fit(max_iter=200)

    # print()
    # print("Testing hypersurface NMF")
    # hypersurface = HypersurfaceNMF(X, n_components=len(set(Y)))
    # hypersurface.fit(max_iter=200)

    # print()
    # print("Testing L2,1 norm NMF")
    # l21 = L21NMF(X, n_components=len(set(Y)))
    # l21.fit(max_iter=200)

    print()
    print("Testing L1 robust NMF")
    # Just trying a recommended lambda from the paper
    l1robust = L1RobustNMF(X, n_components=len(set(Y)), lam=0.3)
    l1robust.fit(max_iter=200)


if __name__ == "__main__":
    test()
