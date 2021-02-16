import pandas as pd
import magic
import networkx as nx
from tqdm import tqdm
import argparse as agp

import multiprocessing as mp
from numba import njit

import numpy as np
import pandas_flavor as pf
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau

import scprep

from scipy.stats import spearmanr as cor
import pingouin as pg
import pickle


def corr(x, y, tail="two-sided", method="pearson"):
    # Safety check
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove rows with missing values
    x, y = remove_na(x, y, paired=True)
    nx = x.size

    # Compute correlation coefficient
    if method == "pearson":
        r, pval = pearsonr(x, y)
    elif method == "spearman":
        r, pval = spearmanr(x, y)
    elif method == "kendall":
        r, pval = kendalltau(x, y)
    else:
        raise ValueError("Method not recognized.")

    return r, pval


def partial_corr(
    data=None, x=None, y=None, covar=None, tail="two-sided", method="pearson"
):
    from pingouin.utils import _flatten_list

    # Check that columns exist
    col = _flatten_list([x, y, covar, x_covar, y_covar])

    assert all([c in data for c in col]), "columns are not in dataframe."
    # Check that columns are numeric
    assert all([data[c].dtype.kind in "bfiu" for c in col])

    # Drop rows with NaN
    data = data[col].dropna()
    assert data.shape[0] > 2, "Data must have at least 3 non-NAN samples."

    # Standardize (= no need for an intercept in least-square regression)
    C = (data[col] - data[col].mean(axis=0)) / data[col].std(axis=0)

    if covar is not None:
        # PARTIAL CORRELATION
        cvar = np.atleast_2d(C[covar].to_numpy())
        beta_x = np.linalg.lstsq(cvar, C[x].to_numpy(), rcond=None)[0]
        beta_y = np.linalg.lstsq(cvar, C[y].to_numpy(), rcond=None)[0]
        res_x = C[x].to_numpy() - cvar @ beta_x
        res_y = C[y].to_numpy() - cvar @ beta_y
    else:
        # SEMI-PARTIAL CORRELATION
        # Initialize "fake" residuals
        res_x, res_y = data[x].to_numpy(), data[y].to_numpy()

    return corr(res_x, res_y, method=method, tail=tail)


def rankdata(a, method="average"):
    arr = np.ravel(np.asarray(a))
    algo = "quicksort"
    sorter = np.argsort(arr, kind=algo)

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]

    # cumulative counts of each unique value
    count = np.r_[np.nonzero(obs)[0], len(obs)]

    # average method
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def all_res(data=None, genes=[], covar=[]):
    col = genes + covar
    C = (data[col] - data[col].mean(axis=0)) / data[col].std(axis=0)
    cvar = np.atleast_2d(C[covar].to_numpy())
    Cres = C.copy()

    for g in genes:
        beta = np.linalg.lstsq(cvar, C[g].to_numpy(), rcond=None)[0]
        res = C[g].to_numpy() - cvar @ beta
        rk = rankdata(res)
        rk_norm = (rk - rk.mean()) / rk.std()
        Cres[g] = rk_norm

    return Cres


@njit
def compute_cor(row1: np.ndarray, row2: np.ndarray, std1: float, std2: float):
    return np.sum(row1 * row2) / (std1 * std2)


def wrap_compute_cor(T):
    row1, row2, g1, g2, std1, std2 = T
    return (g1, g2, compute_cor(row1, row2, std1, std2))


def all_partial_corr(data=None, genes=[], covar=[]):
    Cres = all_res(data=data, genes=genes, covar=covar)
    stds = {g: np.sqrt(np.sum(Cres[g] ** 2)) for g in genes}

    def sample_generator():
        # require mp
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                row1 = Cres[genes[i]].to_numpy()
                row2 = Cres[genes[j]].to_numpy()
                yield (row1, row2, genes[i], genes[j], stds[genes[i]], stds[genes[j]])

    with mp.Pool(processes=4) as p:
        res = [
            x
            for x in p.imap_unordered(
                wrap_compute_cor, sample_generator(), chunksize=16
            )
        ]

    return res


"""
General idea is to check recursively a new node with potential neighbours
If neighbours are kept, we try to "kill" the triangle (if a--b and we add c that a--c and b--c,
then try to remove them)
"""


def pcor(gene1, gene2, opponent="", data=None):
    r = pg.partial_corr(
        data=data, x=gene1, y=gene2, covar=["total", opponent], method="spearman"
    ).r.values[0]
    return r


# @jit(forceobj=True)
def tfomer(data, genes, cors, verbose=True):
    """
    data is a pandas.DataFrame
    genes is a list of str
    cors is a PDict
    """
    # INIT
    abs_values = np.abs(np.array(list(cors.values())))
    N = max(10, len(genes))
    if N > 50:
        elagate = True
    else:
        elagate = False
    p1 = (1 - (2 / np.log(N))) * 100
    p2 = (1 - (1 / np.log(N))) * 100
    if verbose:
        print("Percentiles : ", p1, p2, len(genes))
        print("Elagate is ", "on" if elagate else "off")
    down_thr_1 = np.percentile(abs_values, p1) if elagate else 0
    down_thr_2 = np.percentile(abs_values, p2) if elagate else 0
    if verbose:
        print("Thresholds :", down_thr_1, down_thr_2)
        print()
    G = nx.Graph()

    # RECURSION
    for gene in tqdm(genes):
        # print("handling", gene)
        # look for all potential neighbour in the graph
        ngbs = list()
        for node in G.nodes:
            if abs(cors[(node, gene)]) > down_thr_2:
                ngbs.append(node)

        # add the gene to the graph
        G.add_node(gene)

        bag_of_ngbs = ngbs.copy()
        # look for triangles
        while len(bag_of_ngbs) > 0:
            ngb = bag_of_ngbs.pop(0)
            ngbs_n = set(G.neighbors(ngb))
            inter = ngbs_n.intersection(bag_of_ngbs)

            # handle
            ok = True
            mincor = cors[(gene, ngb)]
            # with tests, we shall keep the lowest pcor and check for sign
            for third in inter:
                # cut is cut
                # calc
                old_edge_pcor = pcor(third, ngb, opponent=gene, data=data)
                old_edge_cor = cors[(third, ngb)]
                # print(gene, ngb, third, old_edge_pcor, old_edge_cor)
                # test
                if old_edge_cor * old_edge_pcor < 0 or abs(old_edge_pcor) < down_thr_1:
                    # print("remove", third, ngb)
                    G.remove_edge(third, ngb)
                else:
                    G.edges[(third, ngb)]["w"] = min(
                        G.edges[(third, ngb)]["w"], old_edge_pcor, key=abs
                    )

                # remove from bag
                third_pcor = pcor(third, gene, opponent=ngb, data=data)
                third_cor = cors[(third, gene)]
                # print(">", ngb, gene, third, third_pcor, third_cor)
                if third_cor * third_pcor < 0 or abs(third_pcor) < down_thr_1:
                    bag_of_ngbs.remove(third)

                # not ok is not ok
                new_pcor = pcor(gene, ngb, opponent=third, data=data)
                new_cor = cors[(gene, ngb)]
                # print(">>", gene, ngb, third, new_pcor, new_cor)
                if new_cor * new_pcor < 0 or abs(new_pcor) < down_thr_1:
                    ok = False
                    break
                else:
                    mincor = min(mincor, new_pcor, key=abs)

            if ok:
                # print("add", gene, ngb)
                G.add_edge(gene, ngb, w=mincor)

    return G


def print_dict(d):
    for k, v in d.items():
        print(k, ":", v)


class PDict(dict):
    def __init__(self, d):
        super().__init__({(min(k), max(k)): v for k, v in d.items()})

    def __getitem__(self, key):
        return dict.__getitem__(self, (min(key), max(key)))


################################################################
###################        MAIN PIPELINE      ##################
################################################################


def read_data(filepath):
    print("Reading data")
    df = pd.read_csv(filepath, index_col=0).transpose()
    return df


def compute_fastcorr(data):
    """
    data shall be a pandas DataFrame
    """

    magic_op = magic.MAGIC(t=1)
    # we need to define when using magic or not
    # data_magic = data.copy()
    data_magic = magic_op.fit_transform(data)

    genes = list(data_magic.columns)
    col = data.columns
    data_magic[col] -= data_magic[col].mean(axis=0)
    data_magic[col] /= data_magic[col].std(axis=0)

    data_magic["total"] = data_magic.mean(axis=1)

    print("Calculating normal correlations")
    cors = all_partial_corr(data_magic, genes=genes, covar=[])
    # cors = all_partial_corr(data_magic, genes=genes, covar=["total"])
    cors = PDict({(x[0], x[1]): x[2] for x in cors})
    print_dict(cors)

    print("Building network iteratively")
    D = tfomer(data_magic, genes, cors)

    return D

def save_data(network, filename="FastCorr.out"):
    print("Saving data")
    genes1, genes2, types = [], [], []
    for u, v, d in network.edges(data=True):
        genes1.append(u)
        genes2.append(v)
        # types.append("+" if d["w"] > 0 else "-")
        types.append(abs(d["w"]))

    df = pd.DataFrame({"Gene1": genes1, "Gene2": genes2, "EdgeWeight": types})
    df = df.sort_values("EdgeWeight", ascending=False)
    df.to_csv(filename, sep="\t", index=False)
    return df


if __name__ == "__main__":
    # arg parse here
    parser = agp.ArgumentParser()
    parser.add_argument("-i", "--input", default="input.csv")
    parser.add_argument("-o", "--output", default="output.csv")
    args = parser.parse_args()
    print(args)
    data = read_data(args.input)
    net = compute_fastcorr(data)
    save_data(net, filename=args.output)
