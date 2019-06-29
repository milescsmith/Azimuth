import numpy as np
from GPy.models import WarpedGP, GPRegression
from GPy.kern import Linear, RBF, Bias
from .gpy_ssk import WeightedDegree


def gp_on_fold(feature_sets, train, test, y, y_all, learn_options):

    sequences = np.array([str(x) for x in y_all.index.get_level_values(0).tolist()])

    kern = WeightedDegree(
        1, sequences, d=learn_options["kernel degree"], active_dims=[0]
    )
    X = np.arange(len(train))[:, None]

    current_dim = 1

    if "gc_count" in feature_sets:
        kern += RBF(1, active_dims=[current_dim], name="GC_rbf")
        X = np.concatenate((X, feature_sets["gc_count"].values), axis=1)
        current_dim += 1
        assert X.shape[1] == current_dim

    if "drug" in feature_sets:
        Q = feature_sets["drug"].values.shape[1]
        kern += Linear(
            Q, active_dims=range(current_dim, current_dim + Q), name="drug_lin"
        )
        X = np.concatenate((X, feature_sets["drug"].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    if "gene effect" in feature_sets:
        Q = feature_sets["gene effect"].values.shape[1]
        kern += Linear(
            Q, active_dims=range(current_dim, current_dim + Q), name="gene_lin"
        )
        X = np.concatenate((X, feature_sets["gene effect"].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    if "Percent Peptide" in feature_sets:
        Q = feature_sets["Percent Peptide"].values.shape[1]
        kern += RBF(
            Q, active_dims=range(current_dim, current_dim + Q), name="percent_pept"
        )
        X = np.concatenate((X, feature_sets["Percent Peptide"].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    if "Nucleotide cut position" in feature_sets:
        Q = feature_sets["Nucleotide cut position"].values.shape[1]
        kern += RBF(
            Q, active_dims=range(current_dim, current_dim + Q), name="nucleo_cut"
        )
        X = np.concatenate((X, feature_sets["Nucleotide cut position"].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    if "Strand effect" in feature_sets:
        Q = feature_sets["Strand effect"].values.shape[1]
        kern += Linear(
            Q, active_dims=range(current_dim, current_dim + Q), name="strand"
        )
        X = np.concatenate((X, feature_sets["Strand effect"].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    if "NGGX" in feature_sets:
        Q = feature_sets["NGGX"].values.shape[1]
        kern += Linear(Q, active_dims=range(current_dim, current_dim + Q), name="NGGX")
        X = np.concatenate((X, feature_sets["NGGX"].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    if "TM" in feature_sets:
        Q = feature_sets["TM"].values.shape[1]
        kern += RBF(
            Q, ARD=True, active_dims=range(current_dim, current_dim + Q), name="TM"
        )
        X = np.concatenate((X, feature_sets["TM"].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    if "gene features" in feature_sets:
        Q = feature_sets["gene features"].values.shape[1]
        kern += Linear(
            Q,
            ARD=True,
            active_dims=range(current_dim, current_dim + Q),
            name="genefeat",
        )
        X = np.concatenate((X, feature_sets["gene features"].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    kern += Bias(X.shape[1])

    if learn_options["warpedGP"]:
        m = WarpedGP(X[train], y[train], kernel=kern)
    else:
        m = GPRegression(X[train], y[train], kernel=kern)

    m.optimize_restarts(3)
    y_pred, y_uncert = m.predict(X[test])

    # TODO add offset such that low scores are around 0 (not -4 or so)

    return y_pred, m[:]
