import glob
import os
import pickle
import warnings

import azimuth.models.ensembles as ensembles
import Bio.Seq as Seq
import Bio.SeqUtils as SeqUtil
import Bio.SeqUtils.MeltingTemp as Tm

import numpy as np
import pandas as pd

import scipy as sp
import scipy.stats
import scipy.stats as st
import sklearn.metrics

from . import metrics as ranking_metrics
from azimuth.load_data import combine_organisms


def estimate_lambda(pv):
    """
    estimate the lambda for a given array of P-values
    ------------------------------------------------------------------
    pv          numpy array containing the P-values
    ------------------------------------------------------------------
    L           lambda value
    ------------------------------------------------------------------
    """
    LOD2 = sp.median(st.chi2.isf(pv, 1))
    L = LOD2 / 0.456
    return L


def get_pval_from_predictions(
    m0_predictions, m1_predictions, ground_truth, twotailed=False, method="steiger"
):
    """
    If twotailed==False, then need to check that the one of corr0 and corr1 that is higher is the correct one
    """
    from . import corrstats

    n0 = len(m0_predictions)
    n1 = len(m1_predictions)
    n2 = len(ground_truth)
    assert n0 == n1
    assert n0 == n2
    corr0, _ = scipy.stats.spearmanr(m0_predictions, ground_truth)
    corr1, _ = scipy.stats.spearmanr(m1_predictions, ground_truth)
    corr01, _ = scipy.stats.spearmanr(m0_predictions, m1_predictions)
    t2, pv = corrstats.dependent_corr(
        corr0, corr1, corr01, n0, twotailed=twotailed, method=method
    )
    return t2, pv, corr0, corr1, corr01


def get_thirty_one_mer_data():
    """
    Load up our processed data file for all of V1 and V2, make a 31mer so that
    we can use the SSC trained model to compare to
    Assumes we call this from the analysis subdirectory
    """
    myfile = r"..\data\FC_RES_5304.csv"
    newfile = r"..\data\FC_RES_5304_w_31mer.csv"
    data = pd.read_csv(myfile)
    thirty_one_mer = []
    for i in range(data.shape[0]):
        thirty_one_mer.append(
            convert_to_thirty_one(
                data.iloc[i]["30mer"], data.iloc[i]["Target"], data.iloc[i]["Strand"]
            )
        )
    data["31mer"] = thirty_one_mer
    data.to_csv(newfile)


def guide_positional_features(guide_seq, gene, strand):
    """
    Given a guide sequence, a gene name, and strand (e.g. "sense"), return the (absolute) nucleotide cut position,
    and the percent amino acid.
    From John's email:
    the cut site is always 3nts upstream of the NGG PAM:
    5' - 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 <cut> 18 19 20 N G G - 3'
    To calculate percent protein, we determined what amino acid number was being cut and just divided by the total
    number of amino acids. In the case where the cutsite was between two amino acid codons, I believe we rounded down

    """

    guide_seq = Seq.Seq(guide_seq)
    gene_seq = Seq.Seq(get_gene_sequence(gene)).reverse_complement()
    if strand == "sense":
        guide_seq = guide_seq.reverse_complement()
    ind = gene_seq.find(guide_seq)
    if ind == -1:
        print(f"returning None, could not find guide {guide_seq} in gene {gene}")
        return ""
    assert gene_seq[ind : (ind + len(guide_seq))] == guide_seq, "match not right"
    ## now get what we want from this:
    import pdb

    pdb.set_trace()
    raise NotImplementedError("incomplete implemented for now")


def convert_to_thirty_one(guide_seq, gene, strand):
    """
    Given a guide sequence, a gene name, and strand (e.g. "sense"), return a 31mer string which is our 30mer,
    plus one more at the end.
    """
    guide_seq = Seq.Seq(guide_seq)
    gene_seq = Seq.Seq(get_gene_sequence(gene)).reverse_complement()
    if strand == "sense":
        guide_seq = guide_seq.reverse_complement()
    ind = gene_seq.find(guide_seq)
    if ind == -1:
        print(
            f"returning sequence+'A', could not find guide {guide_seq} in gene {gene}"
        )
        return gene_seq + "A"
    assert gene_seq[ind : (ind + len(guide_seq))] == guide_seq, "match not right"
    new_mer = gene_seq[(ind - 1) : (ind + len(guide_seq))]
    # this actually tacks on an extra one at the end for some reason
    if strand == "sense":
        new_mer = new_mer.reverse_complement()
    return str(new_mer)


def concatenate_feature_sets(feature_sets, keys=None):
    """
    Given a dictionary of sets of features, each in a pd.DataFrame,
    concatenate them together to form one big np.array, and get the dimension
    of each set
    Returns: inputs, dim
    """
    assert feature_sets != {}, "no feature sets present"
    if keys is None:
        keys = list(feature_sets.keys())

    F = feature_sets[keys[0]].shape[0]
    for set in list(feature_sets.keys()):
        F2 = feature_sets[set].shape[0]
        assert F == F2, "not same # individuals for features %s and %s" % (keys[0], set)

    N = feature_sets[keys[0]].shape[0]
    inputs = np.zeros((N, 0))
    feature_names = []
    dim = {}
    dimsum = 0
    for set in keys:
        inputs_set = feature_sets[set].values
        dim[set] = inputs_set.shape[1]
        dimsum = dimsum + dim[set]
        inputs = np.hstack((inputs, inputs_set))
        feature_names.extend(feature_sets[set].columns.tolist())

    # print "final size of inputs matrix is (%d, %d)" % inputs.shape
    return inputs, dim, dimsum, feature_names


def extract_individual_level_data(one_result):
    """
    Extract predictions and truth for each fold
    Returns: ranks, predictions

    assumes that results here is the value for a results dictionary for one key, i.e. one entry in a dictionary loaded
    up from saved results with pickle e.g. all_results, all_learn_options = pickle.load(some_results_file)
    then call extract_individual_level_data(one_results = all_results['firstkey'])
    then, one_results contains: metrics, gene_pred, fold_labels, m, dimsum, filename, feature_names
    """
    metrics, gene_pred, fold_labels, m, dimsum, filename, feature_names = one_result
    all_true_ranks = np.empty(0)
    all_pred = np.empty(0)
    for f in list(fold_labels):
        these_ranks = gene_pred[0][0][f]["ranks"]  # similar for thrs
        these_pred = gene_pred[0][1][f]
        all_true_ranks = np.concatenate((all_true_ranks, these_ranks))
        all_pred = np.concatenate((all_pred, these_pred))
    return all_true_ranks, all_pred


def spearmanr_nonan(x, y):
    """
    same as scipy.stats.spearmanr, but if all values are equal, returns 0 instead of nan
    (Output: rho, pval)
    """
    r, p = st.spearmanr(x, y)
    r = np.nan_to_num(r)
    p = np.nan_to_num(p)
    return r, p


def impute_gene_position(gene_position):
    """
    Some amino acid cut position and percent peptide are blank because of stop codons, but
    we still want a number for these, so just set them to 101 as a proxy
    """

    gene_position["Percent Peptide"] = gene_position["Percent Peptide"].fillna(101.00)

    if "Amino Acid Cut position" in gene_position.columns:
        gene_position["Amino Acid Cut position"] = gene_position[
            "Amino Acid Cut position"
        ].fillna(gene_position["Amino Acid Cut position"].mean())

    return gene_position


def datestamp(appendrandom=False):
    import datetime

    now = datetime.datetime.now()
    s = str(now)[:19].replace(" ", "_").replace(":", "_")
    if appendrandom:
        import random

        s += "_" + str(random.random())[2:]
    return s


def get_gene_sequence(gene_name):
    try:
        gene_file = "../../gene_sequences/%s_sequence.txt" % gene_name
        # gene_file = '../gene_sequences/%s_sequence.txt' % gene_name
        # gene_file = 'gene_sequences/%s_sequence.txt' % gene_name
        with open(gene_file, "rb") as f:
            seq = f.read()
            seq = seq.replace("\r\n", "")
    except:
        raise Exception(
            f"could not find gene sequence file {gene_file}, please see examples and generate one for your gene "
            f"as needed, with this filename"
        )

    return seq


def target_genes_stats(
    genes=("HPRT1", "TADA1", "NF2", "TADA2B", "NF1", "CUL3", "MED12", "CCDC101")
):
    for gene in genes:
        seq = get_gene_sequence(gene)
        if seq != None:
            print(
                f"{gene} \t\t\t\t "
                f"len: {len(seq)} \t "
                f"GCcont: {SeqUtil.GC(seq):.3f} \t "
                f"Temp: {Tm.Tm_staluc(seq, rna=False):.4f} \t "
                f"molweight: {SeqUtil.molecular_weight(seq, 'DNA'):.4f}"
            )


def ranktrafo(data):
    X = data.values[:, None]
    Is = X.argsort(axis=0)
    RV = sp.zeros_like(X)
    rank = sp.zeros_like(X)
    for i in range(X.shape[1]):
        x = X[:, i]
        rank = sp.stats.rankdata(x)
        rank /= X.shape[0] + 1
        RV[:, i] = sp.sqrt(2) * sp.special.erfinv(2 * rank - 1)

    return RV.flatten()


def get_ranks(y, thresh=0.8, prefix="", flip=False, col_name="score"):
    """
    y should be a DataFrame with one column
    thresh is the threshold at which to call it a knock-down or not
    col_name = 'score' is only for V2 data
    flip should be FALSE for both V1 and V2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """

    if prefix is not None:
        prefix = prefix + "_"

    # y_rank = y.apply(ranktrafo)
    y_rank = y.apply(sp.stats.mstats.rankdata)
    y_rank /= y_rank.max()

    if flip:
        y_rank = (
            1.0 - y_rank
        )  # before this line, 1-labels where associated with low ranks, this flips it around
        # (hence the y_rank > thresh below)
        # we should NOT flip (V2), see README.txt in ./data

    y_rank.columns = [prefix + "rank"]
    y_threshold = (y_rank > thresh) * 1

    y_threshold.columns = [prefix + "threshold"]

    # JL: undo the log2 transform (not sure this matters?)
    y_rank_raw = (2 ** y).apply(scipy.stats.mstats.rankdata)
    y_rank_raw /= y_rank_raw.max()
    if flip:
        y_rank_raw = 1.0 - y_rank_raw
    y_rank_raw.columns = [prefix + "rank raw"]
    assert ~np.any(np.isnan(y_rank)), "found NaN ranks"

    y_quantized = y_threshold.copy()
    y_quantized.columns = [prefix + "quantized"]

    return y_rank, y_rank_raw, y_threshold, y_quantized


def get_data(data, y_names, organism="human", target_gene=None):
    outputs = pd.DataFrame()
    """
    this is called once for each gene (aggregating across cell types)
    y_names are cell types
    e.g. call: X_CD13, Y_CD13 = get_data(cd13, y_names=['NB4 CD13', 'TF1 CD13'])
    """

    # generate ranks for each cell type before aggregating to match what is in Doench et al
    thresh = 0.8
    for y_name in y_names:  # for each cell type
        y = pd.DataFrame(data[y_name])
        # these thresholds/quantils are not used:
        y_rank, y_rank_raw, y_threshold, y_quantiles = get_ranks(
            y, thresh=thresh, flip=False, col_name=y_name
        )
        y_rank.columns = [y_name + " rank"]
        y_rank_raw.columns = [y_name + " rank raw"]
        y_threshold.columns = [y_name + " threshold"]

        outputs = pd.concat([outputs, y, y_rank, y_threshold, y_rank_raw], axis=1)

    # aggregated rank across cell types
    average_activity = pd.DataFrame(outputs[[y_name for y_name in y_names]].mean(1))
    average_activity.columns = ["average activity"]

    average_rank_from_avg_activity = get_ranks(
        average_activity, thresh=thresh, flip=False, col_name="average activity"
    )[0]
    average_rank_from_avg_activity.columns = ["average_rank_from_avg_activity"]
    average_threshold_from_avg_activity = (average_rank_from_avg_activity > thresh) * 1
    average_threshold_from_avg_activity.columns = [
        "average_threshold_from_avg_activity"
    ]

    average_rank = pd.DataFrame(
        outputs[[y_name + " rank" for y_name in y_names]].mean(1)
    )
    average_rank.columns = ["average rank"]
    # higher ranks are better (when flip=False as it should be)
    average_threshold = (average_rank > thresh) * 1
    average_threshold.columns = ["average threshold"]

    # undo the log2 trafo on the reads per million, apply rank trafo right away
    average_rank_raw = pd.DataFrame(
        outputs[[y_name + " rank raw" for y_name in y_names]].mean(1)
    )
    average_rank_raw.columns = ["average rank raw"]
    outputs = pd.concat(
        [
            outputs,
            average_rank,
            average_threshold,
            average_activity,
            average_rank_raw,
            average_rank_from_avg_activity,
            average_threshold_from_avg_activity,
        ],
        axis=1,
    )

    # import pdb; pdb.set_trace()

    # sequence-specific computations
    # features = featurize_data(data)
    # strip out featurization to later
    features = pd.DataFrame(data["30mer"])

    if organism is "human":
        target_gene = y_names[0].split(" ")[1]

    outputs["Target gene"] = target_gene
    outputs["Organism"] = organism

    features["Target gene"] = target_gene
    features["Organism"] = organism
    features["Strand"] = pd.DataFrame(data["Strand"])

    return features, outputs


def create_cachedir(dirname="./cache/default"):
    if os.path.exists(dirname):
        return dirname
    else:
        os.makedirs(dirname)
        return dirname


def dcg(relevances, rank=20):
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.0
    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)


def ndcgk(relevances, rank=20):
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.0
    return dcg(relevances, rank) / best_dcg


def extract_feature_from_model(method, results, split):
    model_type = results[method][3][split]
    if isinstance(model_type, sklearn.linear_model.coordinate_descent.ElasticNet):
        tmp_imp = results[method][3][split].coef_[:, None]
    elif isinstance(model_type, sklearn.ensemble.GradientBoostingRegressor):
        tmp_imp = results[method][3][split].feature_importances_[:, None]
    else:
        raise Exception("need to add model %s to feature extraction" % model_type)
    return tmp_imp


def extract_feature_from_model_sum(method, results, split, indexes):
    model_type = results[method][3][split]
    if isinstance(model_type, sklearn.linear_model.coordinate_descent.ElasticNet):
        tmp_imp = np.sum(results[method][3][split].coef_[indexes])
    elif isinstance(model_type, sklearn.ensemble.GradientBoostingRegressor):
        tmp_imp = np.sum(results[method][3][split].feature_importances_[indexes])
    else:
        raise Exception("need to add model %s to feature extraction" % model_type)
    return tmp_imp


def feature_importances(results, fontsize=16, figsize=(14, 8)):
    for method in list(results.keys()):
        feature_names = results[method][6]

        seen = set()
        uniq = []
        for ft in feature_names:
            if ft not in seen:
                uniq.append(ft)
            else:
                seen.add(ft)
        if len(seen) > 0:
            raise Exception(f"feature name appears more than once: {seen}")

        pd_order1, pi_order1, pd_order2, pi_order2, nggx = [], [], [], [], []
        for i, s in enumerate(feature_names):
            if "False" in s:
                continue
            elif "_" in s:
                nucl, pos = s.split("_")
                if len(nucl) == 1:
                    pd_order1.append(i)
                elif len(nucl) == 2:
                    pd_order2.append(i)
            elif "NGGX_pd.Order2" in s:
                nggx.append(i)
            else:
                nucl = s
                if len(nucl) == 1:
                    pi_order1.append(i)
                elif len(nucl) == 2:
                    pi_order2.append(i)

        grouped_feat = {
            "pd_order2": pd_order2,
            "pi_order2": pi_order2,
            "pd_order1": pd_order1,
            "pi_order1": pi_order1,
            "NGGX_pd.Order2": nggx,
        }

        grouped_feat_ind = []
        [grouped_feat_ind.extend(grouped_feat[a]) for a in list(grouped_feat.keys())]
        remaining_features_ind = set.difference(
            set(range(len(feature_names))), set(grouped_feat_ind)
        )

        for i in remaining_features_ind:
            grouped_feat[feature_names[i]] = [i]

        feature_importances_grouped = {}
        for k in grouped_feat:
            if len(grouped_feat[k]) == 0:
                continue
            else:
                for split in list(results[method][3].keys()):
                    split_feat_importance = extract_feature_from_model_sum(
                        method, results, split, grouped_feat[k]
                    )
                    if k not in feature_importances_grouped:
                        feature_importances_grouped[k] = [split_feat_importance]
                    else:
                        feature_importances_grouped[k].append(split_feat_importance)

        all_split_importances = None
        for split in list(results[method][3].keys()):

            split_feat_importance = extract_feature_from_model(method, results, split)

            if all_split_importances is None:
                all_split_importances = split_feat_importance.copy()
            else:
                all_split_importances = np.append(
                    all_split_importances, split_feat_importance, axis=1
                )

        avg_importance = np.mean(all_split_importances, axis=1)[:, None]
        std_importance = np.std(all_split_importances, axis=1)[:, None]
        imp_array = np.concatenate(
            (np.array(feature_names)[:, None], avg_importance, std_importance), axis=1
        )

        df = pd.DataFrame(
            data=imp_array,
            columns=["Feature name", "Mean feature importance", "Std. Dev."],
        )
        df = df.convert_objects(convert_numeric=True)

        feature_dictionary = {
            "pd_order2": "position dep. order 2 ",
            "pd_order1": "position dep. order 1 ",
            "pi_order1": "position ind. order 1 ",
            "pi_order2": "position ind. order 2 ",
            "5mer_end_False": "Tm (5mer end)",
            "5mer_start_False": "Tm (5mer start)",
            "Amino Acid Cut position": "amino acid cut position ",
            "8mer_middle_False": "Tm (8mer middle)",
            "NGGX_pd.Order2": "NGGN interaction ",
            "Tm global_False": "Tm (30mer)",
            "Percent Peptide": "percent peptide ",
        }

        for i in range(df.shape[0]):
            thisfeat = df["Feature name"].iloc[i]
            if thisfeat in feature_dictionary.keys():
                df["Feature name"].iloc[i] = feature_dictionary[thisfeat]

        return df


def check_learn_options_set(learn_options_set):
    if learn_options_set is None:
        return "ranks"

    non_binary_target_name = None
    for l in list(learn_options_set.values()):
        if non_binary_target_name is None:
            non_binary_target_name = l["testing_non_binary_target_name"]
        else:
            assert non_binary_target_name == l["testing_non_binary_target_name"], (
                "need to have same "
                "testing_non_binary_target_name "
                "across all learn options in a set "
                "for metrics to be comparable"
            )
    return non_binary_target_name


def get_all_metrics(
    results,
    learn_options_set=None,
    test_metrics=["spearmanr"],
    add_extras=False,
    force_by_gene=False,
):
    """
    'metrics' here are the metrics used to evaluate
    """
    all_results = dict([(k, {}) for k in list(results.keys())])
    genes = list(results[list(results.keys())[0]][1][0][0].keys())

    for metric in test_metrics:
        for method in list(all_results.keys()):
            all_results[method][metric] = []

    non_binary_target_name = check_learn_options_set(learn_options_set)

    for method in list(results.keys()):
        truth, predictions = results[method][1][0]
        test_indices = results[method][-1]
        tmp_genes = list(results[method][1][0][0].keys())
        if len(tmp_genes) != len(tmp_genes) or np.any(tmp_genes == genes):
            "genes have changed, need to modify code"
        all_truth_raw, all_truth_thrs, all_predictions = (
            np.array([]),
            np.array([]),
            np.array([]),
        )

        fpr_gene = {}
        tpr_gene = {}
        y_truth_thresh_all = np.array([])
        y_pred_all = np.array([])

        for gene in genes:
            y_truth, y_pred = truth[gene], predictions[gene]
            all_truth_raw = np.append(all_truth_raw, y_truth[non_binary_target_name])
            all_truth_thrs = np.append(all_truth_thrs, y_truth["thrs"])
            all_predictions = np.append(all_predictions, y_pred)

            y_truth_thresh_all = np.append(y_truth_thresh_all, y_truth["thrs"])
            y_pred_all = np.append(y_pred_all, y_pred)

            if "spearmanr" in test_metrics:
                spearmanr = np.nan_to_num(
                    st.spearmanr(y_truth[non_binary_target_name], y_pred)[0]
                )
                all_results[method]["spearmanr"].append(spearmanr)

            if "spearmanr>2.5" in test_metrics:
                selected = y_truth[non_binary_target_name] > 1.0
                # spearmanr = sp.stats.spearmanr(y_truth[non_binary_target_name][selected], y_pred[selected])[0]
                spearmanr = np.sqrt(
                    np.mean(
                        (y_truth[non_binary_target_name][selected] - y_pred[selected])
                        ** 2
                    )
                )
                all_results[method]["spearmanr>2.5"].append(spearmanr)

            if "RMSE" in test_metrics:
                rmse = np.sqrt(np.mean((y_truth[non_binary_target_name] - y_pred) ** 2))
                all_results[method]["RMSE"].append(rmse)

            if "NDCG@5" in test_metrics:
                ndcg = ranking_metrics.ndcg_at_k_ties(
                    y_truth[non_binary_target_name], y_pred, 5
                )
                all_results[method]["NDCG@5"].append(ndcg)

            if "NDCG@10" in test_metrics:
                ndcg = ranking_metrics.ndcg_at_k_ties(
                    y_truth[non_binary_target_name], y_pred, 10
                )
                all_results[method]["NDCG@10"].append(ndcg)

            if "NDCG@20" in test_metrics:
                ndcg = ranking_metrics.ndcg_at_k_ties(
                    y_truth[non_binary_target_name], y_pred, 20
                )
                all_results[method]["NDCG@20"].append(ndcg)

            if "NDCG@50" in test_metrics:
                ndcg = ranking_metrics.ndcg_at_k_ties(
                    y_truth[non_binary_target_name], y_pred, 50
                )
                all_results[method]["NDCG@50"].append(ndcg)

            if "precision@5" in test_metrics:
                y_top_truth = (
                    y_truth[non_binary_target_name]
                    >= np.sort(y_truth[non_binary_target_name])[::-1][:5][-1]
                ) * 1
                y_top_pred = (y_pred >= np.sort(y_pred)[::-1][:5][-1]) * 1
                all_results[method]["precision@5"].append(
                    sklearn.metrics.precision_score(y_top_pred, y_top_truth)
                )

            if "precision@10" in test_metrics:
                y_top_truth = (
                    y_truth[non_binary_target_name]
                    >= np.sort(y_truth[non_binary_target_name])[::-1][:10][-1]
                ) * 1
                y_top_pred = (y_pred >= np.sort(y_pred)[::-1][:10][-1]) * 1
                all_results[method]["precision@10"].append(
                    sklearn.metrics.precision_score(y_top_pred, y_top_truth)
                )

            if "precision@20" in test_metrics:
                y_top_truth = (
                    y_truth[non_binary_target_name]
                    >= np.sort(y_truth[non_binary_target_name])[::-1][:20][-1]
                ) * 1
                y_top_pred = (y_pred >= np.sort(y_pred)[::-1][:20][-1]) * 1
                all_results[method]["precision@20"].append(
                    sklearn.metrics.precision_score(y_top_pred, y_top_truth)
                )

            if "AUC" in test_metrics:
                fpr_gene[gene], tpr_gene[gene], _ = sklearn.metrics.roc_curve(
                    y_truth["thrs"], y_pred
                )
                auc = sklearn.metrics.auc(fpr_gene[gene], tpr_gene[gene])
                all_results[method]["AUC"].append(auc)

    if add_extras:
        fpr_all, tpr_all, _ = sklearn.metrics.roc_curve(y_truth_thresh_all, y_pred_all)
        return all_results, genes, fpr_all, tpr_all, fpr_gene, tpr_gene
    else:
        return all_results, genes


def load_results(
    directory, all_results, all_learn_options, model_filter=None, append_to_key=None
):
    """
    Only load up files which contain one of the strings in model_filter in their names
    model_filter should be a list, or a string
    """
    num_added = 0
    filelist = glob.glob(directory + "\\*.pickle")
    if filelist == []:
        raise Exception("found no pickle files in %s" % directory)
    else:
        print()
        "found %d files in %s" % (len(filelist), directory)

    for results_file in filelist:
        if "learn_options" in results_file:
            continue

        if model_filter != None:
            if isinstance(model_filter, list):
                in_filt = False
                for m in model_filter:
                    if m in results_file:
                        in_filt = True
                if not in_filt:
                    print(f"{results_file} not in model_filter")
                    continue
            elif model_filter not in results_file:
                continue

        try:
            with open(results_file, "rb") as f:
                results, learn_options = pickle.load(f)
            gene_names = None
        except:
            with open(results_file, "rb") as f:
                # this is when I accidentally saved from the plotting routine and should not generally be needed
                results, learn_options, gene_names = pickle.load(f)

        for k in list(results.keys()):
            if append_to_key is not None:
                k_new = k + "_" + append_to_key
            else:
                k_new = k
            assert k_new not in list(all_results.keys()), "found %s already" % k
            print(f"adding key {k_new} (from file {os.path.split(results_file)[-1]})")
            all_results[k_new] = results[k]
            all_learn_options[k_new] = learn_options[k]
            num_added = num_added + 1

    if num_added == 0:
        raise Exception("found no files to add from dir=%s" % directory)

    return all_results, all_learn_options


def plot_cluster_results(
    metrics=["spearmanr", "NDCG@5"],
    plots=["boxplots"],
    directory=r"\\fusi1\crispr2\analysis\cluster\results",
    results=None,
    learn_options=None,
    filter=None,
):
    all_results = {}
    all_learn_options = {}

    if results is None:
        if type(directory) == list:
            for exp_dir in directory:
                all_results, all_learn_options = load_results(
                    exp_dir, all_results, all_learn_options, filter
                )
        else:
            all_results, all_learn_options = load_results(
                directory, all_results, all_learn_options, filter
            )

    else:
        for k in list(results.keys()):
            assert k not in list(all_results.keys())
            all_results[k] = results[k]
            all_learn_options[k] = learn_options[k]

    all_metrics, gene_names = get_all_metrics(all_results, test_metrics=metrics)
    # plot_all_metrics(all_metrics, gene_names, all_learn_options, plots=plots, save=False)


def ensemble_cluster_results(
    directory=r"\\fusi1\crispr2\analysis\cluster\results\cluster_experiment_izf_ob",
    ensemble_type="median",
    models_to_ensemble=["all"],
):
    all_results = {}
    all_learn_options = {}

    for results_file in glob.glob(directory + "\\*.pickle"):
        if "learn_options" in results_file:
            continue

        with open(results_file, "rb") as f:
            results, learn_options = pickle.load(f)

        for k in list(results.keys()):
            assert k not in list(all_results.keys())
            all_results[k] = results[k]
            all_learn_options[k] = learn_options[k]

    genes = list(all_results[list(all_results.keys())[0]][1][0][0].keys())
    models = list(all_results.keys())

    ens_predictions = {}
    ens_truths = {}
    for g, gene in enumerate(genes):
        test_predictions = None
        cv_predictions = None
        cv_truth = None

        prev_model_truth = None
        for i, model in enumerate(models):
            if len([m for m in models_to_ensemble if m in model]) == 0:
                continue

            truth, predictions = all_results[model][1][0]

            if test_predictions == None:
                test_predictions = predictions[gene][:, None]
            else:
                test_predictions = np.append(
                    test_predictions, predictions[gene][:, None], axis=1
                )

            # this is just to check that all the models are using the same ordering of
            # the ground truth and hence of the samples, as this might mess up the ensemble.
            if prev_model_truth is not None:
                assert np.all(truth[gene]["ranks"] == prev_model_truth)
            else:
                prev_model_truth = truth[gene]["ranks"]

            # take all the other genes and stack the predictions under a given model.
            cv_predictions_gene_j = np.array([])
            cv_truth_gene_j = np.array([])
            for other_gene in genes:
                if gene == other_gene:
                    continue
                cv_predictions_gene_j = np.append(
                    cv_predictions_gene_j, predictions[other_gene]
                )
                cv_truth_gene_j = np.append(cv_truth_gene_j, truth[other_gene]["ranks"])

            if cv_truth is None:
                cv_truth = cv_truth_gene_j.copy()[:, None]

            if cv_predictions is None:
                cv_predictions = cv_predictions_gene_j[:, None]
            else:
                cv_predictions = np.append(
                    cv_predictions, cv_predictions_gene_j[:, None], axis=1
                )

        if ensemble_type is "majority":
            y_pred = ensembles.pairwise_majority_voting(test_predictions)
        if ensemble_type is "median":
            y_pred = ensembles.median(test_predictions)
        if ensemble_type is "stacking":
            y_pred = ensembles.linear_stacking(
                cv_truth, cv_predictions, test_predictions
            )

        ens_predictions[gene] = y_pred
        ens_truths[gene] = truth[gene]

    all_results[ensemble_type] = [None, [[ens_truths, ens_predictions]], None, None]
    all_learn_options[ensemble_type] = None

    return all_results, all_learn_options


if __name__ == "__main__":
    get_thirty_one_mer_data()
    import pdb

    pdb.set_trace()

    V = "1"
    if V == "1":
        human_data = pd.read_excel("data/V1_data.xlsx", sheetname=0, index_col=[0, 1])
        mouse_data = pd.read_excel("data/V1_data.xlsx", sheetname=1, index_col=[0, 1])
        X, Y = combine_organisms()
        X.to_pickle("../data/X.pd")  # sequence features (i.e. inputs to prediction)
        Y.to_pickle(
            "../data/Y.pd"
        )  # cell-averaged ranks, plus more (i.e. possible targets for prediction)
        print()
        "done writing to file"
    elif V == "2":
        # this is now all in predict.py
        pass
    elif V == "0":
        pass
