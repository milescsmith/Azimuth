import glob
import os
import pickle
import warnings

# import azimuth
# import azimuth.models
# import azimuth.models.ensembles as ensembles
import Bio.Seq as Seq
import Bio.SeqUtils as SeqUtil
import Bio.SeqUtils.MeltingTemp as Tm
# import matplotlib.pylab as plt
import numpy as np
import pandas as pd
# import pylab as pl  # so can just grab qqplotting code from fastlmm directly
import scipy as sp
import scipy.stats
import scipy.stats as st
import sklearn.metrics

from . import metrics as ranking_metrics
from azimuth.load_data import combine_organisms

# def qqplot(pvals,
#            fileout=None,
#            alphalevel=0.05,
#            legend=None,
#            xlim=None,
#            ylim=None,
#            fixaxes=True,
#            addlambda=True,
#            minpval=1e-20,
#            title=None,
#            h1=None,
#            figsize=[5, 5],
#            grid=True,
#            markersize=2):
#     """
#     performs a P-value QQ-plot in -log10(P-value) space
#     -----------------------------------------------------------------------
#     Args:
#         pvals       P-values, for multiple methods this should be a list (each element will be flattened)
#         fileout    if specified, the plot will be saved to the file (optional)
#         alphalevel  significance level for the error bars (default 0.05)
#                     if None: no error bars are plotted
#         legend      legend string. For multiple methods this should be a list
#         xlim        X-axis limits for the QQ-plot (unit: -log10)
#         ylim        Y-axis limits for the QQ-plot (unit: -log10)
#         fixaxes    Makes xlim=0, and ylim=max of the two ylimits, so that plot is square
#         addlambda   Compute and add genomic control to the plot, bool
#         title       plot title, string (default: empty)
#         h1          figure handle (default None)
#         figsize     size of the figure. (default: [5,5])
#         grid        boolean: use a grid? (default: True)
#     Returns:   fighandle, qnull, qemp
#     -----------------------------------------------------------------------
#     """
#     distr = 'log10'
#     import pylab as pl
#     if type(pvals) == list:
#         pvallist = pvals
#     else:
#         pvallist = [pvals]
#     if type(legend) == list:
#         legendlist = legend
#     else:
#         legendlist = [legend]
#
#     if h1 is None:
#         h1 = pl.figure(figsize=figsize)
#
#     pl.grid(b=grid, alpha=0.5)
#
#     maxval = 0
#
#     for i in range(len(pvallist)):
#         pval = pvallist[i].flatten()
#         M = pval.shape[0]
#         pnull = (0.5 + sp.arange(M)) / M
#
#         pval[pval < minpval] = minpval
#         pval[pval >= 1] = 1
#
#         if distr == 'chi2':
#             qnull = st.chi2.isf(pnull, 1)
#             qemp = (st.chi2.isf(sp.sort(pval), 1))
#             xl = 'LOD scores'
#             yl = '$\chi^2$ quantiles'
#
#         if distr == 'log10':
#             qnull = -sp.log10(pnull)
#             qemp = -sp.log10(sp.sort(pval))  # sorts the object, returns nothing
#             xl = '-log10(P) observed'
#             yl = '-log10(P) expected'
#         if not (sp.isreal(qemp)).all(): raise Exception("imaginary qemp found")
#         if qnull.max > maxval:
#             maxval = qnull.max()
#         pl.plot(qnull, qemp, '.', markersize=markersize)
#         if addlambda:
#             lambda_gc = estimate_lambda(pval)
#             print(f"lambda={lambda_gc:.4f}")
#             # pl.legend(["gc="+ '%1.3f' % lambda_gc],loc=2)
#             # if there's only one method, just print the lambda
#             if len(pvallist) == 1:
#                 legendlist = [f"$\lambda_{{GC}}=${lambda_gc:.4f}"]
#                 # otherwise add it at the end of the name
#             else:
#                 legendlist[i] = f"{legendlist[i]} ($\lambda_{{GC\\}}=${lambda_gc:.4f})"
#
#     addqqplotinfo(qnull, M, xl, yl, xlim, ylim, alphalevel, legendlist, fixaxes)
#
#     if title is not None:
#         pl.title(title)
#
#     if fileout is not None:
#         pl.savefig(fileout)
#
#     return h1, qnull, qemp
#
#
# def qqplotp(pv,
#             fileout=None,
#             alphalevel=0.05,
#             legend=None,
#             xlim=None,
#             ylim=None,
#             plotsize="652x526",
#             title=None,
#             dohist=True,
#             numbins=50,
#             figsize=[5, 5],
#             markersize=2):
#     """
#     Read in p-values from filein and make a qqplot adn histogram.
#     If fileout is provided, saves the qqplot only at present.
#     Searches through p until one is found.
#     """
#
#     import pylab as pl
#     pl.ion()
#
#     fs = 8
#     h1 = qqplot(pv, fileout, alphalevel, legend, xlim, ylim, addlambda=True, figsize=figsize, markersize=markersize)
#     pl.title(title, fontsize=fs)
#
#     pl.get_current_fig_manager()
#     # e.g. "652x526+100+10
#     xcoord = 100
#
#     if dohist:
#         h2 = pvalhist(pv, numbins=numbins, figsize=figsize)
#         pl.title(title, fontsize=fs)
#         width_height = plotsize.split("x")
#         buffer = 10
#         xcoord = int(xcoord + float(width_height[0]) + buffer)
#     else:
#         h2 = None
#
#     return h1, h2


# def addqqplotinfo(qnull, M, xl='-log10(P) observed', yl='-log10(P) expected', xlim=None, ylim=None, alphalevel=0.05,
#                   legendlist=None, fixaxes=False):
#     distr = 'log10'
#     pl.plot([0, qnull.max()], [0, qnull.max()], 'k')
#     pl.ylabel(xl)
#     pl.xlabel(yl)
#     if xlim is not None:
#         pl.xlim(xlim)
#     if ylim is not None:
#         pl.ylim(ylim)
#     if alphalevel is not None:
#         if distr == 'log10':
#             betaUp, betaDown, theoreticalPvals = _qqplot_bar(M=M, alphalevel=alphalevel, distr=distr)
#             lower = -sp.log10(theoreticalPvals - betaDown)
#             upper = -sp.log10(theoreticalPvals + betaUp)
#             pl.fill_between(-sp.log10(theoreticalPvals), lower, upper, color="grey", alpha=0.5)
#     if legendlist is not None:
#         leg = pl.legend(legendlist, loc=4, numpoints=1)
#         # set the markersize for the legend
#         for lo in leg.legendHandles:
#             lo.set_markersize(10)
#
#     if fixaxes:
#         fix_axes()


# def _qqplot_bar(M=1000000, alphalevel=0.05):
#     """
#     calculate error bars for a QQ-plot
#     --------------------------------------------------------------------
#     Input:
#     -------------   ----------------------------------------------------
#     M               number of points to compute error bars
#     alphalevel      significance level for the error bars (default 0.05)
#     distr           space in which the error bars are implemented
#                     Note only log10 is implemented (default 'log10')
#     --------------------------------------------------------------------
#     Returns:
#     -------------   ----------------------------------------------------
#     betaUp          upper error bars
#     betaDown        lower error bars
#     theoreticalPvals    theoretical P-values under uniform
#     --------------------------------------------------------------------
#     """
#
#     # assumes 'log10'
#
#     mRange = 10 ** (sp.arange(sp.log10(0.5), sp.log10(M - 0.5) + 0.1, 0.1))  # should be exp or 10**?
#     numPts = len(mRange)
#     betaalphaLevel = sp.zeros(numPts)  # down in the plot
#     betaOneMinusalphaLevel = sp.zeros(numPts)  # up in the plot
#     betaInvHalf = sp.zeros(numPts)
#     for n in range(numPts):
#         m = mRange[n]  # numplessThanThresh=m;
#         betaInvHalf[n] = st.beta.ppf(0.5, m, M - m)
#         betaalphaLevel[n] = st.beta.ppf(alphalevel, m, M - m)
#         betaOneMinusalphaLevel[n] = st.beta.ppf(1 - alphalevel, m, M - m)
#         pass
#     betaDown = betaInvHalf - betaalphaLevel
#     betaUp = betaOneMinusalphaLevel - betaInvHalf
#
#     theoreticalPvals = mRange / M
#     return betaUp, betaDown, theoreticalPvals
#
#
# def fix_axes(buffer=0.1):
#     """
#     Makes x and y max the same, and the lower limits 0.
#     """
#     maxlim = max(pl.xlim()[1], pl.ylim()[1])
#     pl.xlim([0 - buffer, maxlim + buffer])
#     pl.ylim([0 - buffer, maxlim + buffer])


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
    L = (LOD2 / 0.456)
    return L


# def pvalhist(pv, numbins=50, linewidth=3.0, linespec='--r', figsize=[5, 5]):
#     """
#     Plots normalized histogram, plus theoretical null-only line.
#     """
#     h2 = pl.figure(figsize=figsize)
#     [nn, bins, patches] = pl.hist(pv, numbins, normed=True)
#     pl.plot([0, 1], [1, 1], linespec, linewidth=linewidth)

def get_pval_from_predictions(m0_predictions, m1_predictions, ground_truth, twotailed=False, method='steiger'):
    """
    If twotailed==False, then need to check that the one of corr0 and corr1 that is higher is the correct one
    """
    from . import corrstats
    n0 = len(m0_predictions)
    n1 = len(m1_predictions)
    n2 = len(ground_truth)
    assert (n0 == n1)
    assert (n0 == n2)
    corr0, _ = scipy.stats.spearmanr(m0_predictions, ground_truth)
    corr1, _ = scipy.stats.spearmanr(m1_predictions, ground_truth)
    corr01, _ = scipy.stats.spearmanr(m0_predictions, m1_predictions)
    t2, pv = corrstats.dependent_corr(corr0, corr1, corr01, n0, twotailed=twotailed, method=method)
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
            convert_to_thirty_one(data.iloc[i]["30mer"], data.iloc[i]["Target"], data.iloc[i]["Strand"]))
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
    if strand == 'sense':
        guide_seq = guide_seq.reverse_complement()
    ind = gene_seq.find(guide_seq)
    if ind == -1:
        print(f"returning None, could not find guide {guide_seq} in gene {gene}")
        return ""
    assert gene_seq[ind:(ind + len(guide_seq))] == guide_seq, "match not right"
    ## now get what we want from this:
    import pdb
    pdb.set_trace()
    raise NotImplementedError("incomplete implemented for now")


def convert_to_thirty_one(guide_seq, gene, strand):
    '''
    Given a guide sequence, a gene name, and strand (e.g. "sense"), return a 31mer string which is our 30mer,
    plus one more at the end.
    '''
    guide_seq = Seq.Seq(guide_seq)
    gene_seq = Seq.Seq(get_gene_sequence(gene)).reverse_complement()
    if strand == 'sense':
        guide_seq = guide_seq.reverse_complement()
    ind = gene_seq.find(guide_seq)
    if ind == -1:
        print(f"returning sequence+'A', could not find guide {guide_seq} in gene {gene}")
        return gene_seq + 'A'
    assert gene_seq[ind:(ind + len(guide_seq))] == guide_seq, "match not right"
    new_mer = gene_seq[(ind - 1):(ind + len(guide_seq))]
    # this actually tacks on an extra one at the end for some reason
    if strand == 'sense':
        new_mer = new_mer.reverse_complement()
    return str(new_mer)


def concatenate_feature_sets(feature_sets, keys=None):
    '''
    Given a dictionary of sets of features, each in a pd.DataFrame,
    concatenate them together to form one big np.array, and get the dimension
    of each set
    Returns: inputs, dim
    '''
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
    '''
    Extract predictions and truth for each fold
    Returns: ranks, predictions

    assumes that results here is the value for a results dictionary for one key, i.e. one entry in a dictionary loaded
    up from saved results with pickle e.g. all_results, all_learn_options = pickle.load(some_results_file)
    then call extract_individual_level_data(one_results = all_results['firstkey'])
    then, one_results contains: metrics, gene_pred, fold_labels, m, dimsum, filename, feature_names
    '''
    metrics, gene_pred, fold_labels, m, dimsum, filename, feature_names = one_result
    all_true_ranks = np.empty(0)
    all_pred = np.empty(0)
    for f in list(fold_labels):
        these_ranks = gene_pred[0][0][f]['ranks']  # similar for thrs
        these_pred = gene_pred[0][1][f]
        all_true_ranks = np.concatenate((all_true_ranks, these_ranks))
        all_pred = np.concatenate((all_pred, these_pred))
    return all_true_ranks, all_pred


def spearmanr_nonan(x, y):
    '''
    same as scipy.stats.spearmanr, but if all values are equal, returns 0 instead of nan
    (Output: rho, pval)
    '''
    r, p = st.spearmanr(x, y)
    r = np.nan_to_num(r)
    p = np.nan_to_num(p)
    return r, p


def impute_gene_position(gene_position):
    '''
    Some amino acid cut position and percent peptide are blank because of stop codons, but
    we still want a number for these, so just set them to 101 as a proxy
    '''

    gene_position['Percent Peptide'] = gene_position['Percent Peptide'].fillna(101.00)

    if 'Amino Acid Cut position' in gene_position.columns:
        gene_position['Amino Acid Cut position'] = gene_position['Amino Acid Cut position'].fillna(
            gene_position['Amino Acid Cut position'].mean())

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
        gene_file = '../../gene_sequences/%s_sequence.txt' % gene_name
        # gene_file = '../gene_sequences/%s_sequence.txt' % gene_name
        # gene_file = 'gene_sequences/%s_sequence.txt' % gene_name
        with open(gene_file, 'rb') as f:
            seq = f.read()
            seq = seq.replace('\r\n', '')
    except:
        raise Exception(f"could not find gene sequence file {gene_file}, please see examples and generate one for your gene "
                        f"as needed, with this filename")

    return seq


def target_genes_stats(genes=('HPRT1', 'TADA1', 'NF2', 'TADA2B', 'NF1', 'CUL3', 'MED12', 'CCDC101')):
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
        rank /= (X.shape[0] + 1)
        RV[:, i] = sp.sqrt(2) * sp.special.erfinv(2 * rank - 1)

    return RV.flatten()


def get_ranks(y, thresh=0.8, prefix="", flip=False, col_name='score'):
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
        y_rank = 1.0 - y_rank  # before this line, 1-labels where associated with low ranks, this flips it around
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
    '''
    this is called once for each gene (aggregating across cell types)
    y_names are cell types
    e.g. call: X_CD13, Y_CD13 = get_data(cd13, y_names=['NB4 CD13', 'TF1 CD13'])
    '''

    # generate ranks for each cell type before aggregating to match what is in Doench et al
    thresh = 0.8
    for y_name in y_names:  # for each cell type
        y = pd.DataFrame(data[y_name])
        # these thresholds/quantils are not used:
        y_rank, y_rank_raw, y_threshold, y_quantiles = get_ranks(y, thresh=thresh, flip=False, col_name=y_name)
        y_rank.columns = [y_name + " rank"]
        y_rank_raw.columns = [y_name + " rank raw"]
        y_threshold.columns = [y_name + " threshold"]

        outputs = pd.concat([outputs, y, y_rank, y_threshold, y_rank_raw], axis=1)

    # aggregated rank across cell types
    average_activity = pd.DataFrame(outputs[[y_name for y_name in y_names]].mean(1))
    average_activity.columns = ['average activity']

    average_rank_from_avg_activity = \
        get_ranks(average_activity, thresh=thresh, flip=False, col_name='average activity')[0]
    average_rank_from_avg_activity.columns = ['average_rank_from_avg_activity']
    average_threshold_from_avg_activity = (average_rank_from_avg_activity > thresh) * 1
    average_threshold_from_avg_activity.columns = ['average_threshold_from_avg_activity']

    average_rank = pd.DataFrame(outputs[[y_name + ' rank' for y_name in y_names]].mean(1))
    average_rank.columns = ['average rank']
    # higher ranks are better (when flip=False as it should be)
    average_threshold = (average_rank > thresh) * 1
    average_threshold.columns = ['average threshold']

    # undo the log2 trafo on the reads per million, apply rank trafo right away
    average_rank_raw = pd.DataFrame(outputs[[y_name + ' rank raw' for y_name in y_names]].mean(1))
    average_rank_raw.columns = ['average rank raw']
    outputs = pd.concat(
        [outputs, average_rank, average_threshold, average_activity, average_rank_raw, average_rank_from_avg_activity,
         average_threshold_from_avg_activity], axis=1)

    # import pdb; pdb.set_trace()

    # sequence-specific computations
    # features = featurize_data(data)
    # strip out featurization to later
    features = pd.DataFrame(data['30mer'])

    if organism is "human":
        target_gene = y_names[0].split(' ')[1]

    outputs['Target gene'] = target_gene
    outputs['Organism'] = organism

    features['Target gene'] = target_gene
    features['Organism'] = organism
    features['Strand'] = pd.DataFrame(data['Strand'])

    return features, outputs


def autolabel(ax, rects, strfrm='%.2f'):
    '''
    Automatically add value over each bar in bar chart
    http://matplotlib.org/1.4.2/examples/api/barchart_demo.html
    '''
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, strfrm % float(height),
                ha='center', va='bottom')


def create_cachedir(dirname='./cache/default'):
    if os.path.exists(dirname):
        return dirname
    else:
        os.makedirs(dirname)
        return dirname


def dcg(relevances, rank=20):
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.
    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)


def ndcgk(relevances, rank=20):
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.
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
            if 'False' in s:
                continue
            elif "_" in s:
                nucl, pos = s.split('_')
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

        grouped_feat = {'pd_order2': pd_order2,
                        'pi_order2': pi_order2,
                        'pd_order1': pd_order1,
                        'pi_order1': pi_order1,
                        'NGGX_pd.Order2': nggx, }

        grouped_feat_ind = []
        [grouped_feat_ind.extend(grouped_feat[a]) for a in list(grouped_feat.keys())]
        remaining_features_ind = set.difference(set(range(len(feature_names))), set(grouped_feat_ind))

        for i in remaining_features_ind:
            grouped_feat[feature_names[i]] = [i]

        feature_importances_grouped = {}
        for k in grouped_feat:
            if len(grouped_feat[k]) == 0:
                continue
            else:
                for split in list(results[method][3].keys()):
                    split_feat_importance = extract_feature_from_model_sum(method, results, split, grouped_feat[k])
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
                all_split_importances = np.append(all_split_importances, split_feat_importance, axis=1)

        avg_importance = np.mean(all_split_importances, axis=1)[:, None]
        std_importance = np.std(all_split_importances, axis=1)[:, None]
        imp_array = np.concatenate((np.array(feature_names)[:, None], avg_importance, std_importance), axis=1)

        df = pd.DataFrame(data=imp_array, columns=['Feature name', 'Mean feature importance', 'Std. Dev.'])
        df = df.convert_objects(convert_numeric=True)

        # boxplot_labels = np.array([k for k in list(feature_importances_grouped.keys())])
        # boxplot_arrays = np.concatenate([np.array(feature_importances_grouped[k])[:, None] for k in boxplot_labels],
        #                                 axis=1)

        feature_dictionary = {
            'pd_order2': 'position dep. order 2 ',
            'pd_order1': 'position dep. order 1 ',
            'pi_order1': 'position ind. order 1 ',
            'pi_order2': 'position ind. order 2 ',
            '5mer_end_False': 'Tm (5mer end)',
            '5mer_start_False': 'Tm (5mer start)',
            'Amino Acid Cut position': 'amino acid cut position ',
            '8mer_middle_False': 'Tm (8mer middle)',
            'NGGX_pd.Order2': 'NGGN interaction ',
            'Tm global_False': 'Tm (30mer)',
            'Percent Peptide': 'percent peptide ',
        }

        for i in range(df.shape[0]):
            thisfeat = df['Feature name'].iloc[i]
            if thisfeat in feature_dictionary.keys():
                df['Feature name'].iloc[i] = feature_dictionary[thisfeat]

        # descriptive_labels = np.array(
        #     [feature_dictionary[k] if k in list(feature_dictionary.keys()) else k + " " for k in boxplot_labels])

        # sorted_boxplot = np.argsort(np.median(boxplot_arrays, axis=0))[::-1]
        # boxplot_means = np.mean(boxplot_arrays, axis=0)[sorted_boxplot]
        # boxplot_std = np.std(boxplot_arrays, axis=0)[sorted_boxplot]

        # ind = np.arange(0, len(boxplot_labels) * 2, 2)  # farange(len(boxplot_labels))
        # width = 1.5
        # plt.figure(figsize=figsize)
        # plt.bar(ind, boxplot_means, width, color='#186499', yerr=boxplot_std, ecolor='k', edgecolor='none')

        # ax = plt.gca()
        # ax.set_ylabel('Average Gini importances', fontsize=fontsize)
        # ax.set_xticks(ind + width / 2.0 + 0.1)
        #
        # ax.set_xticklabels(descriptive_labels[sorted_boxplot], rotation=90, fontsize=fontsize)
        # plt.ylim([0.0, 0.5])
        # plt.subplots_adjust(top=0.97, bottom=0.4)

        # plt.boxplot(boxplot_arrays[:, sorted_boxplot])
        # plt.ylabel('Average Gini')
        # plt.xticks(range(1, len(boxplot_labels)+1), np.array(boxplot_labels)[sorted_boxplot], rotation=70)
        # plt.subplots_adjust(top = 0.97, bottom = 0.4)
        return df


def check_learn_options_set(learn_options_set):
    if learn_options_set is None:
        return 'ranks'

    non_binary_target_name = None
    for l in list(learn_options_set.values()):
        if non_binary_target_name is None:
            non_binary_target_name = l["testing_non_binary_target_name"]
        else:
            assert non_binary_target_name == l["testing_non_binary_target_name"], "need to have same " \
                                                                                  "testing_non_binary_target_name " \
                                                                                  "across all learn options in a set " \
                                                                                  "for metrics to be comparable"
    return non_binary_target_name


def get_all_metrics(results,
                    learn_options_set=None,
                    test_metrics=['spearmanr'],
                    add_extras=False, force_by_gene=False):
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
        if len(tmp_genes) != len(tmp_genes) or np.any(tmp_genes == genes): "genes have changed, need to modify code"
        all_truth_raw, all_truth_thrs, all_predictions = np.array([]), np.array([]), np.array([])

        fpr_gene = {}
        tpr_gene = {}
        y_truth_thresh_all = np.array([])
        y_pred_all = np.array([])

        for gene in genes:
            y_truth, y_pred = truth[gene], predictions[gene]
            all_truth_raw = np.append(all_truth_raw, y_truth[non_binary_target_name])
            all_truth_thrs = np.append(all_truth_thrs, y_truth['thrs'])
            all_predictions = np.append(all_predictions, y_pred)

            y_truth_thresh_all = np.append(y_truth_thresh_all, y_truth['thrs'])
            y_pred_all = np.append(y_pred_all, y_pred)

            if 'spearmanr' in test_metrics:
                spearmanr = np.nan_to_num(st.spearmanr(y_truth[non_binary_target_name], y_pred)[0])
                all_results[method]['spearmanr'].append(spearmanr)

            if 'spearmanr>2.5' in test_metrics:
                selected = y_truth[non_binary_target_name] > 1.0
                # spearmanr = sp.stats.spearmanr(y_truth[non_binary_target_name][selected], y_pred[selected])[0]
                spearmanr = np.sqrt(np.mean((y_truth[non_binary_target_name][selected] - y_pred[selected]) ** 2))
                all_results[method]['spearmanr>2.5'].append(spearmanr)

            if 'RMSE' in test_metrics:
                rmse = np.sqrt(np.mean((y_truth[non_binary_target_name] - y_pred) ** 2))
                all_results[method]['RMSE'].append(rmse)

            if 'NDCG@5' in test_metrics:
                ndcg = ranking_metrics.ndcg_at_k_ties(y_truth[non_binary_target_name], y_pred, 5)
                all_results[method]['NDCG@5'].append(ndcg)

            if 'NDCG@10' in test_metrics:
                ndcg = ranking_metrics.ndcg_at_k_ties(y_truth[non_binary_target_name], y_pred, 10)
                all_results[method]['NDCG@10'].append(ndcg)

            if 'NDCG@20' in test_metrics:
                ndcg = ranking_metrics.ndcg_at_k_ties(y_truth[non_binary_target_name], y_pred, 20)
                all_results[method]['NDCG@20'].append(ndcg)

            if 'NDCG@50' in test_metrics:
                ndcg = ranking_metrics.ndcg_at_k_ties(y_truth[non_binary_target_name], y_pred, 50)
                all_results[method]['NDCG@50'].append(ndcg)

            if 'precision@5' in test_metrics:
                y_top_truth = (y_truth[non_binary_target_name] >= np.sort(y_truth[non_binary_target_name])[::-1][:5][
                    -1]) * 1
                y_top_pred = (y_pred >= np.sort(y_pred)[::-1][:5][-1]) * 1
                all_results[method]['precision@5'].append(sklearn.metrics.precision_score(y_top_pred, y_top_truth))

            if 'precision@10' in test_metrics:
                y_top_truth = (y_truth[non_binary_target_name] >= np.sort(y_truth[non_binary_target_name])[::-1][:10][
                    -1]) * 1
                y_top_pred = (y_pred >= np.sort(y_pred)[::-1][:10][-1]) * 1
                all_results[method]['precision@10'].append(sklearn.metrics.precision_score(y_top_pred, y_top_truth))

            if 'precision@20' in test_metrics:
                y_top_truth = (y_truth[non_binary_target_name] >= np.sort(y_truth[non_binary_target_name])[::-1][:20][
                    -1]) * 1
                y_top_pred = (y_pred >= np.sort(y_pred)[::-1][:20][-1]) * 1
                all_results[method]['precision@20'].append(sklearn.metrics.precision_score(y_top_pred, y_top_truth))

            if 'AUC' in test_metrics:
                fpr_gene[gene], tpr_gene[gene], _ = sklearn.metrics.roc_curve(y_truth['thrs'], y_pred)
                auc = sklearn.metrics.auc(fpr_gene[gene], tpr_gene[gene])
                all_results[method]['AUC'].append(auc)

    if add_extras:
        fpr_all, tpr_all, _ = sklearn.metrics.roc_curve(y_truth_thresh_all, y_pred_all)
        return all_results, genes, fpr_all, tpr_all, fpr_gene, tpr_gene
    else:
        return all_results, genes


# def plot_all_metrics(metrics, gene_names, all_learn_options, save, plots=None, bottom=0.19):
#     num_methods = len(list(metrics.keys()))
#     metrics_names = list(metrics[list(metrics.keys())[0]].keys())
#     num_genes = len(gene_names)
#     width = 0.9 / num_methods
#     ind = np.arange(num_genes)
#
#     if save == True:
#         first_key = list(all_learn_options.keys())[0]
#         # basefile = r"..\results\V%s_trmetric%s_%s" % (all_learn_options[first_key]["V"],
#         # all_learn_options[first_key]["training_metric"], datestamp())
#         basefile = r"../results/%s" % (first_key)
#
#         d = os.path.dirname(basefile)
#         if not os.path.exists(d):
#             os.makedirs(d)
#         with open(basefile + ".plot.pickle", "wb") as f:
#             pickle.dump([metrics, all_learn_options, gene_names], f)
#
#     for metric in metrics_names:
#         if 'global' not in metric:
#             plt.figure(metric, figsize=(20, 8))
#         elif plots == None or 'gene level' in plots:
#             plt.figure(metric, figsize=(12, 12))
#
#     boxplot_labels = []
#     boxplot_arrays = {}
#     boxplot_median = {}
#
#     for i, method in enumerate(metrics.keys()):
#         boxplot_labels.append(method)
#         for metric in list(metrics[method].keys()):
#
#             if 'global' in metric:
#                 plt.figure(metric)
#                 plt.bar([i], metrics[method][metric], 0.9, color=plt.cm.Paired(1. * i / len(list(metrics.keys()))),
#                         label=method)
#             else:
#                 if plots == None or 'gene level' in plots:
#                     plt.figure(metric)
#                     plt.bar(ind + (i * width), metrics[method][metric], width,
#                             color=plt.cm.Paired(1. * i / len(list(metrics.keys()))), label=method)
#
#                 median_metric = np.median(metrics[method][metric])
#                 print(f"{method}, {metric}, {median_metric}")
#                 assert not np.isnan(median_metric), "found nan for %s, %s" % (method, metric)
#                 if metric not in list(boxplot_arrays.keys()):
#                     boxplot_arrays[metric] = np.array(metrics[method][metric])[:, None]
#                     boxplot_median[metric] = [np.median(np.array(metrics[method][metric]))]
#                 else:
#                     boxplot_arrays[metric] = np.concatenate(
#                         (boxplot_arrays[metric], np.array(metrics[method][metric])[:, None]), axis=1)
#                     boxplot_median[metric].append(np.median(np.array(metrics[method][metric])))
#
#     for metric in metrics_names:
#         if plots == None or 'gene level' in plots:
#             ax = plt.figure(metric)
#             leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#             # leg.draggable(state=True, use_blit=True)
#             plt.ylabel(metric)
#
#             if 'global' in metric:
#                 plt.xticks(list(range(len(list(metrics.keys())))), list(metrics.keys()), rotation=70)
#                 plt.grid(True, which='both')
#                 plt.subplots_adjust(left=0.05, right=0.8)
#             else:
#                 plt.xticks(ind + width, gene_names)
#                 plt.grid(True, which='both')
#                 plt.subplots_adjust(left=0.05, right=0.8)
#         if save == True:
#             plt.xticks(ind + 0.5, gene_names)
#             if metric == 'AUC':
#                 plt.ylim([0.5, 1.0])
#             plt.savefig(basefile + "_" + metric + "_bar" + ".png")
#
#         if (plots == None or "boxplots" in plots) and 'global' not in metric:
#             plt.figure('Boxplot %s' % metric)
#
#             sorted_boxplot = np.argsort(boxplot_median[metric])[::-1]
#
#             plt.boxplot(boxplot_arrays[metric][:, sorted_boxplot])
#             plt.ylabel(metric)
#             plt.xticks(list(range(1, num_methods + 1)), np.array(boxplot_labels)[sorted_boxplot], rotation=70)
#             plt.subplots_adjust(top=0.97, bottom=bottom)
#
#             if metric == 'RMSE':
#                 plt.ylim((1.0, 2.0))
#
#         if save == True:
#             plt.savefig(basefile + "_" + metric + ".png")


def load_results(directory, all_results, all_learn_options, model_filter=None, append_to_key=None):
    '''
    Only load up files which contain one of the strings in model_filter in their names
    model_filter should be a list, or a string
    '''
    num_added = 0
    filelist = glob.glob(directory + '\\*.pickle')
    if filelist == []:
        raise Exception("found no pickle files in %s" % directory)
    else:
        print()
        "found %d files in %s" % (len(filelist), directory)

    for results_file in filelist:
        if 'learn_options' in results_file:
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
            with open(results_file, 'rb') as f:
                results, learn_options = pickle.load(f)
            gene_names = None
        except:
            with open(results_file, 'rb') as f:
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


def plot_cluster_results(metrics=['spearmanr', 'NDCG@5'], plots=['boxplots'],
                         directory=r'\\fusi1\crispr2\analysis\cluster\results', results=None, learn_options=None,
                         filter=None):
    all_results = {}
    all_learn_options = {}

    if results is None:
        if type(directory) == list:
            for exp_dir in directory:
                all_results, all_learn_options = load_results(exp_dir, all_results, all_learn_options, filter)
        else:
            all_results, all_learn_options = load_results(directory, all_results, all_learn_options, filter)

    else:
        for k in list(results.keys()):
            assert k not in list(all_results.keys())
            all_results[k] = results[k]
            all_learn_options[k] = learn_options[k]

    all_metrics, gene_names = get_all_metrics(all_results, test_metrics=metrics)
    # plot_all_metrics(all_metrics, gene_names, all_learn_options, plots=plots, save=False)


def ensemble_cluster_results(directory=r'\\fusi1\crispr2\analysis\cluster\results\cluster_experiment_izf_ob',
                             ensemble_type='median', models_to_ensemble=['all']):
    all_results = {}
    all_learn_options = {}

    for results_file in glob.glob(directory + '\\*.pickle'):
        if 'learn_options' in results_file:
            continue

        with open(results_file, 'rb') as f:
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
                test_predictions = np.append(test_predictions, predictions[gene][:, None], axis=1)

            # this is just to check that all the models are using the same ordering of
            # the ground truth and hence of the samples, as this might mess up the ensemble.
            if prev_model_truth is not None:
                assert np.all(truth[gene]['ranks'] == prev_model_truth)
            else:
                prev_model_truth = truth[gene]['ranks']

            # take all the other genes and stack the predictions under a given model.
            cv_predictions_gene_j = np.array([])
            cv_truth_gene_j = np.array([])
            for other_gene in genes:
                if gene == other_gene:
                    continue
                cv_predictions_gene_j = np.append(cv_predictions_gene_j, predictions[other_gene])
                cv_truth_gene_j = np.append(cv_truth_gene_j, truth[other_gene]['ranks'])

            if cv_truth is None:
                cv_truth = cv_truth_gene_j.copy()[:, None]

            if cv_predictions is None:
                cv_predictions = cv_predictions_gene_j[:, None]
            else:
                cv_predictions = np.append(cv_predictions, cv_predictions_gene_j[:, None],
                                           axis=1)

        if ensemble_type is 'majority':
            y_pred = ensembles.pairwise_majority_voting(test_predictions)
        if ensemble_type is 'median':
            y_pred = ensembles.median(test_predictions)
        if ensemble_type is 'stacking':
            y_pred = ensembles.linear_stacking(cv_truth, cv_predictions, test_predictions)

        ens_predictions[gene] = y_pred
        ens_truths[gene] = truth[gene]

    all_results[ensemble_type] = [None, [[ens_truths, ens_predictions]], None, None]
    all_learn_options[ensemble_type] = None

    return all_results, all_learn_options


# def plot_old_vs_new_feat(results, models, fontsize=20, filename=None):
#     model_names = []
#     for model in models:
#         if 'doench' in model:
#             model_names.append('SVM + LogReg')
#         elif 'AB_' in model:
#             model_names.append('AdaBoost DT')
#         else:
#             model_names.append(model)
#
#     base_spearman_means = []
#     base_AUC_means = []
#     feat_spearman_means = []
#     feat_AUC_means = []
#     base_spearman_std = []
#     feat_spearman_std = []
#     base_AUC_se = []
#     feat_AUC_se = []
#
#     for model in models:
#         metrics = get_all_metrics({model: results[model]}, test_metrics=['spearmanr', 'AUC'])[0][model]
#         metrics_feat = \
#             get_all_metrics({model + '_feat': results[model + "_feat"]}, test_metrics=['spearmanr', 'AUC'])[0][
#                 model + '_feat']
#
#         base_spearman_means.append(np.mean(metrics['spearmanr']))
#         base_spearman_std.append(np.std(metrics['spearmanr']))
#         base_AUC_means.append(np.mean(metrics['AUC']))
#         base_AUC_se.append(np.std(metrics['AUC']))
#
#         feat_spearman_means.append(np.mean(metrics_feat['spearmanr']))
#         feat_spearman_std.append(np.std(metrics_feat['spearmanr']))
#         feat_AUC_means.append(np.mean(metrics_feat['AUC']))
#         feat_AUC_se.append(np.std(metrics_feat['AUC']))
#
#     print(f"old features\n"
#           f"mean: {str(base_spearman_means)}\n"
#           f"std: {str(base_spearman_std)}\n"
#           f"old + new features\n"
#           f"mean: {str(feat_spearman_means)}\n"
#           f"std: {str(feat_spearman_std)}")
#
#
#     plt.figure()
#     ind = np.arange(len(models))
#     width = 0.4
#     plt.bar(ind, base_spearman_means, width, color='#D14B5D', yerr=base_spearman_std, ecolor='k', edgecolor='none',
#             label='Old features')
#     plt.bar(ind + width, feat_spearman_means, width, color='#852230', yerr=feat_spearman_std, ecolor='k',
#             edgecolor='none', label='Old + new features')
#     ax = plt.gca()
#     ax.set_ylabel('Spearman r', fontsize=fontsize)
#     ax.set_xticks(ind + width)
#     ax.set_xticklabels(model_names, fontsize=fontsize)
#     plt.legend(loc=0, fontsize=fontsize)
#     plt.yticks(fontsize=fontsize)
#     plt.ylim((0.0, 0.7))
#     remove_top_right_on_plot()
#     if filename is not None:
#         plt.savefig(filename + '_spearman.pdf')
#
#     plt.figure()
#     ind = np.arange(len(models))
#     width = 0.4
#     plt.bar(ind, base_AUC_means, width, color='#D14B5D', yerr=base_AUC_se, ecolor='k', edgecolor='none',
#             label='Old features')
#     plt.bar(ind + width, feat_AUC_means, width, color='#852230', yerr=feat_AUC_se, ecolor='k', edgecolor='none',
#             label='Old + new features')
#     ax = plt.gca()
#     ax.set_ylabel('AUC', fontsize=fontsize)
#     ax.set_xticks(ind + width)
#     ax.set_xticklabels(model_names, fontsize=fontsize)
#     plt.legend(loc=0)
#     plt.ylim((0.5, 0.85))
#     plt.legend(loc=0, fontsize=fontsize)
#     plt.yticks(fontsize=fontsize)
#     remove_top_right_on_plot()
#     if filename is not None:
#         plt.savefig(filename + '_AUC.pdf')
#
#     # plt.subplots_adjust(top = 0.97, bottom = 0.4)


# def remove_top_right_on_plot(ax=None):
#     if ax == None:
#         ax = plt.gca()
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)


if __name__ == '__main__':
    get_thirty_one_mer_data();
    import pdb;

    pdb.set_trace()

    V = "1"
    if V == "1":
        human_data = pd.read_excel("data/V1_data.xlsx", sheetname=0, index_col=[0, 1])
        mouse_data = pd.read_excel("data/V1_data.xlsx", sheetname=1, index_col=[0, 1])
        X, Y = combine_organisms()
        X.to_pickle('../data/X.pd')  # sequence features (i.e. inputs to prediction)
        Y.to_pickle('../data/Y.pd')  # cell-averaged ranks, plus more (i.e. possible targets for prediction)
        print()
        "done writing to file"
    elif V == "2":
        # this is now all in predict.py
        pass
    elif V == "0":
        pass
