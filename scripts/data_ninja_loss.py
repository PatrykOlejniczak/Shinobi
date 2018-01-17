import numpy as np


def data_ninja_loss(y_true, y_pred):
    """
    Calculates loss function of OLX Data Ninja 2018 competition:
    L_{rank} = \frac{1}{\sum_{k < l} n_k n_l} \sum_{y_i < y_j} \left ( [[ f(x_i) > f(x_j) ]] + 1/2 [[ f(x_i) = f(x_j) ]] )

    Parameters (in scikit-learn-like convention):
    ----------
    y_true : 1d numpy array or list
        Ground truth (correct) values/classes.
    y_pred : 1d numpy array or list
        Predicted values/classes as returned by a regressor/classifier.

    Returns
    -------
    score : float
    """

    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    if y_pred.shape != y_true.shape:
            raise ValueError("Lengths of y_true and y_pred do not match!")

    return _fast_data_ninja_loss(y_true, y_pred)


def _data_ninja_loss(y_true, y_pred):

    yt_rank = y_true.argsort()
    yt = y_true[yt_rank]
    yp = y_pred[yt_rank]
    yt_uniq, yt_index, yt_uniq_count = np.unique(yt, return_index=True, return_counts=True)

    loss = 0
    loss_div = 0
    for i in range(1, yt_uniq.shape[0]):
        loss_div += yt_uniq_count[i] * yt_uniq_count[0:i].sum()

    for k in range(1, yt_index.shape[0]):
        for i in range(yt_uniq_count[k]):
            i = yt_index[k] + i
            for j in range(yt_index[k]):
                if yp[i] < yp[j]:
                    loss += 1
                elif yp[i] == yp[j]:
                    loss += 0.5

    return loss / loss_div


def _fast_data_ninja_loss(y_true, y_pred):

    yt_uniq, yt_uniq_count = np.unique(y_true, return_counts=True)
    yt_uniq_dict = {}
    for i in range(yt_uniq.shape[0]):
        yt_uniq_dict[yt_uniq[i]] = i

    yp_rank = y_pred.argsort()
    yt = y_true[yp_rank]
    yp = y_pred[yp_rank]

    loss = 0
    loss_div = 0
    for i in range(1, yt_uniq.shape[0]):
        loss_div += yt_uniq_count[i] * yt_uniq_count[0:i].sum()

    yt_seen_count = np.zeros(yt_uniq.shape, dtype=np.int)
    yt_seen_local_count = np.zeros(yt_uniq.shape, dtype=np.int)
    for i in range(yp.shape[0] - 1, -1, -1):
        if i < yp.shape[0] - 1 and yp[i + 1] != yp[i]:
            for j in range(yt_uniq.shape[0]):
                loss += 0.5 * yt_seen_local_count[j] * yt_seen_local_count[j + 1:].sum()

            yt_seen_count += yt_seen_local_count
            yt_seen_local_count = np.zeros(yt_uniq.shape, dtype=np.int)

        for j in range(yt_uniq_dict[yt[i]]):
            loss += yt_seen_count[j]

        yt_seen_local_count[yt_uniq_dict[yt[i]]] += 1

    for j in range(yt_uniq.shape[0]):
        loss += 0.5 * yt_seen_local_count[j] * yt_seen_local_count[j + 1:].sum()

    return loss / loss_div
