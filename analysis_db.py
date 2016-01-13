import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

def sort_by_pause_length(file_name):
    pause_before, pause_after = [], []
    conn = sqlite3.connect('./alignment_data/{}.db'.format(file_name))
    cur = conn.cursor()

    cur.execute("SELECT word, word_index, count_before, count_after, avg_pause_before, avg_pause_after FROM words")
    vals = cur.fetchall()

    for val in vals:
        entry_before = (val[0], val[4], val[2])
        entry_after  = (val[0], val[5], val[3])

        pause_before.append(entry_before)
        pause_after.append(entry_after)

    pause_before = np.asarray(pause_before)
    pause_after  = np.asarray(pause_after)

    idx_before = np.argsort(pause_before[:, 1])[::-1]
    idx_after  = np.argsort(pause_after[:, 1])[::-1]

    idx_count_before = np.argsort(pause_before[:, 2])[::-1]
    idx_count_after = np.argsort(pause_after[:, 2])[::-1]

    return pause_before[idx_before], pause_after[idx_after], pause_before[idx_count_before], pause_after[idx_count_after]



def plot_pauses_pronunc(show_name, file_name, word):
    """
    Plot the avg. pause lengths (preceding and succeeding) associated with
    the different pronunciations of a particular word.
    """
    sns.set(style="white")
    word = word.lower().strip()

    conn = sqlite3.connect('./alignment_data/{}.db'.format(file_name))
    cur = conn.cursor()
    cur.execute("SELECT word_index, prnc, count_before, count_after, avg_pause_before, avg_pause_after FROM pronunciations WHERE word_index IN (SELECT word_index FROM words WHERE word=?)", (word, ))
    vals = cur.fetchall()

    pause_before, pause_after, prnc_a, prnc_b = [], [], [], []
    for val in vals:
        pause_before.append(val[4])
        pause_after.append(val[5])
        prnc_a.append(val[1] + '\nC: {}'.format(str(val[3])))
        prnc_b.append(val[1] + '\nC: {}'.format(str(val[2])))

    title_a = "Avg. Succeeding Pause Length for " \
              "Pronunciations \nof `{}` in {}".format(word, show_name)
    title_b = "Avg. Preceding Pause Length for " \
              "Pronunciations of \n`{}` in {}".format(word, show_name)

    sort_idx = np.argsort(prnc_a)
    pause_before = np.array(pause_before)[sort_idx]
    pause_after = np.array(pause_after)[sort_idx]
    prnc_a = np.array(prnc_a)[sort_idx]
    prnc_b = np.array(prnc_b)[sort_idx]

    fig, ax = plt.subplots()
    ax = sns.barplot(prnc_a, pause_after, palette="Set3", ax=ax)
    ax.set_title(title_a)
    ax.set_ylabel("Avg. Pause Length")
    save_path = './alignment_data/{}_{}_pause_after.png'.format(word, file_name)
    plt.savefig(save_path)
    plt.close()

    fig, ax = plt.subplots()
    ax = sns.barplot(prnc_b, pause_before, palette="Set3", ax=ax)
    ax.set_title(title_b)
    ax.set_ylabel("Avg. Pause Length")
    save_path = './alignment_data/{}_{}_pause_before.png'.format(word, file_name)
    plt.savefig(save_path)
    plt.close()


def plot_prnc_counts(show_name, file_name, word):
    """
    Unfortunately, I think this isn't right. This function plots the
    counts for each pronunciation, but only if they are associated with
    pre & succeeding pauses.
    """
    sns.set(style="white")
    word = word.lower().strip()

    conn = sqlite3.connect('./alignment_data/{}.db'.format(file_name))
    cur = conn.cursor()

    cur.execute("SELECT prnc, count_before, count_after FROM pronunciations WHERE word_index IN (SELECT word_index FROM words WHERE word=?)", (word,))
    val = cur.fetchall()
    counts = [np.sum([i,q]) for i,q in [(v[1], v[2]) for v in val]]
    prncs = [v[0] for v in val]

    title = 'Counts for Pronunciations of \n`{}` in {}'.format(word, show_name)

    fig, ax = plt.subplots()
    ax = sns.barplot(prncs, counts, palette="Set3", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Count")
    save_path = './alignment_data/{}_{}_prnc_count.png'.format(word, file_name)
    plt.savefig(save_path)
    plt.close()


def sort_by_pronunc_diversity(file_name, n=10):
    """
    Returns the n words with the largest number of different pronunciations
    in the corpus.
    """
    conn = sqlite3.connect('./alignment_data/{}.db'.format(file_name))
    cur = conn.cursor()

    cur.execute("SELECT word_index FROM pronunciations")
    vals = np.array(cur.fetchall()).ravel()
    counts = np.bincount(vals)
    word_idxs = np.argsort(counts)[::-1]

    sort_idxs = np.argsort(word_idxs[:n])

    sql_list = "(" + ', '.join('?' * n) + ")"
    sql = "SELECT word FROM words WHERE word_index IN " + sql_list + " ORDER BY word_index"
    cur.execute(sql, tuple(word_idxs[:n]))
    words = np.array(cur.fetchall()).ravel()
    prnc_counts = np.sort(counts)[::-1][sort_idxs]
    sorted_counts = np.argsort(prnc_counts)[::-1]

    return np.array([(i,j) for i,j in zip(words, prnc_counts)])[sorted_counts]


def plot_pause_words(show_name, file_name):
    """
    Show words with the largest Avg. Pause Length After * Word Count
    products
    """
    sns.set(style="white")
    homophones = ['aye', 'eye', 'butt', 'hee', 'hugh', 'u', 'yu', 'sow', 'mi']
    _, _, c, d = sort_by_pause_length(file_name)

    rr_after = np.array(
        [(w, float(j) * float(i), i, j) for w, i, j in d if float(i) != 0.])
    rr_before = np.array(
        [(w, float(j) * float(i), i, j) for w, i, j in c if float(i) != 0.])

    idxs_after  = np.argsort(rr_after[:, 1].astype(float))[::-1]
    idxs_before = np.argsort(rr_before[:, 1].astype(float))[::-1]

    ratio_a = rr_after[idxs_after]
    ratio_b = rr_before[idxs_before]

    labels_a, labels_b, labels_a_idx, labels_b_idx = [], [], [], []
    ii = 0
    while len(labels_a) < 10:
        ll = '\nC: {0}\nP: {1:.5}'.format(ratio_a[ii, 3], ratio_a[ii, 2])
        labels_a.append(ratio_a[ii, 0] + ll)
        labels_a_idx.append(ii)
        if ratio_a[ii, 0] in homophones:
            labels_a.pop()
            labels_a_idx.pop()

        ll = '\nC: {0}\nP: {1:.5}'.format(ratio_b[ii, 3], ratio_b[ii, 2])
        labels_b.append(ratio_b[ii, 0] + ll)
        labels_b_idx.append(ii)
        if ratio_b[ii, 0] in homophones:
            labels_b.pop()
            labels_b_idx.pop()
        ii += 1

    yax_a = ratio_a[np.array(labels_a_idx), 1].astype(float)
    yax_b = ratio_b[np.array(labels_b_idx), 1].astype(float)

    title_a = 'Words with Top Word Count * Avg. Succeeding ' \
              'Pause Length\nProducts in {}'.format(show_name)
    title_b = 'Words with Top Word Count * Avg. Preceding ' \
              'Pause Length\nProducts in {}'.format(show_name)

    idxs_a = np.argsort(yax_a)[::-1] # for consistency across plots
    idxs_b = np.argsort(yax_b)[::-1] # for consistency across plots

    fig, ax = plt.subplots()
    ax = sns.barplot(np.array(labels_a)[idxs_a], yax_a[idxs_a],
                     palette="Set3", ax=ax)
    ax.set_title(title_a)
    ax.set_ylabel("Word Count * Avg. Pause Length")
    save_path = './alignment_data/{}_pause_after.png'.format(file_name)
    plt.savefig(save_path)
    plt.close()

    fig, ax = plt.subplots()
    ax = sns.barplot(np.array(labels_b)[idxs_b], yax_b[idxs_b],
                     palette="Set3", ax=ax)
    ax.set_title(title_b)
    ax.set_ylabel("Word Count * Avg. Pause Length")
    save_path = './alignment_data/{}_pause_before.png'.format(file_name)
    plt.savefig(save_path)
    plt.close()
