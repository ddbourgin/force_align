import os
import pickle
import numpy as np
import glob
import json
import re
from pydub import AudioSegment
import sqlite3

def compare_pronunciations(word, show_name, file_name):
    """
    Generates a plot showing the frequency of each pronunciation of a given
    word in corpus.
    """
    word = word.lower()
    as_client = init_as_client()
    as_ep_ids, as_show_id = find_episodes(show_name, as_client)

    pdict_fp = './alignment_data/pronunciations/' \
               '{}_prnc_dict.pickle'.format(file_name)

    if os.path.lexists(pdict_fp):
        with open(pdict_fp, 'rb') as handle:
            prnc_dict = pickle.load(handle)
    else:
        prnc_dict = compile_prnc_dict(as_ep_ids, file_name)

    plot_pronunciations(prnc_dict, word, show_name, file_name)
    return prnc_dict



def plot_pronunciations(prnc_dict, word, show_name, file_name):
    sns.set(style="white")
    fig, ax = plt.subplots()

    counts, labels = [], []
    prncs = prnc_dict[word].keys()
    title = 'Pronunciations for `{}` in {}'.format(word, show_name)

    for prnc in prncs:
        counts.append(prnc_dict[word][prnc]["count"])
        labels.append(prnc + '\n(Count: {})'.format(counts[-1]))

    idxs = np.argsort(labels) # for consistency across plots
    freqs = np.asarray(counts, dtype=float) / np.sum(counts)
    ax = sns.barplot(labels[idxs], freqs[idxs], palette="Set3", ax=ax)
    save_path = './alignment_data/pronunciations/' \
                '{}_pronunc_{}.png'.format(word, file_name)

    ax.set_ylabel("Frequency")
    ax.set_title(title)
    plt.savefig(save_path)
    plt.close()


def create_analysis_db(file_name):
    conn = sqlite3.connect('./alignment_data/{}.db'.format(file_name))
    cur = conn.cursor()

    cur.execute("""CREATE TABLE IF NOT EXISTS words
        (word_index INTEGER PRIMARY KEY, word TEXT, count_after INT, count_before INT, avg_pause_after FLOAT, avg_pause_before FLOAT, UNIQUE(word))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS pronunciations
        (prnc_index INTEGER PRIMARY KEY, word_index INTEGER, prnc TEXT, count INTEGER, count_after INT, count_before INT, avg_pause_after FLOAT, avg_pause_before FLOAT, UNIQUE(word_index, prnc))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS locations
        (loc_index INTEGER PRIMARY KEY, word_index INTEGER, prnc_index INTEGER, segment_file TEXT, line_num INT, start_time FLOAT, end_time FLOAT, speaker TEXT, UNIQUE(word_index, prnc_index, segment_file, line_num, start_time, end_time, speaker))""")
    conn.commit()
    return cur, conn


def compile_prnc_dict(as_ep_ids, file_name):
    """
    Construct a pronunciation dictionary for all words in the corpus.
    """
    prnc_dict = {}
    cur, conn = create_analysis_db(file_name)

    for ep_num in as_ep_ids:
        phoneme_file = \
            './alignment_data/alignments_json/{}_seg*.json'\
                                            .format('TAL' + str(ep_num))

        if len(glob.glob(phoneme_file)) == 0:
            print('\tNo alignments found for episode ' \
                  '{}. Skipping.'.format(ep_num))
            continue

	print('Compiling episode {}'.format(ep_num))

        for ff in glob.glob(phoneme_file):
            fname = os.path.split(ff)[-1].replace('_aligned.json', '')

            with open(ff) as df:
                algn = json.load(df)["words"]

                for val in algn:
                    word = val["alignedWord"].lower()
                    if word == 'sp':
                        continue
                    prnc_dict = grow_prnc_dict(val, word, prnc_dict, fname)

    grow_prnc_db(prnc_dict, cur, conn)
    return prnc_dict


def grow_prnc_db(prnc_dict, cur, conn):
    for word in prnc_dict.keys():
    	word1 = word.lower().strip()
    	vals = cur.execute("SELECT word_index, word FROM words WHERE word=?", (word1, )).fetchone()

    	if vals:
    	    word_index, word1 = vals

    	else:
    	    cur.execute("INSERT INTO words (word) VALUES (?)", (word1, ))
    	    word_index = cur.lastrowid
    	    conn.commit()

    	for prnc in prnc_dict[word].keys():
    	    vals = cur.execute("SELECT prnc_index, prnc, count FROM pronunciations WHERE word_index=? AND prnc=?", (word_index, prnc)).fetchone()
    	    count = prnc_dict[word1][prnc]['count']

    	    if vals:
        		prnc_index, prnc, prev_count = vals
          		cur.execute("UPDATE pronunciations SET count=(?) WHERE prnc_index=?", (prev_count + count, prnc_index))
        		conn.commit()

    	    else:
        		cur.execute("INSERT INTO pronunciations (word_index, prnc, count) VALUES (?, ?, ?)", (word_index, prnc, count))
        		prnc_index = cur.lastrowid
        		conn.commit()

            data = zip(prnc_dict[word][prnc]['locations'], prnc_dict[word][prnc]['timesteps'], prnc_dict[word][prnc]['speaker'])
    	    data =[(word_index, prnc_index) + i[0] + i[1] + (i[2], ) for i in data]
    	    cur.executemany("INSERT OR IGNORE INTO locations (word_index, prnc_index, line_num, segment_file, start_time, end_time, speaker) VALUES (?, ?, ?, ?, ?, ?, ?)", data)
    	    conn.commit()


def grow_prnc_dict(val, word, prnc_dict, seg_name):
    pronunc = ' '.join([vv[0] for vv in val["phonemes"]])

    # if this is the first time we've seen the word
    if word not in prnc_dict.keys():
        prnc_dict[word] = {}
        prnc_dict[word][pronunc] = \
            {"count": 1,
             "locations": [(val["line_idx"], seg_name)],
             "speaker":   [val["speaker"]],
             "timesteps": [(val["start"], val["end"])]}

    # otherwise, if we've seen the word before
    else:
        prev_pronunc = prnc_dict[word].keys()

        # if we have already seen this pronunciation, increment the count
        if pronunc in prev_pronunc:
            prnc_dict[word][pronunc]["count"] += 1
            prnc_dict[word][pronunc]["speaker"]\
                                .append(val["speaker"])
            prnc_dict[word][pronunc]["locations"]\
                                .append((val["line_idx"], seg_name))
            prnc_dict[word][pronunc]["timesteps"]\
                                .append((val["start"], val["end"]))

        # otherwise, make a new entry
        else:
            prnc_dict[word][pronunc] = \
                    {"count": 1,
                     "locations": [(val["line_idx"], seg_name)],
                     "speaker":   [val["speaker"]],
                     "timesteps": [(val["start"], val["end"])]}
    return prnc_dict


def find_num_pronunciations(prnc_dict):
    """
    Sorts the words in prnc_dict by the number of different
    pronunciations that exist for each. Returns a sorted 2d array
    where each row contains [word, n_pronunciations].
    """
    n_prncs = np.array(
       [[wrd, len(prnc_dict[wrd].keys())] for wrd in prnc_dict.keys()])
    idxs = n_prncs[:, 1].argsort()[::-1]
    return n_prncs[idxs]


def pause_dict_word(file_name, word, prnc_dict=None):
    """
    For testing. We can delete this once we're sure compile_pause_dict is
    working correctly
    """
    if not prnc_dict:
        pdict_fp = './alignment_data/pronunciations/' \
                   '{}_prnc_dict.pickle'.format(file_name)

        if os.path.lexists(pdict_fp):
            with open(pdict_fp, 'rb') as handle:
                prnc_dict = pickle.load(handle)
        else:
            raise OSError('Path {} does not exist!'.format(pdict_fp))

    pause_dict = {}
    prev_file = None
    path = './{}_bookworm/{}_phonemes/' \
           'texts/input.txt'.format(file_name, file_name)

    with open(path, 'r') as ff:
        for prnc in prnc_dict[word].keys():
            regex_after = r'(\{(sp|\d{1}\.\d{2})\} ' + \
                                prnc + r' \{(\d{1}\.\d{2})\})'
            regex_before = r'(\{(\d{1}\.\d{2})\} ' + \
                                prnc + r' \{(sp|\d{1}\.\d{2})\})'

            for line in ff.readlines():
                res_after  = re.findall(regex_after, line, re.IGNORECASE)
                res_before = re.findall(regex_before, line, re.IGNORECASE)

                if res_after or res_before:
                    print(line)
                    print(res_after)
                    print(res_before)
                    print('\n')


def compile_pause_db(file_name):
    """
    Creates a dictionary containing the average pause lengths both before and
    after a particular pronunciation. Requires bookworm files in the current
    directory and the pronunciation database to already exist.
    """
    cur, conn = create_analysis_db(file_name)

    prev_file = None
    path = './{}_bookworm/{}_phonemes/' \
           'texts/input.txt'.format(file_name, file_name)

    vals = cur.execute("SELECT word, word_index FROM words").fetchall()
    words = [i[0] for i in vals]
    ids  = [i[1] for i in vals]

    words = np.sort(words)
    n_words = len(words)
    word_idxs = np.array(ids)[np.argsort(words)]
    w_len = max([len(w) for w in words]) + 6 + len(str(n_words)) * 2

    with open(path, 'r') as ff:
        f = 'Reading pauses for {:<%d}{:^%d}'%(w_len, w_len)

        for idx, wrd in enumerate(words):
    	    pause_dict = {}
            pause_dict[wrd] = {}
            count_total_after, count_total_before = 0., 0.
            pause_total_after, pause_total_before = 0., 0.

            query = "SELECT prnc FROM pronunciations WHERE word_index=?"
            prncs = cur.execute(query, (word_idxs[idx], ))
            prncs = [i[0] for i in prncs.fetchall()]

            print(f.format('`{}`'.format(wrd),
                           '({} / {})'.format(idx + 1, n_words)))

            for prnc in prncs:
                pause_dict[wrd][prnc] = {}
                count_after, count_before = 0., 0.
                pause_after, pause_before = 0., 0.
                regex_after = r'(\{(sp|\d{1}\.\d{2})\} ' + prnc + r' \{(\d{1}\.\d{2})\})'
                regex_before = r'(\{(\d{1}\.\d{2})\} ' + prnc + r' \{(sp|\d{1}\.\d{2})\})'

                for line in ff.readlines():
                    res_after  = re.findall(regex_after, line, re.IGNORECASE)
                    res_before = re.findall(regex_before, line, re.IGNORECASE)

                    for ss1 in res_after:
                        count_after += 1
                        pause_after += float(ss1[2])

                    for ss2 in res_before:
                        count_before += 1
                        pause_before += float(ss2[1])

                ff.seek(0)

                if count_after:
                    pause_dict[wrd][prnc]['avg_after']  = \
                                        pause_after / count_after
                else:
                    pause_dict[wrd][prnc]['avg_after'] = 0.

                if count_before:
                    pause_dict[wrd][prnc]['avg_before'] = \
                                        pause_before / count_before
                else:
                    pause_dict[wrd][prnc]['avg_before'] = 0.

                pause_dict[wrd][prnc]['count_after']  = int(count_after)
                pause_dict[wrd][prnc]['count_before'] = int(count_before)

                count_total_after  += count_after
                count_total_before += count_before

                pause_total_after += pause_after
                pause_total_before += pause_before

            if count_total_after:
                pause_dict[wrd]['avg_after'] = \
                    pause_total_after / count_total_after
            else:
                pause_dict[wrd]['avg_after'] = 0.

            if count_total_before:
                pause_dict[wrd]['avg_before'] = \
                    pause_total_before / count_total_before
            else:
                pause_dict[wrd]['avg_before'] = 0.

            pause_dict[wrd]['count_after']   = int(count_total_after)
            pause_dict[wrd]['count_before']  = int(count_total_before)

	    grow_pause_db(conn, cur, pause_dict, wrd, word_idxs[idx])


def grow_pause_db(conn, cur, pause_dict, wrd, wrd_idx):
    word1 = wrd.lower().strip()
    prncs = pause_dict[wrd].keys()

    cur.execute("UPDATE words SET count_before=?, count_after=?, avg_pause_before=?, avg_pause_after=? WHERE word=?",
		(pause_dict[wrd]['count_before'], pause_dict[wrd]['count_after'], pause_dict[wrd]['avg_before'], pause_dict[wrd]['avg_after'], word1))
    conn.commit()

    query = "SELECT prnc_index, prnc FROM pronunciations WHERE word_index=?"
    vals = cur.execute(query, (wrd_idx, )).fetchall()
    # prnc_indexs = [i[0] for i in vals]
    # prncs_db = [i[1] for i in vals]

    data = []
    for idx, pr in vals:
    	data.append((pause_dict[wrd][pr]['avg_after'], pause_dict[wrd][pr]['avg_before'], pause_dict[wrd][pr]['count_after'], pause_dict[wrd][pr]['count_before'], idx))

    cur.executemany("UPDATE pronunciations SET avg_pause_after=?, avg_pause_before=?, count_after=?, count_before=? WHERE prnc_index=?", data)
    conn.commit()


def sort_by_pause_length(file_name, pause_dict=None):
    if not pause_dict:
        pdict_fp = './alignment_data/{}_pause_dict.pickle'.format(file_name)

        if os.path.lexists(pdict_fp):
            with open(pdict_fp, 'rb') as handle:
                pause_dict = pickle.load(handle)
        else:
            raise OSError('Path {} does not exist!'.format(pdict_fp))

    pause_before, pause_after = [], []
    for wrd in pause_dict.keys():
        wrd_dict = pause_dict[wrd]

        entry_before = (wrd, wrd_dict['avg_before'], wrd_dict['count_before'])
        entry_after  = (wrd, wrd_dict['avg_after'],  wrd_dict['count_after'])

        pause_before.append(entry_before)
        pause_after.append(entry_after)

    pause_before = np.asarray(pause_before)
    pause_after  = np.asarray(pause_after)

    idx_before = np.argsort(pause_before[:, 1])[::-1]
    idx_after  = np.argsort(pause_after[:, 1])[::-1]

    idx_count_before = np.argsort(pause_before[:, 2])[::-1]
    idx_count_after = np.argsort(pause_after[:, 2])[::-1]

    return pause_before[idx_before], pause_after[idx_after], pause_before[idx_count_before], pause_after[idx_count_after]


def plot_pause_words(show_name, file_name, pause_dict=None):
    """
    Show words with the largest Avg. Pause Length After * Word Count
    products
    """
    sns.set(style="white")

    _, _, c, d = sort_by_pause_length(file_name, pause_dict=pause_dict)

    rr_after = np.array(
        [(w, float(j) * float(i), i, j) for w, i, j in d if float(i) != 0.])
    rr_before = np.array(
        [(w, float(j) * float(i), i, j) for w, i, j in c if float(i) != 0.])

    idxs_after  = np.argsort(rr_after[:, 1].astype(float))[::-1]
    idxs_before = np.argsort(rr_before[:, 1].astype(float))[::-1]

    ratio_a = rr_after[idxs_after]
    ratio_b = rr_before[idxs_before]

    labels_a, labels_b = [], []
    for ii in range(10):
        ll = '\nC: {0}\nP: {1:.5}'.format(ratio_a[ii, 3], ratio_a[ii, 2])
        labels_a.append(ratio_a[ii, 0] + ll)

        ll = '\nC: {0}\nP: {1:.5}'.format(ratio_b[ii, 3], ratio_b[ii, 2])
        labels_b.append(ratio_b[ii, 0] + ll)

    yax_a = ratio_a[:10, 1].astype(float)
    yax_b = ratio_b[:10, 1].astype(float)

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



def compile_supercut(prnc_dict, word, file_name, show_name):
    pad_length = 500. # in milliseconds
    prev_file = None
    word_dict = prnc_dict[word]
    silence = AudioSegment.silent(duration=1000)

    for prnc in word_dict.keys():
        supercut = AudioSegment.empty()
        locations = np.asarray(word_dict[prnc]['locations'])
        timesteps = np.asarray(word_dict[prnc]['timesteps']) * 1000.

        for ((line, file_id), (start, stop)) in zip(locations[:30], timesteps[:30]):
            if prev_file == file_id:
                pass

            else:
                path = './alignment_data/seg_audio/{}.wav'.format(file_id)
                audio = AudioSegment.from_file(path, format='wav', channels=1,
                                               sample_width=2)
                seg_len = (audio.frame_count() / audio.frame_rate) * 1000.
                prev_file = file_id

            if start < pad_length:
                start = pad_length

            if stop + pad_length > seg_len:
                stop = seg_len - pad_length

            supercut += silence + audio[start - pad_length:stop + pad_length]


        supercut_fp = "./alignment_data/supercuts/{}_{}_{}.wav"\
                                            .format(word, prnc, file_name)
        tag_dict = {"word": word, "pronunciation": prnc, "show": show_name}
        supercut.export(supercut_fp, format="wav", tags=tag_dict)
