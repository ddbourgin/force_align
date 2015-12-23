#!/usr/bin/python

from audiosearch import Client
from pydub import AudioSegment
import pandas as pd
import numpy as np
import subprocess
import psycopg2
import dotenv
import re
import os
import csv
import cgi
import json
import glob
import click
import pickle
import shutil
import datetime
import requests
import Levenshtein
import seaborn as sns
import matplotlib.pyplot as plt

import align

def make_bw_directories(file_name):
    if not os.path.lexists('./metadata'):
        os.mkdir('./metadata')
    if not os.path.lexists('./texts'):
        os.mkdir('./texts')
    if not os.path.lexists('./{}'.format(file_name)):
        os.mkdir('./'+file_name)
    if not os.path.lexists('./{}_phonemes'.format(file_name)):
        os.mkdir('./{}_phonemes'.format(file_name))
    if not os.path.lexists('./{}_phonemes/metadata'.format(file_name)):
        os.mkdir('./'+file_name+'_phonemes/metadata')
    if not os.path.lexists('./{}_phonemes/texts'.format(file_name)):
        os.mkdir('./'+file_name+'_phonemes/texts')


def load_timesteps(file_name):
    ts = np.genfromtxt("./alignment_data/timesteps_{}.csv".format(file_name),
                       delimiter=b",", dtype=None)
    return np.array([list(i) for i in ts])


def get_timestep_chunks(ep_num, ts_csv):
    row = np.argwhere(ts_csv[:, 0] == str(int(ep_num))).ravel()[0]
    chunks = [ii.split(' :: ') for ii in ts_csv[row, 1:]]
    chunks = [tuple([float(ii), float(jj)]) for [ii,jj] in chunks]
    return chunks


def create_timesteps_csv(n_segs, file_name):
  with open('../timesteps_{}.csv'.format(file_name), 'wt') as fp:
    writer = csv.writer(fp, delimiter=b",", quoting=csv.QUOTE_MINIMAL)
    header = ['Seg_'+ str(i) for i in range(1, n_segs + 1)]
    writer.writerow( ['Ep_Number'] + header )


def make_alignments_directory():
    if not os.path.lexists('./alignment_data'):
        os.mkdir('./alignment_data')
    if not os.path.lexists('./alignment_data/alignments_json'):
        os.mkdir('./alignment_data/alignments_json')
    if not os.path.lexists('./alignment_data/seg_json'):
        os.mkdir('./alignment_data/seg_json')
    if not os.path.lexists('./alignment_data/seg_audio'):
        os.mkdir('./alignment_data/seg_audio')
    if not os.path.lexists('./alignment_data/full_audio'):
        os.mkdir('./alignment_data/full_audio')
    if not os.path.lexists('./alignment_data/pronunciations'):
        os.mkdir('./alignment_data/pronunciations')
    if not os.path.lexists('./alignment_data/supercuts'):
        os.mkdir('./alignment_data/supercuts')


def init_as_client():
    vv = dotenv.get_variables('.env')
    key = str(vv[u'AS_ID'])
    secret = str(vv[u'AS_SECRET'])
    return Client(key, secret)


def db_connect():
    vv = dotenv.get_variables('.env')
    dbname = str(vv[u'dbname'])
    user = str(vv[u'db_user'])
    pwd = str(vv[u'db_pwd'])
    host = str(vv[u'db_host'])
    conn_string = "host='%s' dbname='%s' " \
                  "user='%s' password='%s'"%(host, dbname, user, pwd)
    return psycopg2.connect(conn_string)


def get_transcript(db, trans_id):
    query = """SELECT "timed_texts".*
               FROM "timed_texts"
               WHERE "timed_texts"."transcript_id" IN (%s)
               ORDER BY start_time ASC"""%str(trans_id)
    return pd.read_sql_query(query, con=db)


def get_episode_info(db, ep_id):
    query = """SELECT "items".*
               FROM "items"
               WHERE "items"."id" IN (%s)"""%str(ep_id)
    return pd.read_sql_query(query, con=db)


def collect_episode_metadata(db, ep_id, as_client):
    # first try getting things from as db, otherwise query the API
    meta = {}
    try:
        ep_data = get_episode_info(db, ep_id)
        meta['title'] = ep_data.title[0].replace('\xc3\x89', 'E')
        meta['tags'] = ep_data.tags[0]
        meta['link'] = ep_data.digital_location[0]
        meta['airdate'] = ep_data.date_broadcast[0]

    except IndexError:
        episode = as_client.get_episode(ep_id)
        meta['title'] = episode["title"].replace('\xc3\x89', 'E')
        meta['tags'] = episode["tags"]
        meta['link'] = episode["digital_location"]
        meta['airdate'] = None

    if not meta['airdate']:
        meta['airdate'] = 'Unknown'

    if type(meta['airdate']) is datetime.date:
        meta['airdate'] = meta['airdate'].strftime("%Y-%m-%d")

    return meta


def load_item_key():
    with open("./data/itemid_transcriptid.csv", "rb") as ff:
        return np.loadtxt(ff, delimiter=b",")


def append_timestamps_csv(csv_line, file_name):
    with open('../timesteps_{}.csv'.format(file_name), 'ab') as fp:
        writer = csv.writer(fp, delimiter=b",", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(csv_line)


def clean_text(text):
    return unicode(text, 'utf-8').strip().replace('\n', '').replace('\t', '')


def rename_file(audio_url, ep_id):
    files = os.listdir('.')
    files.sort(key=os.path.getmtime)
    os.rename(files[-1], '{}.mp3'.format(ep_id))


def find_episodes(show_name, as_client):
    as_show_id = None
    res = as_client.search({ 'q': show_name }, 'shows')

    # find the corresponding audiosearch show id for show_name
    for show in res["results"]:
        if show["title"].lower() == show_name.lower():
            as_show_id = show["id"]
            break

    if not as_show_id:
        IndexError('Unable to find show `{}` in AS database'.format(show_name))

    # given the audiosearch show id, find the AS episode numbers
    else:
        show_info = as_client.get_show(as_show_id)
        as_ep_ids = show_info["episode_ids"]
        return as_ep_ids, as_show_id


def find_episode_transcript_ids(as_ep_ids):
    trans_dict = {}
    epid_transid_key = load_item_key()

    for ep_num in as_ep_ids:
        try:
            trans_idx = np.argwhere(epid_transid_key[:,0] == ep_num).ravel()[0]
            trans_dict[ep_num] = epid_transid_key[trans_idx, 1]
        except IndexError:
            print('Unable to find transcript ' \
                  'for AS episode {}'.format(ep_num))
    return trans_dict


def compile_audio_and_transcripts(trans_dict, n_segs, as_client, file_name):
    db = db_connect()

    for (ep_id, trans_id) in trans_dict.items():
        transcript = compile_episode_transcript(trans_id, db)

        if len(transcript) == 0:
            print('Unable to find transcript ID {}'
                  ' in AS database'.format(trans_id))
            continue
        prepare_for_alignment(transcript, ep_id, as_client, n_segs, file_name)


def segment_audio_and_transcript(transcript, ep_id, n_segs, file_name):
    if not os.path.exists('../timesteps_{}.csv'.format(file_name)):
        create_timesteps_csv(n_segs, file_name)

    times, time_idxs = [0.], [0]
    ep_audio = AudioSegment.from_file('{}.mp3'.format(ep_id),
                        format='mp3', channels=1, sample_width=2)
    csv_line = [ep_id]
    fs = float(ep_audio.frame_rate)
    ep_length = (ep_audio.frame_count() / fs) / 60. # in minutes
    timesteps = [np.round(float(i), 3) for i in transcript[:, 0]]

    # if we aren't segmenting the file
    if n_segs <= 1:
        start = min(timesteps)
        end = max(timesteps)
        seg = '{0:.3f} '.format(start) + ':: {0:.3f}'.format(end)
        csv_line.append(seg)
        append_timestamps_csv(csv_line, file_name)
        return [0, len(timesteps)-1]

    # add 0.5s silence to the start + end of an audio segment
    silence = AudioSegment.silent(duration=500)
    division = ep_length / float(n_segs)

    # calc the most appropriate timesteps to segment at given n_segs, and
    # add them to timesteps.csv
    for ii in np.arange(1, n_segs + 1):
        seg_ts = min(timesteps, key=lambda x:abs(x - (ii * division)))
        times.append(seg_ts)

        idx = np.argwhere(timesteps == np.round(times[-1], 3)).ravel()[0]
        time_idxs.append(idx)

        start_in_ms = times[ii - 1] * 60. * 1000.
        end_in_ms = times[ii] * 60. * 1000.

        aud_slice = silence + ep_audio[start_in_ms:end_in_ms] + silence
        fn = '../seg_audio/{}_seg{}.wav'.format(ep_id, ii)
        aud_slice.export(fn, 'wav')

        seg = '{0:.3f} '.format(times[-2]) + ':: {0:.3f}'.format(times[-1])
        csv_line.append(seg)

    append_timestamps_csv(csv_line, file_name)
    time_idxs.append(len(timesteps) - 1)
    return time_idxs


def prepare_for_alignment(transcript, ep_id, as_client, n_segs, file_name):
    """
    Downloads the audio for an episode and segments it into bite-sized chunks.
    Writes the transcript for each audio segment to an individual json file
    stored in ./alignment_data/seg_json/
    """
    episode = as_client.get_episode(ep_id)
    audio_url = episode["digital_location"]

    os.chdir('./alignment_data/full_audio/')

    # first try downloading the audio from soundcloud
    # (suppress stderr to avoid cluttering the console if link is rotted)
    with open(os.devnull, 'w') as devnull:
        res = subprocess.call(["soundscrape", audio_url], stderr=devnull)

    if res == 0:
        rename_file(audio_url, ep_id)

    # if soundcloud fails, try looking for the audio link elsewhere
    else:
        try:
            links = episode['audio_files'][0]['url']
            audio_url = [ii for ii in links if '.mp3' in ii]
            resp = requests.get(audio_url[0])

            with open('{}.mp3'.format(ep_id), 'w') as ff:
                ff.write(resp.content)
                ff.close()

        except:
            # if we're still unable to download the audio, open the debugger
            import ipdb; ipdb.set_trace()
            return

    # segment audio & transcript
    seg_row_ids = segment_audio_and_transcript(transcript, ep_id, n_segs, file_name)

    # write the segmented transcript to individual json files
    write_transcript_segments(transcript, seg_row_ids, ep_id)

    os.chdir('../../')


def write_transcript_segments(transcript, seg_row_ids, ep_id):
    """
    Write transcript segments to individual json files for use with p2fa.
    """
    event_regex = r'\[.*\]'

    for ii in range(len(seg_row_ids) - 1):
        start_row = seg_row_ids[ii]
        end_row = seg_row_ids[ii + 1]
        trans = transcript[start_row:end_row, :]
        slice_id = '_seg{}'.format(ii + 1)

        tscrpit = []
        file_name = str(ep_id) + slice_id + '.json'
        file_name = '../seg_json/' + file_name

        for line in trans:
            speaker = str(line[3]).upper()
            utter = re.sub(event_regex, "", line[2])
            utter = utter.replace(' :', ':')\
                         .replace('].', ']')\
                         .replace('--', '')\
                         .replace('  ', ' ')\
                         .replace(' [UNINTELLIGIBLE]', '')\
                         .replace('-',' ')\
                         .replace('&', 'and')\
                         .replace('[? ', '')\
                         .replace(' ?]', '')\
                         .replace(' !', '!')\
                         .replace(' ?', '?')\
                         .replace(' ,', ',')\
                         .replace(' .', '.')\
                         .replace('#', '')\
                         .replace('^', '')\
                         .replace('*', '')
            catalog = {'speaker': speaker, 'line': utter}
            tscrpit.append(catalog)

        with open(file_name, 'wb') as f:
            json.dump(tscrpit, f)
            f.close()


def compile_episode_transcript(trans_id, db):
    transcript = []
    trans = get_transcript(db, trans_id).sort_values(by="start_time")

    # line contents: [start_time, end_time, utterance, speaker_id]
    for idx in range(trans.shape[0]):
        speaker = trans['speaker_id'][idx]
        text = clean_text(trans['text'][idx])
        start = trans['start_time'][idx]/60.
        end = trans['end_time'][idx]/60.

        if speaker is None or np.isnan(speaker):
          speaker = -1

        if text == '.':
          continue

        line = [start, end, text, speaker]

        # skip duplicate lines
        if idx > 0 and line[2] == transcript[-1][2]:
          continue

        transcript.append(line)
    return np.asarray(transcript)



def align_transcripts(as_ep_ids):
    """
    Align the transcript segments in ./alignment_data/seg_json/ to episode
    audio using p2fa. If there are any errors during alignment, skip the
    episode on which they occur and move on.
    """
    problem_episodes = []

    for ep_num in as_ep_ids:
        wavfile = './alignment_data/seg_audio/{}_seg*.wav'.format(ep_num)

        for ff in glob.glob(wavfile):
            file_name = os.path.split(ff)[-1].split('.')[0]

            trsfile = './alignment_data/seg_json/{}.json'.format(file_name)
            outfile = './alignment_data/alignments_json/' \
                      '{}_aligned.json'.format(file_name)

            if os.path.lexists(outfile):
                fn = os.path.split(outfile)[-1]
                print('\t{} already exists! Skipping'.format(fn))
                continue

            # try:
            print('\n\tAligning {}...'.format(file_name))
            align.do_alignment(ff, trsfile, outfile, json=True,
                                   textgrid=False, phonemes=True,
                                   breaths=False)
            # except:
            #     print('\n\tError Aligning {}'.format(file_name))
            #     problem_episodes.append(ep_num)
            #     as_ep_ids.remove(ep_num)
            #     break

    return as_ep_ids, problem_episodes


def write_bw_catalog(transcript, phoneme_transcript, counter, ep_num, meta,
                     file_name):
    for idx,line in enumerate(transcript):
        if str(idx) in phoneme_transcript[:,2]:
            counter += 1
            ts = int(float(line[0])) # recast minutes to int for bookworm
            row = np.argwhere(phoneme_transcript[:,2] == str(idx)).ravel()[0]
            utter = str(phoneme_transcript[row, 1].encode('ascii', 'ignore'))
            txt = cgi.escape(unicode(utter, 'utf-8'), quote=True)
            link = ' [<a href="{}">{}</a>]'\
                            .format(meta['link'], meta['title'].title())
            sstring = txt + link

            catalog = \
                {"searchstring": sstring,
                 "filename": unicode(str(counter), 'utf-8'),
                 "year": meta['airdate'],
                 "Topics": meta['tags'],
                 "transcript": meta['link'],
                 "ep_number": int(ep_num),
                 "Speaker_ID": unicode(str(line[3]), 'utf-8'),
                 "minutes": ts}

            catalog = json.dumps(catalog)

            # remove pauses from sql database string
            db_string = line[2].encode('ascii', 'ignore')
            text_string = '{}\t{}\n'.format(counter, db_string)

            with open('./texts/input.txt', 'a') as ff:
                ff.write(text_string)
                ff.close()

            with open('./metadata/jsoncatalog.txt', 'a') as ff:
                ff.write(str(catalog)+'\n')
                ff.close()

            # fix this
            modal_phone = \
            '<div class="modal fade" id="#sentmodal' + str(counter) + '" role="dialog"> <div class="modal-dialog modal-sm"> <div class="modal-content"> <div class="modal-header"> <button type="button" class="close" data-dismiss="modal">&times;</button><h4> </h4></div> <div class="modal-body"> <p>' + txt + '</p> </div> </div> </div>'

            link_phone = ' [<a href="' + meta['link'] + '">'+ meta['title'].title() +'</a>, <a href="#" data-toggle="modal" data-target="#sentmodal' + str(counter) + '">sentence</a>]' + modal_phone

            # TODO: for phone_catalog, i need to include a link to the sstring sentence that the phonemes here correspond to
            phone_catalog = \
                {"searchstring": phoneme_transcript[row, 0] + link_phone,
                 "filename": unicode(str(counter), 'utf-8'),
                 "year": meta['airdate'],
                 "Topics": meta['tags'],
                 "transcript": meta['link'],
                 "ep_number": int(ep_num),
                 "Speaker_ID": unicode(str(line[3]), 'utf-8'),
                 "minutes": ts}

            phone_catalog = json.dumps(phone_catalog)
            phoneme_line = phoneme_transcript[row, 0].encode('ascii', 'ignore')
            text_string_phonemes = '{}\t{}\n'.format(counter, phoneme_line)

            with open('./{}_phonemes/texts/input.txt'.format(file_name), 'a') as ff:
                ff.write(text_string_phonemes)
                ff.close()

            with open('./'+file_name+'_phonemes/metadata/jsoncatalog.txt', 'a') as f:
                f.write(str(phone_catalog) + '\n')
                f.close()
    return counter


def clean_sentence(sentence):
    event_regex = r'\[.*\]'
    cleaned_sentence = re.sub(r'{.*?}', '', sentence)\
      .replace(' :', ':')\
      .replace('].', ']')\
      .replace('--', '')\
      .replace('  ', ' ')\
      .replace(' [UNINTELLIGIBLE]', '')\
      .replace('-',' ')\
      .replace('&', 'and')\
      .replace('[? ', '')\
      .replace(' ?]', '')\
      .replace(' ]',']')\
      .replace('#!MLF!#', '')\
      .replace('", ', '"')\
      .replace('"! ', '"')

    cleaned_sentence = re.sub(event_regex, "", cleaned_sentence)
    cleaned_sentence = cleaned_sentence\
                            .replace('  ', ' ')\
                            .replace('?] ', '')\
                            .strip()
    return cleaned_sentence



def find_line_in_transcript(transcript, sentence):
    """
    Uses the Levenshtein (i.e., edit) distance between a sentence and the
    lines in transcript to identify the row most likely to correspond to
    sentence in transcript
    """
    edit_dist = []
    clean_sent = clean_sentence(sentence)

    for idx, line in enumerate(transcript):
        trans_line = line[2]
        edit_dist.append(Levenshtein.distance(clean_sent, trans_line))
    idx = np.argmin(edit_dist)
    return idx


def compile_phoneme_transcript(ff, ep_num, ts_csv, transcript):
    """
    Reads the phoneme alignments produced by p2fa and matches them to their
    corresponding line in transcript. Returns phoneme_transcript, a list of
    lists where each sublist contains

        [phoneme transcription, transcript line, transcript row index]
    """
    phoneme_transcript = []
    file_name = os.path.split(ff)[-1].split('.')[0]

    with open(ff) as data_file:
        line_dicts, line_num = [], []
        word_list = json.load(data_file)["words"]

        for w_dict in word_list:

            # if word token NOT a pause...
            if 'line_idx' in w_dict.keys():
                line_num.append(w_dict['line_idx'])

            # if word token IS a pause
            else:
                start = w_dict['start'] / 60.
                end = w_dict['end'] / 60.
                w_dict['word'] = '{' + '{0:.2f}'.format((end - start) * 60.) + '}'

            # if new token comes from the same line as the previous tokens
            if len(set(line_num)) < 2:
                line_dicts.append(w_dict)
                continue

            # if we have reached the end of a line, construct the aligned
            # sentence and match it to our transcript
            else:
                next_line = line_num.pop()
                sentence = ' '.join([ll['word'] for ll in line_dicts])
                trans_idx = find_line_in_transcript(transcript, sentence)
                sentence_phones = compile_phoneme_sentence(line_dicts)

                line = [sentence_phones, sentence, trans_idx]
                phoneme_transcript.append(line)

            # reset line_dicts and line_num for the next line in transcript
            line_dicts, line_num = [w_dict], [next_line]
    return phoneme_transcript



def compile_phoneme_sentence(line_dicts):
    """
    Gathers the phonemes for each word in a transcript row and concatenates
    them into a single string to be added to phoneme_transcript
    """
    sentence_phones = []

    for ll in line_dicts:
        if 'phonemes' in ll.keys():
            phones = [ph[0] for ph in ll['phonemes']]
            sentence_phones = sentence_phones + phones

            # signify the end of a word using the {sp} token
            sentence_phones = sentence_phones + ['{sp}']

        elif 'alignedWord' in ll.keys() and ll['alignedWord'] == 'sp':
            # remove redundant {sp} token if we are inserting a
            # pause directly afterward
            if len(sentence_phones) != 0 and sentence_phones[-1] == '{sp}':
                sentence_phones.pop()
                sentence_phones = sentence_phones + [ll['word']]

    sentence_phones = ' '.join(sentence_phones)
    return sentence_phones



def write_field_descriptions(file_name):
    field_descriptions = \
    [
        {"field":"Speaker_ID", "datatype":"categorical", "type":"text",
         "unique":False},
        {"field":"transcript", "datatype":"categorical", "type":"text",
         "unique":False},
        {"field":"Topics", "datatype":"categorical", "type":"text",
         "unique":False},
        {"field":"minutes", "datatype":"time", "type":"integer",
         "unique":False, "derived":[ {"resolution":"year"} ]},
        {"field":"year", "datatype":"time", "type":"character",
         "unique":False, "derived":[ {"resolution":"year"} ]},
        {"field":"searchstring", "datatype":"searchstring", "type":"text",
         "unique":True},
        {"field":"ep_number", "datatype":"time", "type":"integer",
         "unique":False, "derived":[ {"resolution":"year"} ]}
    ]

    with open('./metadata/field_descriptions.json', 'w') as f:
        json.dump(field_descriptions, f)
        f.close()

    with open('./'+file_name+'_phonemes/metadata/field_descriptions.json', 'w') as f:
        json.dump(field_descriptions, f)
        f.close()




def compile_alignments_bookworm(as_ep_ids, file_name):
    """
    Reads the phoneme alignments for each episode, matches them to
    the corresponding line in transcript, gets the appropriate episode
    metadata, and constructs the corresponding catalog.json and input.txt
    files for both a phoneme bookworm and a regular bookworm.
    """
    counter = 0
    db = db_connect()
    as_client = init_as_client()
    make_bw_directories(file_name)
    ts_csv = load_timesteps(file_name)
    trans_dict = find_episode_transcript_ids(as_ep_ids)

    for (ep_num, trans_id) in trans_dict.items():
        phoneme_file = \
            './alignment_data/alignments_json/{}_seg*.json'.format(ep_num)

        if len(glob.glob(phoneme_file)) == 0:
            print('No phoneme alignments for episode' \
                  ' {}. Skipping.'.format(ep_num))
            continue

        transcript = compile_episode_transcript(trans_id, db)
        meta = collect_episode_metadata(db, ep_num, as_client)

        if len(transcript) == 0:
            print('Unable to find transcript ID {} ' \
                  'in AS database. Skipping.'.format(trans_id))
            continue

        phoneme_transcript = []

        for ff in np.sort(glob.glob(phoneme_file)):
            phoneme_transcript = phoneme_transcript + \
                compile_phoneme_transcript(ff, ep_num, ts_csv, transcript)

        print('Writing BW entry for {}'.format(phoneme_file))

        # phoneme_transcript[idx] =
        # [phoneme sentence, transcript sentence, transcript index]
        phoneme_transcript = np.asarray(phoneme_transcript)
        counter = write_bw_catalog(transcript, phoneme_transcript,
                                   counter, ep_num, meta, file_name)
        write_field_descriptions(file_name)

    shutil.move('./metadata', './' + file_name)
    shutil.move('./texts', './' + file_name)

    os.mkdir('./' + file_name + '_bookworm')

    shutil.move('./' + file_name, './' + file_name + '_bookworm')
    shutil.move('./' + file_name + '_phonemes', './' + file_name + '_bookworm')

# @click.command()
# @click.option('--n_segs', default=2,
#                 help='number of segments to split the transcript into ' \
#                 'for forced alginment (more segments ~> faster alignment)')
# @click.argument('show_name')
# @click.argument('file_name')
def align_show(show_name, n_segs, file_name):
    """
    Finds all the episode ids associated with show_name in the audiosearch db,
    find the corresponding transcript ids for each episode, downloads the
    episode audio from soundcloud, compiles the episode transcript from the
    audiosearch db, segments the audio and transcripts into bite-sized
    segments, runs each segment through p2fa to get phoneme-level
    alignments, and writes both regular and phoneme-level bookworm files.
    """
    make_alignments_directory()
    as_client = init_as_client()

    as_ep_ids, as_show_id = find_episodes(show_name, as_client)
    trans_dict = find_episode_transcript_ids(as_ep_ids)
    compile_audio_and_transcripts(trans_dict, n_segs, as_client, file_name)

    as_ep_ids, problem_episodes = align_transcripts(as_ep_ids)
    compile_alignments_bookworm(as_ep_ids, file_name)



def compare_pronunciations(word, show_name, file_name):
    word = word.lower()
    as_client = init_as_client()
    as_ep_ids, as_show_id = find_episodes(show_name, as_client)

    pdict_fp = './alignment_data/pronunciations/' \
               '{}_prnc_dict.pickle'.format(file_name)

    if os.path.lexists(pdict_fp):
        with open(pdict_fp, 'rb') as handle:
            prnc_dict = pickle.load(handle)
    else:
        prnc_dict = compile_prnc_dict(as_ep_ids, pdict_fp)

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



def compile_prnc_dict(as_ep_ids, pdict_fp):
    """
    Construct a pronunciation dictionary for all words in the corpus and
    cache it in ./pronuncations for future queries.
    """
    prnc_dict = {}
    for ep_num in as_ep_ids:
        phoneme_file = \
            './alignment_data/alignments_json/{}_seg*.json'.format(ep_num)

        if len(glob.glob(phoneme_file)) == 0:
            print('No alignments found for episode ' \
                  '{}. Skipping.'.format(ep_num))
            continue

        for ff in glob.glob(phoneme_file):
            fname = os.path.split(ff)[-1].replace('_aligned.json', '')

            with open(ff) as df:
                algn = json.load(df)["words"]

                for val in algn:
                    word = val["alignedWord"].lower()
                    if word == 'sp':
                        continue
                    prnc_dict = grow_prnc_dict(val, word, prnc_dict, fname)

    with open(pdict_fp, 'wb') as handle:
        pickle.dump(prnc_dict, handle)

    return prnc_dict


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
            regex_after = r'(\{(sp|\d{1}\.\d{2})\} ' + prnc + r' \{(\d{1}\.\d{2})\})'
            regex_before = r'(\{(\d{1}\.\d{2})\} ' + prnc + r' \{(sp|\d{1}\.\d{2})\})'

            for line in ff.readlines():
                res_after  = re.findall(regex_after, line, re.IGNORECASE)
                res_before = re.findall(regex_before, line, re.IGNORECASE)

                if res_after or res_before:
                    print(line)
                    print(res_after)
                    print(res_before)
                    print('\n')








def compile_pause_dict(file_name, prnc_dict=None):
    """
    Requires bookworm files in the current directory.
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
        words = np.sort(prnc_dict.keys())
        n_words = len(words)
        w_len = max([len(w) for w in words]) + 6 + len(str(n_words)) * 2
        f = 'Reading pauses for {:<%d}{:^%d}'%(w_len, w_len)

        for idx, wrd in enumerate(words):
            pause_dict[wrd] = {}
            count_total_after, count_total_before = 0., 0.
            pause_total_after, pause_total_before = 0., 0.
            print(f.format('`{}`'.format(wrd),
                           '({} / {})'.format(idx + 1, n_words)))

            for prnc in prnc_dict[wrd].keys():
                pause_dict[wrd][prnc] = {}
                count_after, count_before = 0., 0.
                pause_after, pause_before = 0., 0.
                regex_after = r'(\{(sp|\d{1}\.\d{2})\} ' + prnc + r' \{(\d{1}\.\d{2})\})'
                regex_before = r'(\{(\d{1}\.\d{2})\} ' + prnc + r' \{(sp|\d{1}\.\d{2})\})'

                # regex_after = r'\b' + prnc + r' \{\d{1}\.\d{2}\}'
                # regex_before = r'\{\d{1}\.\d{2}\} ' + prnc + r'\b'

                for line in ff.readlines():
                    res_after  = re.findall(regex_after, line, re.IGNORECASE)
                    res_before = re.findall(regex_before, line, re.IGNORECASE)

                    for ss1 in res_after:
                        count_after += 1
                        pause_after += float(ss1[2])
                        # pause_after += float(ss1.upper().replace(prnc + ' {', '').replace('}', ''))

                    for ss2 in res_before:
                        count_before += 1
                        pause_before += float(ss2[1])
                        # pause_before += float(ss2.upper().replace('} ' + prnc, '').replace('{', ''))

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

    pause_dict_fp = './alignment_data/{}_pause_dict.pickle'.format(file_name)
    with open(pause_dict_fp, 'wb') as handle:
        pickle.dump(pause_dict, handle)

    return pause_dict


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


def plot_pause_words(show_name, file_name):
    """
    Plots the words with the largest Avg. Pause Length After / Word Count ratios
    """
    sns.set(style="white")

    _,_,c,d = sort_by_pause_length(file_name) # we only need one list
    rr_after = np.array([(w, float(j) * float(i), i, j) for w, i, j in d if float(i) != 0.])
    rr_before = np.array([(w, float(j) * float(i), i, j) for w, i, j in c if float(i) != 0.])

    idxs_after = np.argsort(rr_after[:, 1].astype(float))[::-1]
    idxs_before = np.argsort(rr_before[:, 1].astype(float))[::-1]

    ratio_a = rr_after[idxs_after]
    ratio_b = rr_before[idxs_before]

    labels_a, labels_b = [], []
    for ii in range(10):
        ll = '\nC: {0}\nP: {1:.5}'.format(ratio_a[ii, 3], ratio_a[ii, 2])
        labels_a.append(ratio_a[ii,0] + ll)
        ll = '\nC: {0}\nP: {1:.5}'.format(ratio_b[ii, 3], ratio_b[ii, 2])
        labels_b.append(ratio_b[ii,0] + ll)


    yax_a = ratio_a[:10, 1].astype(float)
    yax_b = ratio_b[:10, 1].astype(float)

    title_a = 'Words with Top Word Count * Avg. Succeeding Pause Length\nProducts in {}'.format(show_name)
    title_b = 'Words with Top Word Count * Avg. Preceding Pause Length\nProducts in {}'.format(show_name)

    idxs_a = np.argsort(yax_a)[::-1] # for consistency across plots
    idxs_b = np.argsort(yax_b)[::-1] # for consistency across plots

    fig, ax = plt.subplots()
    ax = sns.barplot(np.array(labels_a)[idxs_a], yax_a[idxs_a],
                     palette="Set3", ax=ax)
    ax.set_ylabel("Word Count * Avg. Pause Length")
    ax.set_title(title_a)
    save_path = './alignment_data/{}_pause_after.png'.format(file_name)
    plt.savefig(save_path)
    plt.close()

    fig, ax = plt.subplots()
    ax = sns.barplot(np.array(labels_b)[idxs_b], yax_b[idxs_b],
                     palette="Set3", ax=ax)
    ax.set_ylabel("Word Count * Avg. Pause Length")
    ax.set_title(title_b)
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


# if __name__ == '__main__':
    # show_name = "Political Gabfest"
    # n_segs = 10
    # file_name = 'gabfest'
    # align_show(show_name, n_segs, file_name)
