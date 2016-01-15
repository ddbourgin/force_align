#!/usr/bin/python

from pydub import AudioSegment
import numpy as np
import subprocess
import os
import cgi
import json
import glob
import click
import shutil
import requests
import Levenshtein

from db_utils import Postgres_Connect

import align

from utils import make_bw_directories, load_timesteps, get_timestep_chunks, \
                  create_timesteps_csv, make_alignments_directory, \
                  init_as_client, get_transcript, get_episode_info, \
                  load_item_key, append_timestamps_csv, clean_text, \
                  rename_file, clean_sentence, find_episodes, \
                  find_episode_transcript_ids, collect_episode_metadata

from analysis import compare_pronunciations, plot_pronunciations, \
                     compile_prnc_dict, grow_prnc_dict, \
                     find_num_pronunciations, pause_dict_word, \
                     compile_pause_dict, sort_by_pause_length, \
                     plot_pause_words, compile_supercut


def compile_audio_and_transcripts(trans_dict, n_segs, as_client, file_name):
    db = Postgres_Connect().connection()
    # db = db_connect()

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
    if os.path.lexists('./{}.mp3'.format(ep_id)):
        print("Episode file {}.mp3 already exists! Skipping".format(ep_id))
        os.chdir('../../')
        return

    # first try downloading the audio from soundcloud
    # (suppress stderr to avoid cluttering the console if link is rotted)
    with open(os.devnull, 'w') as devnull:
        try:
            res = subprocess.call(["soundscrape", audio_url], stderr=devnull)
        except:
            print('Error decoding {}.mp3. Skipping'.format(ep_id))
            os.chdir('../../')
            return

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
            utter = clean_sentence(line[2])
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
    # db = db_connect()
    db = Postgres_Connect().connection()
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

    prnc_dict = compile_prnc_dict(as_ep_ids, pdict_fp)
    pause_dict = compile_pause_dict(file_name, prnc_dict)

    plot_pause_words(show_name, file_name, pause_dict)

# if __name__ == '__main__':
    # show_name = "Political Gabfest"
    # n_segs = 10
    # file_name = 'gabfest'
    # align_show(show_name, n_segs, file_name)
