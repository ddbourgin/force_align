import numpy as np
import requests
import time
import os
import shutil
import json
import glob
import re
import pandas as pd
from bs4 import BeautifulSoup
# import dotenv
# from genderize import Genderize
from TAL_scraper import TAL_scraper, load_TAL_tags
from alignment_BW_helper import grow_phoneme_dict, find_line_in_transcript


def make_bw_directories(file_name, aligned):
  if not os.path.lexists('./metadata'):
    os.mkdir('./metadata')
  if not os.path.lexists('./texts'):
      os.mkdir('./texts')
  if not os.path.lexists('./'+file_name):
      os.mkdir('./'+file_name)
  if not os.path.lexists('./'+file_name+'_phonemes') and aligned:
      os.mkdir('./'+file_name+'_phonemes')
  if not os.path.lexists('./'+file_name+'_phonemes/metadata') and aligned:
      os.mkdir('./'+file_name+'_phonemes/metadata')
  if not os.path.lexists('./'+file_name+'_phonemes/texts') and aligned:
      os.mkdir('./'+file_name+'_phonemes/texts')


def load_name_data():
  data2 = np.load('./data/name_genders.npz')
  unknown = data2['unknown']
  male = data2['male']
  female = data2['female']

  names = dict(male)
  names.update(female)
  names.update(unknown)
  return names


def init_sessions():
  # dotenv.load_dotenv('.env')
  # key = str(os.environ.get(u'genderize_user'))
  # secret = str(os.environ.get(u'genderize_key'))

  session = requests.Session()
  session.mount('http://', requests.adapters.HTTPAdapter(max_retries=50))
  # genderize = Genderize(user_agent=key, api_key=secret)
  genderize = None
  return session, genderize


def connect(session, idd, trans_url, stagger):
  url = trans_url.replace('####', str(idd))
  time.sleep(stagger)
  site = session.get(url, timeout=50.0)
  return site, url


def session_error(site, idd):
  if site.status_code == 404:
    print('    ERROR 404 - episode %s not found'%(str(idd)))
  elif site.status_code == 503:
    print('    ERROR 503 - episode %s is currently unavailable'%(str(idd)))
  else:
    print('    UNKNOWN ERROR - episode %s not transcribed'%(str(idd)))


def scrape_transcripts(episode_ids, trans_url, stagger, scraper, bookworm, align):
  if type(episode_ids) != list:
    episode_ids = [episode_ids]

  names = load_name_data()
  tag_eps, tag_links, ep_tags = load_TAL_tags()
  session, genderize = init_sessions()
  cached_names, scrapers = {}, []
  counter = 0

  for idd in episode_ids:
    print('Downloading episode ' + str(idd))
    site, url = connect(session, idd, trans_url, stagger)

    if site.status_code != 200:
      session_error(site, idd)
      continue

    soup = BeautifulSoup(site.text)
    cached = scraper.scrape(session, soup, url, idd,
                            genderize, names, cached_names, ep_tags[idd])
    cached_names.update(cached)
    scrapers.append(scraper)

    if bookworm:
      counter = scraper.write_catalog_and_raw(counter)

    if align:
      scraper.align_transcript(n_chunks=8)

  if bookworm:
    scraper.write_field_descriptions()

  return scrapers


def transcript_json_dump(scrapers):
    """
    Construct a single json file containing the transcripts for multiple TAL episodes.
    transs is a list of transcript arrays, and metas is a list of metadata dictionaries.
    Output json file is of the format
    {
        searchstring : [string; the transcribed utterance]
        year : [string; the original airdate in 'YYYY-MM-DD' format],
        link : [string; url pointing to the episode transcript],
        ep_number : [int; the episode number],
        speaker : [string; the speaker's name],
        gender : [string; our best guess for the speaker's gender],
        speaker_type : [string; the speaker's role in the show],
        minutes : [float; the timestamp of the utterance in minutes]
    }
    """
    transs = [ss.transcript for ss in scrapers]
    metas  = [ss.meta for ss in scrapers]
    transfile = []
    for ii in xrange(len(transs)):
        trans = transs[ii]
        meta = metas[ii]
        for line in trans:
            dd = {'searchstring': line[2],
                  'year': meta['airdate'],
                  'link': meta['transcript'],
                  'ep_number': meta['id'],
                  'speaker': line[1],
                  'gender': line[3],
                  'speaker_type': line[4],
                  'minutes': line[0]}
            transfile.append(dd)

    with open('texts/TAL561-566.json', 'w') as file_id:
        json.dump(transfile, file_id)


def read_aligned_transcripts(episode_ids, trans_url, stagger, scraper, bookworm, align):
  if type(episode_ids) != list:
    episode_ids = [episode_ids]

  word_phonemes = {}
  counter = 0

  for ee in episode_ids:
    transcript = []
    ss = scrape_transcripts(ee, trans_url, stagger, scraper, False, False)[0]
    ss.transcript_phonemes = {}
    ss.write_timestamps_csv(n_chunks=8) # for now - this is a hack
    json_dir = './alignment_data/alignments_json/TAL'+str(ee)+'_*.json'
    line_offset = [0]
    mod = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7]
    trans_idx = 0

    for trans in np.sort(glob.glob(json_dir)):
      file_id = os.path.split(trans)[-1].split('.')[0]
      chunk_num = int(file_id.split('_')[1].split('seg')[-1])
      print('\nReading %s...'%file_id)

      with open(trans) as data_file:
        aligned_json = json.load(data_file)
        word_list = aligned_json["words"]

        offset = ss.chunk_durs[chunk_num - 1][0] # in minutes
        line_dicts, line_num = [], []

        for w_dict in word_list:

          # if word token NOT a pause...
          if 'line_idx' in w_dict.keys():
            line_num.append(w_dict['line_idx'] + np.sum(line_offset))

          # if word token IS a pause
          else:
            start = (w_dict['start'] / 60.) + offset
            end = (w_dict['end'] / 60.) + offset
            w_dict['word'] = '{' + '{0:.2f}'.format((end - start) * 60.) + '}'

          # if new token comes from the same line as the previous tokens
          if len(set(line_num)) < 2:
            line_dicts.append(w_dict)
            continue

          # if we have reached the end of a line, construct the aligned
          # sentence and match it to our transcript
          else:
            # import ipdb; ipdb.set_trace()
            sentence = ' '.join([ll['word'] for ll in line_dicts])
            next_line = line_num.pop()

            # this step matches the line in ss.transcript to our new
            # sentence and replaces it (hence why we return a
            # new version of ss)
            ss, idx = find_line_in_transcript(ss, mod, line_num, sentence, trans_idx)

            if idx == None:
              print('Unable to Match Line. Skipping.'.upper())
              print('\tLine: '.upper() + sentence)
              continue

            else:
              trans_idx = idx
              sentence_phones = []

              for ll in line_dicts:
                if 'phonemes' in ll.keys():
                  sentence_phones = sentence_phones + [ph[0] for ph in ll['phonemes']]

                  # signify the end words using {sp} token
                  sentence_phones = sentence_phones + ['{sp}']
                  word_phonemes = grow_phoneme_dict(word_phonemes, w_dict, offset, file_id, ss.transcript[trans_idx, :])

                elif 'alignedWord' in ll.keys() and ll['alignedWord'] == 'sp':
                  # remove redundant {sp} token if we are inserting a
                  # pause immediately after it
                  if sentence_phones[-1] == '{sp}':
                    sentence_phones.pop()
                  sentence_phones = sentence_phones + ll['word']

              # import ipdb; ipdb.set_trace()
              sentence_phones = ' '.join(sentence_phones)
              ss.transcript_phonemes[idx] = sentence_phones

            # reset line_dicts and line_num for the next line in transcript
            line_dicts, line_num = [w_dict], [next_line]

        line_offset.append( len(set([ll['line_idx'] for ll in word_list if 'line_idx' in ll.keys()])) )

    counter = ss.write_catalog_and_input_align(counter)
    print(counter)
  return word_phonemes

if __name__ == '__main__':
  FILE_ID = 'TAL_Pauses'
  EPISODE_IDS = range(144,566)
  STAGGER = 3.1 # stagger requests to reduce server load
  TRANSCRIPT_URL = 'http://www.thisamericanlife.org/radio-archives/episode/####/transcript'
  SCRAPER = TAL_scraper(FILE_ID)
  BOOKWORM = False # True for regular (non-aligned) Bookworm formatting
  ALIGN = True   # True for performing speech alignment with p2fa-vislab
  ALIGNMENT_BW = False # True for reading the aligned json into bw format

  if ALIGNMENT_BW:
    # problem episodes - TODO: read this from problem_episodes.txt instead of hard coding
    EPISODE_IDS.remove(35)
    EPISODE_IDS.remove(51)
    EPISODE_IDS.remove(62)
    EPISODE_IDS.remove(105)
#    EPISODE_IDS.remove(144)

  if BOOKWORM or ALIGNMENT_BW:
    make_bw_directories(FILE_ID, ALIGNMENT_BW)

  if BOOKWORM or ALIGN:
    scrapers = scrape_transcripts(EPISODE_IDS, TRANSCRIPT_URL, STAGGER, SCRAPER, BOOKWORM, ALIGN)
  elif ALIGNMENT_BW:
    word_phonemes = read_aligned_transcripts(EPISODE_IDS, TRANSCRIPT_URL, STAGGER, SCRAPER, False, False)
    SCRAPER.write_field_descriptions(phonemes=True)

  if BOOKWORM:
    shutil.move('./metadata', './' + FILE_ID)
    shutil.move('./texts', './' + FILE_ID)
