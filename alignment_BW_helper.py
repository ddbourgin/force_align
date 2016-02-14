import numpy as np
import re
import Levenshtein

def grow_phoneme_dict(word_phonemes, w_dict, offset, file_id, line):
  phonemes = []
  for ph in w_dict['phonemes']:
    ll = [ph[0], (ph[1] * 60.) + offset, (ph[2] * 60.) + offset]
    phonemes.append(ll)

  if w_dict['word'] not in word_phonemes:
    word_phonemes[w_dict['word']] = \
        [{'file': file_id,
          'alignedWord': w_dict['alignedWord'],
          'speaker': line[1],
          'gender': line[3],
          'phonemes': phonemes # timesteps in minutes
        }]

  else:
    word_phonemes[w_dict['word']].append(
         {'file': file_id,
          'alignedWord': w_dict['alignedWord'],
          'speaker': line[1],
          'gender': line[3],
          'phonemes': phonemes # timesteps in minutes
          })
  return word_phonemes


def exact_match(ss, ii, sentence, line_num):
  try:
    a = ss.transcript[line_num[-1] + ii, 2]
  except IndexError:
    return False, ss, None

  cleaned_line, cleaned_sentence = clean_sentence_n_lines(ss, sentence, line_num, ii)

  if re.search(re.escape(cleaned_sentence), cleaned_line):
    trans_idx = line_num[0] + ii
    trans_line = ss.transcript[trans_idx, 2]
    ss.transcript[trans_idx, 2] = sentence.strip()
    ss.event_line_dict[trans_idx] = ['Speech']
    ss = check_for_pauses(ss, sentence, trans_idx, pause_dur=1.0)
    return True, ss, trans_idx

  else:
    return False, ss, None


def partial_match(ss, ii, sentence, line_num):
  try:
    a = ss.transcript[line_num[-1] + ii, 2]
  except IndexError:
    return False, ss, None, None

  cleaned_line, cleaned_sentence = clean_sentence_n_lines(ss, sentence, line_num, ii)
  shortened_sentence = ' '.join(cleaned_sentence.split(' ')[:3])
  bab = len(shortened_sentence)

  if cleaned_sentence == '':
    return False, ss, cleaned_sentence, None

  if shortened_sentence in cleaned_line[:bab+30]:
    print('\tShort Query: '.upper() + shortened_sentence)
    print('\tFull Query: '.upper() + sentence)
    print('\tPotential Match: '.upper() + cleaned_line + '\n')

    trans_idx = line_num[-1] + ii
    trans_line = ss.transcript[trans_idx, 2]
    ss.transcript[trans_idx, 2] = sentence.strip()
    ss.event_line_dict[trans_idx] = ['Speech']
    ss = check_for_pauses(ss, sentence, trans_idx, pause_dur=1.0)
    return True, ss, cleaned_sentence, trans_idx

  else:
    return False, ss, cleaned_sentence, None


def guess_match(ss, ii, sentence, line_num, trans_idx):
    try:
      a = ss.transcript[trans_idx + ii, 2]
    except IndexError:
      return False, ss, None

    cleaned_line, cleaned_sentence = clean_sentence_n_lines(ss, sentence, [trans_idx], ii)
    shortened_sentence = ' '.join(cleaned_sentence.split(' ')[:2])
    bab = len(shortened_sentence)
    testing_idx = trans_idx + ii

    if shortened_sentence == '':
      return False, ss, None

    if shortened_sentence in cleaned_line[:bab+40]:
      print('\tSHORT Query: '.upper() + shortened_sentence)
      print('\tFULL QUERY: '.upper() + cleaned_sentence)
      print('\t??Guess??: '.upper() + ss.transcript[testing_idx, 2] + '\n')

      trans_idx = testing_idx
      trans_line = ss.transcript[trans_idx, 2]
      ss.transcript[trans_idx, 2] = sentence.strip()
      ss.event_line_dict[trans_idx] = ['Speech']
      ss = check_for_pauses(ss, sentence, trans_idx, pause_dur=1.0)
      return True, ss, trans_idx

    else:
      # if sentence == '':
    	return False, ss, None


def hail_mary_guess(trans_idx, ss, mod, sentence, line_num, ii):
  if trans_idx < ss.transcript.shape[0]-1:
    trans_idx = trans_idx + 1

  while ss.transcript[trans_idx, 1] == 'EVENT':
    trans_idx += 1

  for ii in mod:
    matched, ss, idx = guess_match(ss, ii, sentence, line_num, trans_idx)
    if matched == True:
      return True, ss, idx

  _, cleaned_sentence = clean_sentence_n_lines(ss, sentence, line_num, 0)
  qq=np.array([tt.replace(',', '').replace('[BLEEP] ', '') for tt in ss.transcript[:, 2]])
  qq2 = np.array([' '.join(rr.strip().split(' ')[:2]) for rr in qq])

  if np.argwhere(qq == cleaned_sentence.replace(',', '')):
    trans_idx = np.argwhere(qq == cleaned_sentence.replace(',', '')).flatten()[0]
    ss.transcript[trans_idx, 2] = sentence.strip()
    ss.event_line_dict[trans_idx] = ['Speech']
    ss = check_for_pauses(ss, sentence, trans_idx, pause_dur=1.0)

    print('\tQUERY: '.upper() + cleaned_sentence)
    print('\t??Hail Mary Guess??: '.upper() + ss.transcript[trans_idx, 2] + '\n')
    return True, ss, trans_idx

  if len(np.argwhere(qq2 == ' '.join(cleaned_sentence.replace(',', '').split(' ')[:2]))):

    trans_idx = np.argwhere(qq2 == ' '.join(cleaned_sentence.replace(',', '').split(' ')[:2])).flatten()[0]
    ss.transcript[trans_idx, 2] = sentence.strip()
    ss.event_line_dict[trans_idx] = ['Speech']
    ss = check_for_pauses(ss, sentence, trans_idx, pause_dur=1.0)

    print('\tQUERY: '.upper() + cleaned_sentence)
    print('\t??Hail Mary Guess??: '.upper() + ss.transcript[trans_idx, 2] + '\n')
    return True, ss, trans_idx

  else:
    if sentence == u'Wow. {0.33}':
      ss.transcript[151, 2] = sentence.strip()
      ss.event_line_dict[151] = ['Speech']
      ss = check_for_pauses(ss, sentence, 151, pause_dur=1.0)
      return True, ss, 151

    if cleaned_sentence == u"Wow. We're driving fast out of here. OK, we're out of here.":
      ss.transcript[140, 2] = sentence.strip()
      ss.event_line_dict[140] = ['Speech']
      ss = check_for_pauses(ss, sentence, 140, pause_dur=1.0)
      return True, ss, 140

    if sentence == u"Come on! {2.01}":
      ss.transcript[81, 2] = sentence.strip()
      ss.event_line_dict[81] = ['Speech']
      ss = check_for_pauses(ss, sentence, 81, pause_dur=1.0)
      return True, ss, 81 # ? not positive about this one.. (Episode 258)

    if sentence == u'[MUSIC "LIVE AND LET DIE" {0.08} BY GUNS \'N\' ROSES.':
      ss.transcript[316, 2] = sentence.strip()
      ss.event_line_dict[316] = ['Music']
      ss = check_for_pauses(ss, sentence, 316, pause_dur=1.0)
      return True, ss, 316
    # still no match :-(
    return False, ss, None


def levenshtein_find(ss, sentence, cleaned_sentence):
    """
    Uses the Levenshtein distance between a sentence and the lines in a
    transcript to identify the line in transcript most likely to correspond to
    the query sentence

    Parameters
    ----------
    transcript : np.array of shape (n, 4)
       A podcast transcript as produced via compile_episode_transcript. Each
       row corresponds to a line in the transcript, and the columns
       correspond to [start_time, end_time, utterance, speaker_id]

    sentence : string
       A sentence as compiled from the p2fa transcription of a podcast.

    Returns
    -------
    idx : int
      The index to the row in transcript most likely to contain sentence
    """
    edit_dist = []
    for idx, line in enumerate(ss.transcript):
        trans_line = line[2]
        edit_dist.append(Levenshtein.distance(cleaned_sentence, trans_line))
    trans_idx = np.argmin(edit_dist)
    trans_line = ss.transcript[trans_idx, 2]
    ss.transcript[trans_idx, 2] = sentence.strip()
    ss.event_line_dict[trans_idx] = ['Speech']
    ss = check_for_pauses(ss, sentence, trans_idx, pause_dur=1.0)
    return ss, idx


def find_line_in_transcript(ss, mod, line_num, sentence, trans_idx):
  """
  Ugh, what a mess. Maybe should rewrite this to use edit distance to align
  the pause lines with the appropriate line in the transcript?
  """

  for ii in mod: # first pass
    matched, ss, idx = exact_match(ss, ii, sentence, line_num)
    if matched == True:
      return ss, idx

  # second pass
  for ii in mod:
    matched, ss, cleaned_sentence, idx = partial_match(ss, ii, sentence, line_num)
    if matched == True:
      return ss, idx

  if cleaned_sentence == '':
    print('SKIPPING LINE: ' + sentence)
    return ss, None

  # penultimate pass
  matched, ss, idx = hail_mary_guess(trans_idx, ss, mod, sentence, line_num, ii)
  if matched == True:
    return ss, idx

  # final pass
  try:
    return levenshtein_find(ss, sentence, cleaned_sentence)
  except:
    # if, somehow, we still can't find a match
    print('UNABLE TO FIND THE FOLLOWING:')
    print('sentence: {}\ncleaned_sentence: {}\nline_num: {}\nii: {}'.format(sentence, cleaned_sentence, line_num, ii))
    import ipdb; ipdb.set_trace()

def check_for_pauses(ss, sentence, trans_idx, pause_dur=1.0):
  if re.search(r'{.*?}', sentence):
    ppp = re.findall(r'{.*?}', sentence)
    ppp = np.array([float(aa.replace('{', '').replace('}', '')) for aa in ppp])
    if any(ppp >= float(pause_dur)):
      ss.event_line_dict[trans_idx].append('Pause > %ss'%str(pause_dur))
  return ss


def clean_sentence_n_lines(ss, sentence, line_num, ii):
  event_regex = r'\[.*\]'

  cleaned_line = ss.transcript[line_num[-1] + ii, 2]\
      .replace(' :', ':')\
      .replace('].', ']')\
      .replace('--', '')\
      .replace('  ', ' ')\
      .replace(' [UNINTELLIGIBLE]', '')\
      .replace('-',' ')\
      .replace('&quot;', '"')\
      .replace('andquot;', '"')\
      .replace('&', 'and')\
      .replace('[? ', '')\
      .replace(' ?]', '')\
      .replace(' ]',']')\
      .replace('#!MLF!#', '')\
      .replace('", ', '"')\
      .replace('"! ', '"')

  cleaned_line = re.sub(event_regex, "", cleaned_line)
  cleaned_line = cleaned_line.replace('  ', ' ').replace('?] ', '').strip()

  cleaned_sentence = re.sub(r'{.*?}', '', sentence)\
      .replace(' :', ':')\
      .replace('].', ']')\
      .replace('--', '')\
      .replace('  ', ' ')\
      .replace(' [UNINTELLIGIBLE]', '')\
      .replace('-',' ')\
      .replace('&quot;', '"')\
      .replace('andquot;', '"')\
      .replace('&', 'and')\
      .replace('[? ', '')\
      .replace(' ?]', '')\
      .replace(' ]',']')\
      .replace('#!MLF!#', '')\
      .replace('", ', '"')\
      .replace('"! ', '"')

  cleaned_sentence = re.sub(event_regex, "", cleaned_sentence)
  cleaned_sentence = cleaned_sentence.replace('  ', ' ').replace('?] ', '').strip()
  return cleaned_line, cleaned_sentence
