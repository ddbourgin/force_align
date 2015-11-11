#!/usr/bin/env python

from bs4 import BeautifulSoup
import numpy as np
import requests
import re
import json
import cgi
import os
import glob
from pydub import AudioSegment # for slicing mp3s
from align import * 

def load_TAL_tags():
  data1 = np.load('./data/TAL_episode_tags.npz')
  tag_eps = data1['tag_eps']
  tag_links = data1['tag_links']
  ep_tags = data1['ep_tags']
  return tag_eps, tag_links, ep_tags


def make_data_directories():
  if not os.path.lexists('./alignment_data'):
    os.mkdir('./alignment_data')
  if not os.path.lexists('./alignment_data/full_audio'):
      os.mkdir('./alignment_data/full_audio')
  if not os.path.lexists('./alignment_data/seg_audio'):
      os.mkdir('./alignment_data/seg_audio')


def make_json_directory():
  if not os.path.lexists('./alignment_data'):
    os.mkdir('./alignment_data')
  if not os.path.lexists('./alignment_data/json'):
      os.mkdir('./alignment_data/json')


def make_transcript_directory():
  if not os.path.lexists('./alignment_data'):
    os.mkdir('./alignment_data')
  if not os.path.lexists('./alignment_data/transcript'):
      os.mkdir('./alignment_data/transcript')

def make_alignments_directory():
  if not os.path.lexists('./alignment_data'):
    os.mkdir('./alignment_data')
  if not os.path.lexists('./alignment_data/alignments_json'):
      os.mkdir('./alignment_data/alignments_json')

class TAL_scraper(object):
    """
    To use with the bookworm browser, make sure that all scraper classes
    have scrape, write_catalog_and_raw, and write_field_descriptions methods.
    """
    def __init__(self):
        pass

    def scrape(self, session, soup, url, ep_id,
               genderize, names, cached_names, ep_tags):
        self.ep_number = ep_id
        self.session = session
        self.url = url
        self.genderize = genderize
        self.tags = ep_tags
        self.names = names
        self.cached_names = cached_names
        self.soup = soup
        self.show = 'This American Life'

        self.collect_metadata()
        self.construct_transcript()
        return self.cached_names

    def collect_metadata(self):
        air_date = self.soup.find_all("div", class_="radio-date")[0].getText().split('Originally aired ')[1]
        ep_title = self.soup.find_all("div", class_="radio")[0].findAll('a')[0].getText()

        try:
          ep_audio = self.soup.find_all("p", class_="full-audio")[0].getText().split('Full audio: ')[1]
        except IndexError:
          ep_audio = u'Not Available'

        try:
          keywords = self.soup.find_all("meta", {"name":"keywords"})[0]['content'].split(',')
        except IndexError:
          keywords = u'Not Available'

        self.meta = {'airdate':air_date, 'id':self.ep_number, 'title':ep_title,
                     'transcript':self.url, 'audio':ep_audio, 'keywords':keywords}


    def construct_transcript(self):
        # get timesteps for all events in podcast
        full_transcript = self.soup.findAll("p", begin=True)
        timesteps = np.array([pp['begin'] for pp in full_transcript])

        host = self.soup.find_all("div", class_="host")
        subject = self.soup.find_all("div", class_="subject")
        interv = self.soup.find_all("div", class_="interviewer")

        # get timestamps for each text segment, organized by speaker class
        host_ts = np.array([ee.findAll("p", begin=True)[0]['begin'] for ee in host], dtype=str)
        sub_ts = np.array([ee.findAll("p", begin=True)[0]['begin'] for ee in subject], dtype=str)
        inter_ts = np.array([ee.findAll("p", begin=True)[0]['begin'] for ee in interv], dtype=str)

        # get speaker names for each text segment, organized by speaker class
        host_id  = [ss.findAll('h4') for ss in host]
        sub_id   = [ss.findAll('h4') for ss in subject]
        inter_id = [ss.findAll('h4') for ss in interv]

        # fill in any gaps in host, subject, and interviewer ids
        for ii in range(len(host_id)):
            if host_id[ii] == []:
                host_id[ii] = host_id[ii-1]
        if host_id[0] == []: # in case both 1st and last host ids are blank
            host_id[0] = host_id[1]

        for ii in range(len(sub_id)):
            if sub_id[ii] == []:
                sub_id[ii] = sub_id[ii-1]

        for ii in range(len(inter_id)):
            if inter_id[ii] == []:
                inter_id[ii] = inter_id[ii-1]

        # record final speaker list for hosts, subjects, and interviewers
        host_id  = [tt[0].getText() for tt in host_id]
        sub_id   = [tt[0].getText() for tt in sub_id]
        inter_id = [tt[0].getText() for tt in inter_id]

        #get text for each segment, organized by speaker class
        host_txt = [ss.findAll("p")[0].getText() for ss in host]
        sub_txt = [ss.findAll("p")[0].getText() for ss in subject]
        inter_txt = [ss.findAll("p")[0].getText() for ss in interv]

        # capture any additional audio events
        event_regex = r'(^\[.*\]$)'
        text = [pp.getText() for pp in full_transcript]
        events = np.array([re.compile(event_regex).search(b) for b in text])
        events_idx = np.argwhere(events).flatten()

        event_txt = [ee.group() for ee in events[events_idx]]
        event_ts = timesteps[events_idx]
        event_id = ['EVENT'] * len(events_idx)
        event_gender = ['unknown'] * len(events_idx)

        # if there are still unprocessed timestamps, assign them to the most recent label
        label = 'HOST' # assume the first utterance comes from the host
        sid = host_id[0]
        if (len(event_ts) + len(host_ts) + len(sub_ts) + len(inter_ts)) != len(timesteps):
            for idx, ts in enumerate(timesteps):
                if len(np.argwhere(host_ts == ts)):
                    label = 'HOST'
                    sid = host_id[np.argwhere(host_ts == ts).flatten()]
                    continue
                elif len(np.argwhere(sub_ts == ts)):
                    label = 'SUB'
                    sid = sub_id[np.argwhere(sub_ts == ts).flatten()]
                    continue
                elif len(np.argwhere(inter_ts == ts)):
                    label = 'INTER'
                    sid = inter_id[np.argwhere(inter_ts == ts).flatten()]
                    continue
                elif len(np.argwhere(event_ts == ts)):
                    continue
                else:
                    if label == 'HOST':
                        host_txt.append(full_transcript[idx].getText())
                        host_id.append(sid)
                        host_ts = np.hstack([host_ts, ts])
                    elif label == 'SUB':
                        sub_txt.append(full_transcript[idx].getText())
                        sub_id.append(sid)
                        sub_ts = np.hstack([sub_ts, ts])
                    elif label == 'INTER':
                        inter_txt.append(full_transcript[idx].getText())
                        inter_id.append(sid)
                        inter_ts = np.hstack([inter_ts, ts])

        print('    ts sum: %s, len(timesteps): %s'%(str(len(event_ts) + len(host_ts) + len(sub_ts) + len(inter_ts)), str(len(timesteps))))

        host_gender, sub_gender, inter_gender = \
            self.guess_gender(host_id, sub_id, inter_id)

        # combine timesteps, ids, text, host gender, and host role for each utterance/event
        host = np.asarray(zip(host_ts, host_id, host_txt, host_gender,['Host'] * len(host_ts)))
        subject = np.asarray(zip(sub_ts, sub_id, sub_txt, sub_gender, ['Subject'] * len(sub_ts)))
        interv = np.asarray(zip(inter_ts, inter_id, inter_txt, inter_gender, ['Interviewer'] * len(inter_ts)))
        event = np.asarray(zip(event_ts, event_id, event_txt, event_gender, ['Event'] * len(event_ts)))

        self.combine_segments(host, subject, interv, event, timesteps)


    def guess_gender(self, host_id, sub_id, inter_id):
        # get first names for each speaker
        host_first_names, sub_first_names, inter_first_names = [], [], []
        for name in host_id:
            host_first_names.append(name.split(' ')[0])
        for name in sub_id:
            sub_first_names.append(name.split(' ')[0])
        for name in inter_id:
            inter_first_names.append(name.split(' ')[0])

        host_first_names = np.asarray(host_first_names)
        sub_first_names = np.asarray(sub_first_names)
        inter_first_names = np.asarray(inter_first_names)

        # guess gender for each speaker
        host_gender  = np.array(['unknown'] * len(host_first_names))
        sub_gender   = np.array(['unknown'] * len(sub_first_names))
        inter_gender = np.array(['unknown'] * len(inter_first_names))

        # would be better to combine host,sub, and inter names into a single list to make this function generalizable...
        if len(host_first_names):
            host_unique = np.unique(host_first_names)
            for name in host_unique:
                idxs = np.argwhere(host_first_names == name).flatten()
                if name in self.names.keys():
                    host_gender[idxs] = self.names[name]
                elif name in self.cached_names.keys():
                    host_gender[idxs] = self.cached_names[name]
                else:
              #      try:
              #          gender = self.genderize.get([name])[0][u'gender']
              #          self.cached_names[name] = gender
              #          host_gender[idxs] = gender

              #      except requests.ConnectionError:
		    print('Error getting gender for name %s'%name)
                    gender = 'unknown'
                    host_gender[idxs] = gender

                    if gender == None:
                        host_gender[idxs] = 'unknown'
                        self.cached_names[name] = 'unknown'

        if len(sub_first_names):
            sub_unique = np.unique(sub_first_names)
            for name in sub_unique:
                idxs = np.argwhere(sub_first_names == name).flatten()
                if name in self.names.keys():
                    sub_gender[idxs] = self.names[name]
                elif name in self.cached_names.keys():
                    sub_gender[idxs] = self.cached_names[name]
                else:
              #      try:
              #          gender = self.genderize.get([name])[0][u'gender']
              #          self.cached_names[name] = gender
              #          sub_gender[idxs] = gender

              #      except requests.ConnectionError:
                     print('Error getting gender for name %s'%name)
                     gender = 'unknown'
                     sub_gender[idxs] = gender

                     if gender == None:
                        sub_gender[idxs] = 'unknown'
                        self.cached_names[name] = 'unknown'

        if len(inter_first_names):
            inter_unique = np.unique(inter_first_names)
            for name in inter_unique:
                idxs = np.argwhere(inter_first_names == name).flatten()
                if name in self.names.keys():
                    inter_gender[idxs] = self.names[name]
                elif name in self.cached_names.keys():
                    inter_gender[idxs] = cached_genders[name]
                else:
#                    try:
#                        gender = self.genderize.get([name])[0][u'gender']
#                        self.cached_names[name] = gender
#                        inter_gender[idxs] = gender

#                    except requests.ConnectionError:
                     print('Error getting gender for name %s'%name)
                     gender = 'unknown'
                     inter_gender[idxs] = gender

                     if gender == None:
                        inter_gender[idxs] = 'unknown'
                        self.cached_names[name] = 'unknown'

        return host_gender, sub_gender, inter_gender


    def combine_segments(self, host, subject, interv, event, timesteps):
        """
        combine segments into a single transcript with columns
        [timestep, speaker, text, gender, speaker_class]
        """
        transcript = []
        for idx, ts in enumerate(timesteps):
            if len(host):
                if len(np.argwhere(host[:,0] == ts)):
                    transcript.append(host[np.argwhere(host[:,0] == ts).flatten(), :].flatten())
                    continue
            if len(subject):
                if len(np.argwhere(subject[:,0] == ts)):
                    transcript.append(subject[np.argwhere(subject[:,0] == ts).flatten(), :].flatten())
                    continue
            if len(interv):
                if len(np.argwhere(interv[:,0] == ts)):
                    transcript.append(interv[np.argwhere(interv[:,0] == ts).flatten(), :].flatten())
                    continue
            if len(event):
                if len(np.argwhere(event[:,0] == ts)):
                    transcript.append(event[np.argwhere(event[:, 0] == ts).flatten(), :].flatten())
                    continue
            if len(transcript):
                iddx = int(idx)-1
                while transcript[iddx][1] == u"EVENT":
                    iddx = iddx - 1
                transcript.append([ts, transcript[iddx][1], text[idx], transcript[iddx][3], transcript[iddx][4]])
        self.transcript = np.asarray(transcript)


    def write_catalog_and_raw(self, counter):
        """
        For Bookworms. Save each utterance to a separate text file and
        generate the jsoncatalog references accordingly.
        """
        ad = self.meta['airdate'].split('.')
        ad = str(ad[2] + '-' + ad[0] + '-' + ad[1]) # YYYY-MM-DD
        self.catalog = []

        for line in self.transcript:
            counter = counter + 1

            ts = line[0].split(':')
            ts = (int(ts[0]) * 60.) + int(ts[1]) + (float(ts[2])/60.)
            ts = int(ts) # recast minutes to int for bookworm parsing

            txt = cgi.escape(unicode(str(line[2].encode('ascii', 'ignore')), 'utf-8'), quote=True)
            link = ' [<a href="' + self.meta['transcript'] + '">ep. ' + str(self.ep_number) + '</a>]'
            sstring = txt + unicode(link, 'utf-8')

            catalog = {"searchstring":sstring,
                       "filename":unicode(str(counter), 'utf-8'),
                       "year":unicode(ad, 'utf-8'),
                       "transcript":unicode(self.meta['transcript'], 'utf-8'),
                       "ep_number":int(self.ep_number),
                       "speaker":unicode(str(line[1]), 'utf-8'),
                       "gender":unicode(str(line[3]), 'utf-8'),
                       "speaker_type":unicode(str(line[4]), 'utf-8'),
                       "minutes":ts}

            catalog = json.dumps(catalog)

            with open('./texts/raw/'+str(counter)+'.txt', 'wb') as f:
                f.write(unicode(str(line[2].encode('ascii', 'ignore')), 'utf-8'))
                f.close()

            with open('./metadata/jsoncatalog.txt', 'a') as f:
                f.write(str(catalog)+'\n')
                f.close()
        return counter


    def write_field_descriptions(self):
        field_descriptions = \
        [
            {"field":"speaker", "datatype":"categorical", "type":"text",
             "unique":False},
            {"field":"gender", "datatype":"categorical", "type":"text",
             "unique":False},
            {"field":"speaker_type", "datatype":"categorical", "type":"text",
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


    def write_transcript(self, startrow=0, endrow=None, sliceid='_full',
                         JSON=False):
        """
        Save the transcript to a single text file for use with Steve's speech
        editor.
        """
        event_regex = r'(\s?\[.*\])'
        if startrow == 0 and endrow is None:
            trans = self.transcript
        else:
            trans = self.transcript[startrow:endrow, :]
        if JSON:
            tscrpit = []
            make_json_directory()
            file_name = 'TAL' + str(self.ep_number) + sliceid + '.json'
            file_name = './alignment_data/json/' + file_name
            for line in trans:
                if line[1] == 'EVENT': continue
                speaker = str(line[1].split(' ')[-1]).upper()
                utter = line[2].replace(' :', ':').replace('].', ']').replace('--', '').replace('  ', ' ').replace(' [UNINTELLIGIBLE]', '').replace('-',' ').replace('&', 'and').replace('[? ', '').replace(' ?]', '').replace(' ]',']').replace('#!MLF!#', '')
                utter = re.sub(event_regex, "", utter)
                utter = utter.replace('  ', ' ').replace('?] ', '')
		if utter == ' ' or utter == '':
		    continue
                catalog = {'speaker': speaker, 'line': utter}
                tscrpit.append(catalog)
            with open(file_name, 'wb') as f:
                json.dump(tscrpit, f)
                f.close()
        else:
            make_transcript_directory()
            file_name = 'TAL' + str(self.ep_number) + sliceid + '.txt'
            file_name = './alignment_data/transcript/' + file_name
            with open(file_name, 'wb') as tf:
                for line in trans:
                    speaker = str(line[1].split(' ')[-1]).upper()
                    utter = line[2].replace(' :', ':').replace('].', ']').replace('--', '').replace('  ', ' ')
                    tf.write(speaker + ': ' + utter +'\n\n')
                tf.close()


    def chunk_audio_and_transcript(self, n_chunks=8):
        '''
        Divide episode audio into n_chunks segments of ~ equal length,
        segment  transcript accordingly
        '''
        timesteps = self.transcript[:,0]
        timesteps = \
            np.array([(float(ts.split(':')[0]) * 60.) + float(ts.split(':')[1])
             + (float(ts.split(':')[2])/60.) for ts in timesteps]) # in minutes
        file_name='./alignment_data/full_audio/TAL'+str(self.ep_number)+'.mp3'
        ep_audio = AudioSegment.from_file(file_name, format='mp3',
                                          channels=1, sample_width=2)
        fs = float(ep_audio.frame_rate)

        ep_length = (ep_audio.frame_count() / fs)/60. # in minutes
        division  = ep_length/float(n_chunks)
        times, time_idxs, slices = [0.], [0], []
        silence = AudioSegment.silent(duration=500)
        # add 0.5s at the start/end of an audio slice

        for ii in np.arange(1,9):
            times.append(min(timesteps, key=lambda x:abs(x - (ii * division))))
            time_idxs.append(np.argwhere(timesteps == times[-1]).flatten()[0])
            start = times[ii-1]*60.*1000.
            end = times[ii]*60.*1000.
            print('    Writing audio slice ' + str(ii) + ': Minutes ' + str(times[ii-1]) + ' to ' + str(times[ii]))
            aud_slice = silence + ep_audio[start:end] + silence
            sliceid = '_seg'+str(ii)
            self.write_transcript(startrow=time_idxs[ii-1],
                                  endrow=time_idxs[ii],
                                  sliceid=sliceid,
                                  JSON=True)
            file_name = 'TAL'+str(self.ep_number)+sliceid+'.wav'
            file_name = './alignment_data/seg_audio/'+file_name
            aud_slice.export(file_name, 'wav')


    def align_transcript(self, n_chunks=8):
        self.dl_audio()
        self.chunk_audio_and_transcript(n_chunks)
        make_alignments_directory()
        wavfile = './alignment_data/seg_audio/TAL'+str(self.ep_number)+'_seg*.wav'
        for ff in glob.glob(wavfile):
            file_name = os.path.split(ff)[-1].split('.')[0]
            print('    Aligning file %s'%file_name)
            trsfile = './alignment_data/json/'+file_name+'.json'
            outfile = './alignment_data/alignments_json/' + file_name + '_aligned.json'
            do_alignment(ff, trsfile, outfile, json=True, textgrid=False,
                 phonemes=True, breaths=False)


    def dl_audio(self):
        """
        Download the audio file for the episode for use with Steve's speech
        editor
        """
        make_data_directories()
        link = 'http://audio.thisamericanlife.org/jomamashouse/ismymamashouse/' + str(self.meta['id']) + '.mp3'
        print('    Downloading audio for TAL episode %s'%(str(self.meta['id'])))
        audio = self.session.get(link, timeout=50.0)
        filename='./alignment_data/full_audio/TAL'+str(self.ep_number)+'.mp3'
        with open(filename, 'wb') as f:
            f.write(audio.content)
            f.close()



