import os
import numpy as np
import csv
import dotenv
import psycopg2
import pandas as pd
import re
from audiosearch import Client
import datetime
import MySQLdb

class DB_Connection(object):
    def __init__(self, database):
        self.dbname = database
        host, user, paswd, ssh_user, ssh_key = self.get_env()
        self.ssh_tunnel(ssh_user, host, ssh_key)
        self.db = MySQLdb.connect(host='127.0.0.1', user=user, port=9990,
                                  passwd=paswd, db=database)
        self.cursor = self.db.cursor()

    def get_env(self):
        import dotenv
        dotenv.load_dotenv('.env')
        host = os.environ.get("host")
        user = os.environ.get("user")
        paswd = os.environ.get("pwd")
        ssh_user = os.environ.get("ssh_user")
        ssh_key = os.environ.get("ssh_key")
        return host, user, paswd, ssh_user, ssh_key

    def ssh_tunnel(self, ssh_user, host, ssh_key):
        """
        Don't forget to close this tunnel when you're finished
        grabbing stuff from the server!
        """
        import subprocess
        cmd = ['ssh', '{}@{}'.format(ssh_user, host), '-i',
               '{}'.format(ssh_key), '-f','-N', '-L', '9990:localhost:3306']
        subprocess.check_call(cmd)

    def kill_tunnel():
        call = subprocess.check_call(['killall', 'ssh'])




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
      .replace('"! ', '"')\
      .replace(' ?', '?')\
      .replace(' ,', ',')\
      .replace(' .', '.')\
      .replace('#', '')\
      .replace('^', '')\
      .replace('*', '')
    cleaned_sentence = re.sub(event_regex, "", cleaned_sentence)
    cleaned_sentence = cleaned_sentence\
                            .replace('  ', ' ')\
                            .replace('?] ', '')\
                            .strip()
    return cleaned_sentence


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
