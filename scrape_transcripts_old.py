import numpy as np
import requests
import time
import os
import shutil
# import dotenv
from bs4 import BeautifulSoup
# from genderize import Genderize
from TAL_scraper import TAL_scraper, load_TAL_tags


def make_bw_directories(file_name):
  if not os.path.lexists('./metadata'):
    os.mkdir('./metadata')
  if not os.path.lexists('./texts'):
      os.mkdir('./texts')
  if not os.path.lexists('./texts/raw'):
      os.mkdir('./texts/raw')
  if not os.path.lexists('./'+file_name):
      os.mkdir('./'+file_name)


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
#  site = session.get(url, timeout=(50.0, 50.0))
  site = session.get(url, timeout=50.0)
  return site, url


def session_error(site, idd):
  if site.status_code == 404:
    print('    ERROR 404 - episode %s not found'%(str(idd)))
  elif site.status_code == 503: # Temporary Unavailability
    print('    ERROR 503 - episode %s is currently unavailable'%(str(idd)))
  else:
    print('    UNKNOWN ERROR - episode %s not transcribed'%(str(idd)))


def scrape_transcripts(episode_ids, trans_url, stagger, scraper, bookworm):
  names = load_name_data()
  tag_eps, tag_links, ep_tags = load_TAL_tags()
  session, genderize = init_sessions()
  cached_names = {}
  counter = 0

  for idd in episode_ids:
    print('Downloading episode ' + str(idd))
    site, url = connect(session, idd, trans_url, stagger)

    if site.status_code != 200:
      session_error(site, idd)
      continue

    soup = BeautifulSoup(site.text)
    cached = scraper.scrape(session, soup, url, idd,
                            genderize, names, cached_names, ep_tags)
    cached_names.update(cached)

    if bookworm:
      counter = scraper.write_catalog_and_raw(counter)
    else:
      # scraper.write_transcript()
      # scraper.dl_audio()
      # scraper.chunk_audio_and_transcript(n_chunks=8)
      scraper.align_transcript(n_chunks=8)


  if bookworm:
    scraper.write_field_descriptions()



if __name__ == '__main__':
  FILE_ID = 'TAL_Improved2'
  EPISODE_IDS = range(57,566)
  STAGGER = 3.1 # stagger requests to reduce server load
  TRANSCRIPT_URL = 'http://www.thisamericanlife.org/radio-archives/episode/####/transcript'
  SCRAPER = TAL_scraper()
  BOOKWORM = False # True for Bookworm formatting,
                   # False for speech alignment formatting

  if BOOKWORM:
    make_bw_directories(FILE_ID)

  scrape_transcripts(EPISODE_IDS, TRANSCRIPT_URL, STAGGER, SCRAPER, BOOKWORM)

  if BOOKWORM:
    shutil.move('./metadata', './' + FILE_ID)
    shutil.move('./texts', './' + FILE_ID)

