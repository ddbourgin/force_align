force align
============
Fork of [p2fa-vislab](https://github.com/ucbvislab/p2fa-vislab), itself a fork of [p2fa](http://www.ling.upenn.edu/phonetics/p2fa/). This can be used to expose phoneme-level parses of a speech audio file and align it to its verbatim text transcript.

# Notes
The two relevant branches of this project are `master` and `server.` The `master` branch contains files for pulling transcripts from the [Audiosearch](https://www.audiosear.ch/) API and database, while the `server` branch has files for scraping transcripts from the This American Life [radio archive](http://www.thisamericanlife.org/radio-archives).
- Unfortunately, in order to use the files in `master`, you'll need access to a local version of the Audiosearch database dump.
- The files in the `server` branch are much more disorganized. I'm working to clean them up, but for now, it's probably worth just skimming to see if they make sense.

The general workflow for the scripts in both branches is as follows:
    1. Identify the episodes we wish to transcribe (for This American Life this is easy -- all we need are the episode IDs, but for the Audiosearch database this is a bit more involved - we have to query the API to get the appropriate episode and transcript IDs, etc.)
    2. For each episode, scrape the transcript from the appropriate location. For This American Life we can use BeautifulSoup to grab it from the radio archive, while for Audiosearch podcasts we grab the machine transcription from the Audiosearch database.
    3. Once we have an episode's transcript, we download the episode audio.
    4. Next, we chunk the episode transcript and audio into segments and feed each transcript + audio segment pair to [p2fa-vislab](https://github.com/ucbvislab/p2fa-vislab), which returns a json file with the phoneme-level alignment for each transcript chunk (see the p2fa-vislab readme for more information).
    5. Now that we have both the word and phoneme-level transcript for an episode, we can construct a [Bookworm](https://bookworm-project.github.io/Docs/) database and browser for visualizing n-gram counts. I have also included extra plotting functions in `analysis.py` and `analysis_db.py` in the `master` branch for running local pairwise comparisons for, e.g., different pronunciations of particular words by speaker, preceding and succeeding pauses associated with particular words, ranking words by the number of different pronunciations, etc.

Unfortunately, it has been annoying to get [HTK](http://htk.eng.cam.ac.uk/) (a dependency of p2fa and p2fa-vislab) to work nicely on OSX out of the box. For the time being I have been doing the forced alignments in a VM. The Vagrantfile for this is included in the repo, although if you're just exploring for the first time it may not be worth building the entire VM just yet.
