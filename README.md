force align
============
Fork of [p2fa-vislab](https://github.com/ucbvislab/p2fa-vislab), itself a fork of [p2fa](http://www.ling.upenn.edu/phonetics/p2fa/). This can be used to expose phoneme-level parses of a speech audio file and align it to its verbatim text transcript.

## Notes
The two relevant branches of this project are `master` and `server.` The `master` branch contains files for pulling transcripts from the [Audiosearch](https://www.audiosear.ch/) API and database, while the `server` branch has files for scraping transcripts from the This American Life [radio archive](http://www.thisamericanlife.org/radio-archives).
- Unfortunately, in order actually run the code in `master`, you'll need access to a local version of the Audiosearch database dump. Because this is proprietary, I can't include it here :-(
- The files in the `server` branch are much more disorganized. I'm working to clean them up, but for now, it's probably worth just skimming to see if they make sense.

## Transcription and Alignment
The general transcription workflow is as follows:

1. Identify the episodes we wish to transcribe (for This American Life this is easy -- all we need are the episode IDs. For episodes in the Audiosearch database this is a bit more involved - we have to query the API to get the appropriate episode and transcript IDs, etc.)

2. For each episode, scrape the transcript and episode audio from the appropriate locations. For This American Life we use [BeautifulSoup](http://www.crummy.com/software/BeautifulSoup/) to grab the transcript from the radio archive, while for Audiosearch podcasts we can grab the machine transcription from the Audiosearch database.

3. Next, we chunk the episode transcript and audio into segments and feed each transcript + audio segment pair into [p2fa-vislab](https://github.com/ucbvislab/p2fa-vislab), which returns a JSON file with the phoneme-level alignment for each transcript chunk (see the p2fa-vislab readme for more information).

4. Repeat steps 2 & 3 for each episode associated with a particular podcast.

5. Now that we have both the word and phoneme-level transcripts for a podcast, we can construct a [Bookworm](https://bookworm-project.github.io/Docs/) database and browser for visualizing the n-gram counts associated with both word-level and phoneme-level data. In the repo there are also additional plotting functions in `analysis.py` and `analysis_db.py` in the `master` branch for plotting pairwise comparisons for, e.g., different pronunciations of particular words by speaker, preceding and succeeding pauses associated with particular words, ranking words by the number of different pronunciations, etc.

Unfortunately, it has been annoying to get [HTK](http://htk.eng.cam.ac.uk/) (a dependency of p2fa and p2fa-vislab) to work nicely on OSX out of the box. As a work around I have been performing all forced alignments in a VM. The Vagrantfile for this is included in the repo, although if you're just exploring for the first time it may not be worth building it just yet. Eventually I should move this over to a Docker container, which will make everything much more lightweight.

The file `align_audiosearch_transcripts.py` organizes the majority of the transcript scraping/alignment/bookworm construction.
