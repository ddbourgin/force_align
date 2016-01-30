# force align - server
Fork of [p2fa-vislab](https://github.com/ucbvislab/p2fa-vislab), itself a fork of [p2fa](http://www.ling.upenn.edu/phonetics/p2fa/). The code in this branch is run from an AWS EC2 instance dedicated to compiling episode transcripts of This American Life from the [radio archives](http://www.thisamericanlife.org/radio-archives), aligning them to the episode audio using p2fa-vislab, and organizing them into a format for creating n-gram visualizations using [Bookworm](https://bookworm-project.github.io/Docs/).

`scrape_transcripts.py` is the file which organizes the majority of the transcription/alignment/bookworm construction.
