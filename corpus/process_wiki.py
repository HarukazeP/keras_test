#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import logging
import os.path
import sys
from gensim.corpora import WikiCorpus
from distutils.util import strtobool

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    if len(sys.argv) < 4:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    lemmatizeConfigStr, inp, outp = sys.argv[1:4]
    lemmatizeConfig = bool(strtobool(lemmatizeConfigStr))
    space = " "
    i = 0
 
    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=lemmatizeConfig, dictionary={})
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")
 
    output.close()
    logger.info("Finished Saved " + str(i) + " articles")