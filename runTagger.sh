#!/bin/bash
set -eu

# Run the tagger (and tokenizer).
CURPATH=$(dirname $0)
java -Xbootclasspath/a:$CURPATH/ark-tweet-nlp/py4j-0.7.jar:$CURPATH/ark-tweet-nlp/weka.jar -XX:ParallelGCThreads=2 -Xmx500m -jar $CURPATH/ark-tweet-nlp-0.3.2.jar "$@" &
echo $! > /tmp/sctagger-$1.pid
