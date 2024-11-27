#!/bin/bash

# This script is used to generate the list of files as protocol
# the protocol have format: <path> <subdir> <label>

awk '{print "bonafide/" $1 " train bonafide"}' scp/train_bonafide.lst > protocol.txt
awk '{print "spoof/" $1 " train spoof"}' scp/spoof_train.lst >> protocol.txt

# awk '{print "vocoded/hifigan_" $1 " train spoof"}' scp/train_bonafide.lst >> protocol.txt
# awk '{print "vocoded/waveglow_" $1 " train spoof"}' scp/train_bonafide.lst >> protocol.txt
# awk '{print "vocoded/hn-sinc-nsf-hifi_" $1 " train spoof"}' scp/train_bonafide.lst >> protocol.txt

awk '{print "bonafide/" $1 " dev bonafide"}' scp/dev_bonafide.lst >> protocol.txt
awk '{print "spoof/" $1 " dev spoof"}' scp/spoof_dev.lst >> protocol.txt

# awk '{print "vocoded/hifigan_" $1 " dev spoof"}' scp/dev_bonafide.lst >> protocol.txt
# awk '{print "vocoded/waveglow_" $1 " dev spoof"}' scp/dev_bonafide.lst >> protocol.txt
# awk '{print "vocoded/hn-sinc-nsf-hifi_" $1 " dev spoof"}' scp/dev_bonafide.lst >> protocol.txt

# awk '{print "eval/" $2 ".wav eval " $5}' /datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt >> protocol.txt
# awk '{print "eval/" $2 ".wav eval " $5}' /datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt >> protocol.txt

awk '{print "eval/" $2 ".wav eval " $5}' /datab/Dataset/ASVspoof/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt >> protocol.txt
shuf -o protocol.txt < protocol.txt