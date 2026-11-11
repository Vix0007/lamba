#!/usr/bin/env bash
# once the rented box is set up, push the n<=2 version of the miner before its mining step starts
until grep -q BOX14_SETUP_DONE ~/v40t7_boxB.log; do sleep 10; done
rsync -a -e "ssh -i $HOME/.ssh/vix_fleet.pem" ~/antidoom_data/mine_constv12c.py ubuntu@$(cat ~/.lamba-box14a.ip):~/antidoom_data/ && echo "BOXB_MINER_PATCHED $(TZ=Asia/Singapore date +%T)" >> ~/v40t7.log
