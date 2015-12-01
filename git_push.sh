#!/bin/bash

echo $1
git add cox_nnet/ README.md example.py WIHS/ WIHS_plots.r log_likelihoods.png cox_mlp_WIHS.png setup.cfg setup.py
#git add cox_nnet.py README.md example.py WIHS/ WIHS_plots.r log_likelihoods.png cox_mlp_WIHS.png setup.cfg setup.py
git commit -m "$1"
git push --force origin master
