#!/bin/bash

python tiny_full_train.py --loss_type="lg_min" --sigma=0.75 --num_samples_reg_term=25
python tiny_full_train.py --loss_type="rq_min" --constrained_rqmin
python tiny_full_train.py --loss_type="nt_xent" 
python tiny_full_train.py --loss_type="spectral_contrastive"