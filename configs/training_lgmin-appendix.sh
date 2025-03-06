#!/bin/bash

python tiny_full_train.py --loss_type="lg_min" --sigma=0.75 --num_samples_reg_term=10
python tiny_full_train.py --loss_type="lg_min" --sigma=0.75 --num_samples_reg_term=25
python tiny_full_train.py --loss_type="lg_min" --sigma=0.75 --num_samples_reg_term=100
python tiny_full_train.py --loss_type="lg_min" --sigma=0.5 --num_samples_reg_term=10
python tiny_full_train.py --loss_type="lg_min" --sigma=0.5 --num_samples_reg_term=25
python tiny_full_train.py --loss_type="lg_min" --sigma=0.5 --num_samples_reg_term=100
python tiny_full_train.py --loss_type="lg_min" --sigma=0.95 --num_samples_reg_term=10
python tiny_full_train.py --loss_type="lg_min" --sigma=0.95 --num_samples_reg_term=25
python tiny_full_train.py --loss_type="lg_min" --sigma=0.95 --num_samples_reg_term=100
python tiny_full_train.py --loss_type="experimental_trace" --constrained_rqmin
python tiny_full_train.py --loss_type="lg_min" --sigma=0.25 --num_samples_reg_term=10
python tiny_full_train.py --loss_type="lg_min" --sigma=0.25 --num_samples_reg_term=25
python tiny_full_train.py --loss_type="lg_min" --sigma=0.25 --num_samples_reg_term=100