#!/bin/bash

path="../../MW_anisotropy/code/test_snaps/"
snap_name="MWLMC5_40M_20new_b0_6"
init_snap="142"
final_snap="143"

nmax="10"
lmax="5"
r_s_mw="40.82"
nmax_lmc="10"
lmax_lmc="5"
r_s_lmc="20.82"
out_name="ST_MWLMC5_n10_l5_142_143.txt"
LMC="1"
Nhalo="37500000"

#python3 bfe_coeff_parallel.py  --ncores=1 --path=../../MW_anisotropy/code/test_snaps/ --snap_name=MWLMC5_40M_20new_b0_6 --init_snap=142 --final_snap=143 --nmax=10 --lmax=1 --r_s_mw=40 --nmax_lmc=10 --lmax_lmc=1 --r_s_lmc=10 --out_name=test --Nhalo=37500000 

python3 bfe_coeff.py $path $snap_name $init_snap $final_snap $nmax $lmax $r_s_mw $nmax_lmc $lmax_lmc $r_s_lmc $out_name $LMC $Nhalo
