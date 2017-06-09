#!/bin/bash

path="/media/ngaravito/4fb4fd3d-1665-4892-a18d-bdbb1185a07b/simulations/LMCMW40M/MWLMC6/"
snap_name="MWLMC6_40M_b0"
init_snap="115"
final_snap="140"

nmax="12"
lmax="5"
r_s_mw="40.82"
out_name="ST_MWLMC6_n12_l5_115_140.txt"
LMC="1"
Nhalo="37500000"


python3 bfe_coeff.py $path $snap_name $init_snap $final_snap $nmax $lmax $r_s_mw $out_name $LMC $Nhalo
