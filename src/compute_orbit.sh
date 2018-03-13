#!/bin/bash


x_in="10"
y_in="0"
z_in="-100"
vx_in="0"
vy_in="100"
vz_in="-50"

time="2.2"
orbit_dt="0.01"
static="1"

r_s="40.85"
nmax="20"
lmax="5"



path_coeff="./coefficients/ST_MWST_MWLMC6_beta0_100M_snap_"
path_times="./coefficients/ST_MWST_MWLMC6_beta0_100M_snap_"
orbit_name="test_obit_LMC6"
disk="0"

LMC="0"
path_coeff_LMC="./coefficients/ST_LMCST_MWLMC6_beta0_100M_snap_"
nmax_lmc="20"
lmax_lmc="5"
r_s_lmc="25"


python3 orbit.py $x_in $y_in $z_in $vx_in $vy_in $vz_in $time\
                 $orbit_dt $static $r_s $nmax $lmax $path_coeff\
                 $path_times $orbit_name $disk $LMC $path_coeff_LMC\
                 $nmax_lmc $lmax_lmc $r_s_lmc

