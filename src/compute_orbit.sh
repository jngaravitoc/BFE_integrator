#!/bin/bash


x_in="198.7393"
y_in="-47.8209"
z_in="7.2483"
vx_in="-172.159"
vy_in="-113.663"
vz_in="82.7473"

time="0.5"
orbit_dt="0.001"
static="1"

r_s="40.85"
nmax="15"
lmax="8"

path_coeff="../coefficients/ST_MWMWLMC6_40M_s15_l8.txt"
path_times="../coefficients/times_MWMWLMC6_40M_s15_l8.txt"
orbit_name="test_stratic_orbit_LMC6"
disk="1"

LMC="1"
path_coeff_LMC="../coefficients/ST_LMCMWLMC6_40M_s15_l8.txt"
nmax_lmc="15"
lmax_lmc="10"
r_s_lmc="20"


python3 orbit.py $x_in $y_in $z_in $vx_in $vy_in $vz_in $time\
                 $orbit_dt $static $r_s $nmax $lmax $path_coeff\
                 $path_times $orbit_name $disk $LMC $path_coeff_LMC\
                 $nmax_lmc $lmax_lmc $r_s_lmc
 
