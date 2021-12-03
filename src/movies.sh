#!/bin/bash

frame_rate='10'
video_name='BFE_MWLMC6_beta0_dens_0.mp4'
ffmpeg  -r $frame_rate -i ./bfe_2ddensity_MWLMC6_snap_%3d.png -s 1920x1080 -vcodec libx264 -crf 15 -pix_fmt yuv420p $video_name

