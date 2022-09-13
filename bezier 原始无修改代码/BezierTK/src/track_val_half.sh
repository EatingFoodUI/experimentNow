if false;then
python track.py  --frame_stride 1 --sliding_stride 1 && 
python track.py  --frame_stride 1 --sliding_stride 2 &&
python track.py  --frame_stride 1 --sliding_stride 3 &&
python track.py  --frame_stride 1 --sliding_stride 4 &&
python track.py  --frame_stride 1 --sliding_stride 5 &&
python track.py  --frame_stride 1 --sliding_stride 6 
fi



python track.py  --sliding_stride 6 --clip_len 1.0 &&
python track.py  --sliding_stride 10 --clip_len 1.0 
