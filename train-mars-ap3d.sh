python train.py --root ../data/ -d mars --arch ap3dres50 --gpu 0 --save_dir log-mars-ap3d #
python test-all.py --root ../data/ -d mars --arch ap3dres50 --gpu 0 --resume log-mars-ap3d