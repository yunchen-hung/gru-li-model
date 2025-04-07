# python -u main.py --exp VaryParam --setup setup_4features.json
# python -u main.py --exp VaryParam --setup setup_4features_16hidden.json
# python -u main.py --exp VaryParam --setup setup_4features_32hidden.json
python -u main.py --exp VaryParam --setup setup.json
python -u main.py --exp VaryParam --setup setup_featureinput.json
python -u main.py --exp VaryParam --setup setup_memtransform.json
python -u main.py --exp VaryParam --setup setup_gamma0.json
# python -u main.py --exp VaryParam --setup setup_nopretrain.json
