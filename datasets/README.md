## edges2shoes

1. `./download_pix2pix_dataset.sh edges2shoes` download data from source
2. `python split_A_and_B.py --split train` and `python split_A_and_B.py --split train` split paired images to separate ones
3. `python create_edges2shoes_np.py` create 64x64 numpy data
