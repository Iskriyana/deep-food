#!/usr/bin/env python

"""
   generate_artifical_validation_dataset.py:
   -------------------------------------------------------------------------------------------
   Script for generating a new artifical validation set and saving it to disk.
   If an existing directory is given, it will replace the contents with new images.

Usage:
    python generate_artifical_validation_dataset.py OUTPUT_DIRECTORY

Required:
    OUTPUT_DIRECTORY    Where to save the image files

Options:
    --seed INT          Seed for random number generator
"""

import os
import sys
import argparse
import data_helpers
import shutil
import cv2
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('OUTPUT_DIRECTORY')
parser.add_argument("--seed", type=int, default=11,
                    help="Seed for random number generator")
parser.add_argument("--N_samples", type=int, default=10,
                    help="Number of samples to generate")
args = parser.parse_args()

def main():

    # get arguments
    seed = int(args.seed)
    OUTPUT_DIRECTORY = str(args.OUTPUT_DIRECTORY)
    N_samples = int(args.N_samples)

    # Load image files
    data_directories = [str(Path().absolute().parents[0]/'data/FIDS30'),
                        str(Path().absolute().parents[0]/'data/original_clean'),
                        str(Path().absolute().parents[0]/'data/update_9sep'),
                        str(Path().absolute().parents[0]/'data/marianne_update'),
                        str(Path().absolute().parents[0]/'data/data_gleb_upsample'),
                        str(Path().absolute().parents[0]/'data/semi_artificial'),]
    test_size = 0.1
    _, data_df_test, _, _, _ = data_helpers.get_train_test_data_df(data_directories, test_size, seed)

    # Generate artifical validation dataset
    print('Generating images...')
    N_samples = N_samples
    N_min = 5
    N_max = 10
    spacing = 150
    size_jitter=(0.9,1.5)
    bg_path = 'artificial_background'

    val_data = data_helpers.generate_artifical_validation_dataset(data_df_test,
                                                                  bg_path,
                                                                  N_samples=N_samples,
                                                                  N_min=N_min,
                                                                  N_max=N_max,
                                                                  spacing=spacing,
                                                                  size_jitter=size_jitter,
                                                                  seed=seed)

    # Save to disk
    if os.path.isdir(OUTPUT_DIRECTORY):
        shutil.rmtree(OUTPUT_DIRECTORY)
    os.mkdir(OUTPUT_DIRECTORY)

    # Save samples to disk
    print('Saving images...')
    for i in tqdm.tqdm(val_data.keys()):
        img = val_data[i]['image']
        labels = val_data[i]['labels']

        # save image
        cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, f'sample{int(i)}.jpg'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # save labels
        with open(os.path.join(OUTPUT_DIRECTORY, f'sample{int(i)}.txt'), 'w') as f:
            for l in labels:
                f.write(l)
                f.write('\n')


    print(f'{N_samples} samples saved to folder {OUTPUT_DIRECTORY}, ')

if __name__ == '__main__':
    main()
