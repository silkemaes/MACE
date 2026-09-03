'''
Routines to split the training dataset into smaller chunks.
This is useful for batch training, which requires all samples to have the same size.
The splitting is based on a fixed window length, a randomised starting point,
overlapping windows (i.e. the stride is smaller than the window length).
'''

import os
import random
import multiprocessing as mp

window_length = 2000
window_stride = 1000

dataset_ratios = (0.7,  # training set
                  0.15, # validation set
                  0.15) # test set
dir_data = '/STER/hydroModels/camille/phantom/macetraining/3d/v17-5_5e7_a20/chem_trace/'
dir_save = '/STER/hydroModels/camille/phantom/macetraining/MACE/data'

def split_dataset(file, dir_data, dir_save, window_length, window_stride, start_index=0):
    # read in the data
    data = []
    with open(os.path.join(dir_data, file), 'r') as f:
        for line in f:
            data.append(line.strip().split())
    data = data[1:]  # skip the header
    if len(data) - start_index < window_length:
        print(f"    - File {file} is too short ({len(data)} lines), saving as is.", flush=True)
        chunk_file = os.path.join(dir_save, f"{file[:-5]}_00.chem")
        with open(chunk_file, 'w') as f:
            for line in data[start_index:]:
                f.write(' '.join(line) + '\n')
        return
    # split the data into chunks
    chunk_count = 0
    for start in range(start_index, len(data) - window_length + 1, window_stride):
        chunk = data[start:start + window_length]
        chunk_count += 1
        # then write the chunks to new files in a new directory
        chunk_file = os.path.join(dir_save, f"{file[:-5]}_{str(chunk_count).zfill(2)}.chem")
        with open(chunk_file, 'w') as f:
            for line in chunk:
                f.write(' '.join(line) + '\n')
    print(f"{file} split into {chunk_count} chunks.", flush=True)


if __name__ == '__main__':

    dir_split = os.path.join(dir_data, 'split/')
    os.makedirs(dir_split, exist_ok=True)

    # clear output directory
    for f in os.listdir(dir_split):
        if f.endswith('.chem'):
            os.remove(os.path.join(dir_split, f))


    # read in all .chem files in the directory
    files_chem = [f for f in os.listdir(dir_data) if f.endswith('.chem')]
    num_files = len(files_chem)
    print(f'Found {num_files} .chem files in {dir_data}.')

    print('Sorting files and distributing into training, validation and test sets...')
    # sort the files by the integer in the filename (e.g. 1.chem, 2.chem, etc.)
    files_chem.sort(key=lambda x: int(x.split('.')[0]))

    # put the files into bins to ensure that the training, validation and test sets are representative of the entire dataset
    num_bins = 50
    if num_files < num_bins:
        print('Fatal error: not enough files to split into bins. Please reduce the number of bins or add more files.')
        raise SystemExit(1)
    bins = [[] for _ in range(num_bins)]
    bin_size = len(files_chem) // num_bins
    for i in range(len(bins)):
        bins[i] = files_chem[i*bin_size:(i+1)*bin_size]
    if len(files_chem) % num_bins != 0:
        bins[-1] += files_chem[num_bins*bin_size:]  # add the remaining files to the last bin

    
    # shuffle data within the bins
    random.seed(0)
    for bin in bins:
        random.shuffle(bin)

    # split the bins into training, validation and test sets
    print(f'- Splitting {num_files} files into training, validation and test sets with ratios {dataset_ratios}...')
    train_files = []
    valid_files = []
    test_files = []
    for bin in bins:
        train_files += bin[:int(len(bin) * dataset_ratios[0])]
        valid_files += bin[int(len(bin) * dataset_ratios[0]):int(len(bin) * (dataset_ratios[0] + dataset_ratios[1]))]
        test_files += bin[int(len(bin) * (dataset_ratios[0] + dataset_ratios[1])):]

    # save the file lists to text files
    with open(os.path.join(dir_save, 'path_train.txt'), 'w') as f:
        for file in train_files:
            f.write(os.path.join(dir_data, file) + '\n')
    print(f"Training set saved with {len(train_files)} files.", flush=True)
    with open(os.path.join(dir_save, 'path_valid.txt'), 'w') as f:
        for file in valid_files:
            f.write(os.path.join(dir_data, file) + '\n')
    print(f"Validation set saved with {len(valid_files)} files.", flush=True)
    with open(os.path.join(dir_save, 'path_test.txt'), 'w') as f:
        for file in test_files:
            f.write(os.path.join(dir_data, file) + '\n')
    print(f"Test set saved with {len(test_files)} files.", flush=True)

    # get number of cpus available
    num_cpus = mp.cpu_count()//2
    print('Splitting dataset into smaller chunks for batch training...', flush=True)
    print(f'- Using {num_cpus} CPUs.', flush=True)
    # split training data in parallel 
    with mp.Pool(processes=num_cpus) as pool:
        for file in train_files:
            # randomise the starting point for each file
            start_index = random.randint(0, window_stride - 1)
            pool.apply_async(split_dataset, args=(file, dir_data, dir_split, window_length, window_stride, start_index))
        pool.close()
        pool.join()
        
    
    # save split data paths to text files
    with open(os.path.join(dir_save, 'path_train_split.txt'), 'w') as f:
        # sort first
        files = [f for f in os.listdir(dir_split) if f.endswith('.chem')]
        files.sort(key=lambda x: int(x.split('_')[0]))
        for file in files:
            f.write(os.path.join(dir_data, 'split', file) + '\n')

    print("DONE - happy training!", flush=True)
        