import numpy as np
import scipy.misc
import scipy.io
import os

# parameters
in_dataset = '/media/data_cifs/pytorch_projects/datasets/Multicue/multicue'
out_datset = '/media/data_cifs/pytorch_projects/datasets/Multicue_crops'
img_size = [720, 1280]
crop_size = 500

# get list of images and gts from a specified path.
image_dir = os.path.join(in_dataset, 'images')
gt_boundaries_dir = os.path.join(in_dataset, 'ground-truth/images/boundaries')
gt_edges_dir = os.path.join(in_dataset, 'ground-truth/images/edges')
image_list = os.listdir(image_dir)
gt_boundaries_list = os.listdir(gt_boundaries_dir)
gt_edges_list = os.listdir(gt_edges_dir)
image_filenames_int = [file.split('.')[0].split('_')[0] for file in image_list if '.png' in file]
gt_boundaries_filenames_int = [file.split('.')[0].split('_')[0] for file in gt_boundaries_list if '.png' in file]
gt_edges_filenames_int = [file.split('.')[0].split('_')[0] for file in gt_edges_list if '.png' in file]
image_filenames_int = list(np.unique(np.array(image_filenames_int)))
gt_boundaries_filenames_int = list(np.unique(np.array(gt_boundaries_filenames_int)))
gt_edges_filenames_int = list(np.unique(np.array(gt_edges_filenames_int)))
image_filenames_int.sort()
gt_boundaries_filenames_int.sort()
gt_edges_filenames_int.sort()

# sanity check. Partition it into train, val and test
if not ((image_filenames_int == gt_boundaries_filenames_int)
        and (image_filenames_int == gt_edges_filenames_int)):
    raise ValueError('image_filenames and gt_boundaries/edges_filenames do not match.')
else:
    image_fn_int_dict = {'train': image_filenames_int[:int(0.8*len(image_filenames_int))],
                         'test': image_filenames_int[int(0.8*len(image_filenames_int)):],
                         'test_nocrop': image_filenames_int[int(0.8 * len(image_filenames_int)):]}

# loop over examples and apply crops
for train_val in ['test_nocrop']: # ['train','test','test_nocrop']:
    os.makedirs(os.path.join(out_datset, 'data', 'images', train_val))
    os.makedirs(os.path.join(out_datset, 'data', 'groundTruth', train_val))
    for fn in image_fn_int_dict[train_val]:
        print('filename: ' + fn)
        candidate_img_timestamps = [img_filename.split('_')[-1].split('.')[0] for img_filename in image_list if fn in img_filename]
        candidate_img_timestamps.sort()
        timestamp = candidate_img_timestamps[-1]

        img = scipy.misc.imread(os.path.join(in_dataset, 'images', fn + '_left_' + timestamp + '.png'))
        gt_boundaries = [scipy.misc.imread(os.path.join(in_dataset, 'ground-truth/images/boundaries', fn + '_left_' + timestamp + '.' + subj_id + '.png'))
                         for subj_id in ['1','2','3','4','5']]
        gt_edges = [scipy.misc.imread(os.path.join(in_dataset, 'ground-truth/images/edges', fn + '_left_' + timestamp + '.' + subj_id + '.png'))
                         for subj_id in ['1','2','3','4','5']]

        ### PREPROC GT
        gt_mean_boundaries = np.asarray(gt_boundaries, dtype=np.float32).mean(0)
        if len(gt_mean_boundaries.shape) == 3:
            gt_mean_boundaries = gt_mean_boundaries[:, :, 0]
        if np.max(gt_boundaries[0]) == 255:
            print('max gt boundaries value is 255. normalizing to [0,1]')
            gt_mean_boundaries /= 255.
        if np.max(gt_boundaries[0]) == 65535:
            print('max gt boundaries value is 65535. normalizing to [0,1]')
            gt_mean_boundaries /= 65535.
        else:
            print('max gt boundaries value is ' + str(np.max(gt_boundaries[0])) + '. Not doing normalization (?????)')
        gt_mean_edges = np.asarray(gt_edges, dtype=np.float32).mean(0)
        if len(gt_mean_edges.shape) == 3:
            gt_mean_edges = gt_mean_edges[:, :, 0]
        if np.max(gt_edges[0]) == 255:
            print('max gt edges value is 255. normalizing to [0,1]')
            gt_mean_edges /= 255.
        if np.max(gt_edges[0]) == 65535:
            print('max gt edges value is 65535. normalizing to [0,1]')
            gt_mean_edges /= 65535.
        else:
            print('max gt edges value is ' + str(np.max(gt_edges[0])) + '. Not doing normalization (?????)')


        ### CROP AND SAVE
        if train_val is 'test_nocrop':
            scipy.misc.imsave(os.path.join(out_datset, 'data', 'images', train_val, fn + '.jpg'),
                              img)
            np.save(os.path.join(out_datset, 'data', 'groundTruth', train_val,
                                 fn + '.boundaries.npy'),
                    gt_mean_boundaries)
            np.save(os.path.join(out_datset, 'data', 'groundTruth', train_val, fn + '.edges.npy'),
                    gt_mean_edges)
        else:
            for i_th_crop in range(10):
                offset = [np.random.randint(low=0, high=img.shape[0]-crop_size), np.random.randint(low=0, high=img.shape[1]-crop_size)]
                im_crop = img[offset[0]:offset[0]+crop_size, offset[1]:offset[1]+crop_size,:]
                gt_boundaries_crop = gt_mean_boundaries[offset[0]:offset[0]+crop_size, offset[1]:offset[1]+crop_size]
                gt_edges_crop = gt_mean_edges[offset[0]:offset[0]+crop_size, offset[1]:offset[1]+crop_size]

                scipy.misc.imsave(os.path.join(out_datset, 'data', 'images', train_val, fn + '_' + str(i_th_crop) + '.jpg'),
                                  im_crop)
                np.save(os.path.join(out_datset, 'data', 'groundTruth', train_val, fn + '_' + str(i_th_crop) + '.boundaries.npy'),
                                    gt_boundaries_crop)
                np.save(os.path.join(out_datset, 'data', 'groundTruth', train_val, fn + '_' + str(i_th_crop) + '.edges.npy'),
                                    gt_edges_crop)
    print('FIN')