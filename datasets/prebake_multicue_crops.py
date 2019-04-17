import numpy as np
import scipy.misc
import scipy.io
import os

# parameters
in_dataset = '/media/data_cifs/pytorch_projects/datasets/Multicue/multicue'
out_datset = '/media/data_cifs/pytorch_projects/datasets/Multicue_crops'
task = 'boundaries'
img_size = [720, 1280]
crop_size = 500

# get list of images and gts from a specified path.
image_dir = os.path.join(in_dataset, 'images')
gt_dir = os.path.join(in_dataset, 'ground-truth/images', task)
image_list = os.listdir(image_dir)
gt_list = os.listdir(gt_dir)
image_filenames_int = [file.split('.')[0].split('_')[0] for file in image_list if '.png' in file]
gt_filenames_int = [file.split('.')[0].split('_')[0] for file in gt_list if '.png' in file]
image_filenames_int = list(np.unique(np.array(image_filenames_int)))
gt_filenames_int = list(np.unique(np.array(gt_filenames_int)))
image_filenames_int.sort()
gt_filenames_int.sort()

# sanity check. Partition it into train, val and test
if not (image_filenames_int == gt_filenames_int):
    raise ValueError('image_filenames and gt_filenames do not match.')
else:
    image_fn_int_dict = {'train': image_filenames_int[:int(0.8)*len(image_filenames_int)],
                         'test': image_filenames_int[int(0.8)*len(image_filenames_int):]}
    gt_filenames_int = {'train': gt_filenames_int[:int(0.8)*len(gt_filenames_int)],
                         'test': gt_filenames_int[int(0.8)*len(gt_filenames_int):]}

# loop over examples and apply crops
for train_val in ['train','test']:
    os.makedirs(os.path.join(out_datset, 'data', 'images', train_val))
    os.makedirs(os.path.join(out_datset, 'data', 'groundTruth', train_val))
    for fn in image_fn_int_dict[train_val]:
        print('filename: ' + fn)
        candidate_img_timestamps = [img_filename.split('_')[-1] for img_filename in image_list if fn in file]
        candidate_img_timestamps.sort()
        timestamp = candidate_img_timestamps[-1]

        img = scipy.misc.imread(os.path.join(in_dataset, 'data', 'images', fn + '_left_' + timestamp + '.png'))
        gt = [scipy.misc.imread(os.path.join(in_dataset, 'data', 'ground-truth/images', fn + '_left_' + timestamp + '.' + subj_id + '.png'))
              for subj_id in ['1','2','3','4','5']]

        ### PREPROC GT
        gt_mean = np.asarray(gt, dtype=np.float32).mean(0)
        if len(gt_mean.shape) == 3:
            gt_mean = gt_mean[:, :, 0]
        if np.max(gt[0]) == 255:
            print('max gt value is 255. normalizing to [0,1]')
            gt_mean /= 255.
        else:
            print('max gt value is ' + str(np.max(gt[0])) + '. Not doing normalization (?????)')

        ### CROP AND SAVE
        for i_th_crop in range(10):
            offset = [np.random.randint(low=0, high=img.shape[0]-crop_size), np.random.randint(low=0, high=img.shape[1]-crop_size)]
            im_crop = img[offset[0]:offset[0]+crop_size, offset[1]:offset[1]+crop_size,:]
            gt_crop = gt_mean[offset[0]:offset[0]+crop_size, offset[1]:offset[1]+crop_size,:]

            scipy.misc.imsave(os.path.join(out_datset, 'data', 'images', train_val, fn + '_' + str(i_th_crop) + '.jpg'),
                              im_crop)
            np.save(os.path.join(out_datset, 'data', 'groundTruth', train_val, fn + '_' + str(i_th_crop) + '.npy'),
                              gt_crop)

    print('FIN')