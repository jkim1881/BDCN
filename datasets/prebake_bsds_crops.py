import numpy as np
import scipy.misc
import scipy.io
import os

# parameters
in_dataset = '/media/data_cifs/pytorch_projects/datasets/BSDS500'
out_datset = '/media/data_cifs/pytorch_projects/datasets/BSDS500_crops'
img_size = [321, 481]
crop_size = 320

for train_val in ['test_nocrop']: # ['train','test','test_nocrop']:
    # get list of images and gts from a specified path
    image_dir = os.path.join(in_dataset, 'data', 'images', 'test' if train_val is 'test_nocrop' else train_val)
    gt_dir = os.path.join(in_dataset, 'data', 'groundTruth', 'test' if train_val is 'test_nocrop' else train_val)
    image_list = os.listdir(image_dir)
    gt_list = os.listdir(gt_dir)
    image_filenames_int = [file.split('.')[0] for file in image_list if '.jpg' in file]
    gt_filenames_int = [file.split('.')[0] for file in gt_list if '.mat' in file]
    image_filenames_int.sort()
    gt_filenames_int.sort()

    # sanity check
    if not (image_filenames_int == gt_filenames_int):
        raise ValueError('image_filenames and gt_filenames do not match.')
    else:
        files = [str(integer) for integer in image_filenames_int]

    # loop over examples and apply crops
    os.makedirs(os.path.join(out_datset,'data', 'images', train_val))
    os.makedirs(os.path.join(out_datset, 'data', 'groundTruth', train_val))
    for fn in files:
        print('filename: ' + fn)
        img = scipy.misc.imread(os.path.join(in_dataset, 'data', 'images', train_val, fn + '.jpg'))
        gt = scipy.io.loadmat(os.path.join(in_dataset, 'data', 'groundTruth', train_val, fn + '.mat'))['groundTruth'].reshape(-1)

        ### PREPROC GT
        gt_arr = []
        for idx, gt_subj in enumerate(gt):
            gt_arr += [gt_subj.item()[1].astype(np.float32)]
        gt_mean = np.asarray(gt_arr, dtype=np.float32).mean(0)
        if len(gt_mean.shape) == 3:
            gt_mean = gt_mean[:, :, 0]

        ### CROP AND SAVE
        if train_val is 'test_nocrop':
            scipy.misc.imsave(os.path.join(out_datset, 'data', 'images', train_val, fn + '.jpg'),
                              img)
            np.save(os.path.join(out_datset, 'data', 'groundTruth', train_val, fn + '.npy'),
                    gt_mean)
        else:
            for offset in [0, 39, 79, 119, 159]:
                if (img.shape[0]==img_size[0]) and (img.shape[1]==img_size[1]):
                    im_crop = img[:crop_size, offset:offset+crop_size,:]
                    gt_crop = gt_mean[:crop_size, offset:offset+crop_size]
                elif (img.shape[0]==img_size[1]) and (img.shape[1]==img_size[0]):
                    im_crop = img[offset:offset+crop_size, :crop_size]
                    gt_crop = gt_mean[offset:offset+crop_size, :crop_size]
                else:
                    raise ValueError('img shape must be '+str(img_size))

                scipy.misc.imsave(os.path.join(out_datset, 'data', 'images', train_val, fn + '_' + str(offset) + '.jpg'),
                                  im_crop)
                np.save(os.path.join(out_datset, 'data', 'groundTruth', train_val, fn + '_' + str(offset) + '.npy'),
                                  gt_crop)

    print('FIN')