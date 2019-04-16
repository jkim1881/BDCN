import numpy as np

def get_column_as_list(arr, slack, pad_value=0):
    arr_temp_list = []
    for xslack in range(slack*2+1):
        real_xslack = xslack-slack
        if real_xslack < 0:
            arr_temp = np.pad(arr.copy()[:,:real_xslack], ((0,0),(0,-real_xslack)),
                              mode='constant', constant_values=pad_value)
        elif real_xslack > 0:
            arr_temp = np.pad(arr.copy()[:,real_xslack:], ((0,0),(real_xslack, 0)),
                              mode='constant', constant_values=pad_value)
        else:
            arr_temp = arr.copy()
        for yslack in range(slack*2+1):
            real_yslack = yslack-slack
            if real_yslack < 0:
                arr_temp_ = np.pad(arr_temp.copy()[:real_yslack, :], ((0, -real_yslack), (0, 0)),
                                  mode='constant',constant_values=pad_value)
            elif real_yslack > 0:
                arr_temp_ = np.pad(arr_temp.copy()[real_yslack:, :], ((real_yslack, 0), (0, 0)),
                                  mode='constant',constant_values=pad_value)
            else:
                arr_temp_ = arr_temp.copy()
            arr_temp_list.append(arr_temp_)
    return arr_temp_list

def min_w_slack(yhat, y, dist_func, slack, pad_value=0):
    y_column = np.array(get_column_as_list(y, slack, pad_value=pad_value))
    dist_image = np.min(dist_func(yhat, y_column), axis=0)
    return dist_image

if __name__ == '__main__':
    collist = get_column_as_list(np.ones((4,4)), slack=2, pad_value=0)
