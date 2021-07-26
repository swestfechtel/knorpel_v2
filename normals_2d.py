import logging
import math
import sys
import traceback
import utility
import os
import inspect

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from multiprocessing import Pool, cpu_count
from functools import partial


def average_femoral_thickness_per_region(np_image, sitk_image):
    """
    Calculates the average thickness per subregion of the femoral cartilage
    :param np_image: numpy array representation of the mri scan
    :return: two dictionaries containing the summed thickness and the denominator for averaging per region, respectively
    """
    logging.debug(f'BEGIN {inspect.currentframe().f_code.co_name}')

    femoral_cartilage = utility.build_3d_cartilage_array(np_image, 3)
    vectors = [list(elem) for elem in femoral_cartilage]
    cluster = KMeans(n_clusters=1, random_state=0).fit(vectors)
    split_vector = cluster.cluster_centers_[0]
    femoral_split_vector = split_vector
    left_plate, right_plate = utility.split_into_plates(vectors, split_vector)

    first_split, second_split = utility.get_femoral_thirds(left_plate)
    left_femoral_regions = [first_split, second_split]

    first_split, second_split = utility.get_femoral_thirds(right_plate)
    right_femoral_regions = [first_split, second_split]

    with Pool() as pool:
        layers = pool.map(partial(utility.isolate_cartilage, color_code=3), np_image)

    chunksize = len(layers) // cpu_count()
    iterable = []
    i = 0
    for layer in layers:
        iterable.append((layer, left_femoral_regions, right_femoral_regions, i, 3, femoral_split_vector, sitk_image))
        i += 1

    with Pool() as pool:
        res_list = pool.starmap(func=function_for_pool, iterable=iterable, chunksize=chunksize)

    new_res = np.array(res_list)
    mask = new_res != [0, 0]
    new_res = np.extract(mask, new_res)

    i = 0
    thickness_list = list()
    count_list = list()
    while i < len(new_res):
        thickness_list.append(new_res[i])
        count_list.append(new_res[i + 1])
        i += 2

    logging.debug(f'END {inspect.currentframe().f_code.co_name}')
    return thickness_list, count_list


def average_tibial_thickness_per_region(np_image, sitk_image):
    """
    Calculates the average thickness per subregion of the tibial cartilage
    :param np_image: numpy array representation of the mri scan
    :return: two dictionaries containing the summed thickness and the denominator for averaging per region, respectively
    """
    logging.debug(f'BEGIN {inspect.currentframe().f_code.co_name}')

    tibial_cartilage = utility.build_3d_cartilage_array(np_image, 4)
    vectors = [list(elem) for elem in tibial_cartilage]
    cluster = KMeans(n_clusters=1, random_state=0).fit(vectors)
    split_vector = cluster.cluster_centers_[0]
    tibial_split_vector = split_vector
    left_plate, right_plate = utility.split_into_plates(vectors, split_vector)

    left_plate_cog = KMeans(n_clusters=1, random_state=0).fit(left_plate).cluster_centers_[0]
    right_plate_cog = KMeans(n_clusters=1, random_state=0).fit(right_plate).cluster_centers_[0]

    left_plate_radius, left_plate_circle = utility.calculate_ellipse(left_plate, left_plate_cog)
    right_plate_radius, right_plate_circle = utility.calculate_ellipse(right_plate, right_plate_cog)

    # left plate first
    a, b, c, d = utility.get_plate_corners(plate=left_plate)
    left_tibial_regions = [a, b, c, d, left_plate_radius, left_plate_cog]
    # then right plate
    a, b, c, d = utility.get_plate_corners(plate=right_plate)
    right_tibial_regions = [a, b, c, d, right_plate_radius, right_plate_cog]

    with Pool() as pool:
        layers = pool.map(partial(utility.isolate_cartilage, color_code=4), np_image)

    chunksize = len(layers) // cpu_count()
    iterable = []
    i = 0
    for layer in layers:
        iterable.append((layer, left_tibial_regions, right_tibial_regions, i, 4, tibial_split_vector, sitk_image))
        i += 1

    with Pool() as pool:
        res_list = pool.starmap(func=function_for_pool, iterable=iterable, chunksize=chunksize)

    new_res = np.array(res_list)
    mask = new_res != [0, 0]
    new_res = np.extract(mask, new_res)

    i = 0
    thickness_list = list()
    count_list = list()
    while i < len(new_res):
        thickness_list.append(new_res[i])
        count_list.append(new_res[i + 1])
        i += 2

    logging.debug(f'END {inspect.currentframe().f_code.co_name}')
    return thickness_list, count_list


def function_for_pool(layer, left_regions, right_regions, layer_index, color_code, split_vector, sitk_image) -> [dict, dict]:
    """
    Calculates the total thickness of a layer, wrapper for multiprocessing

    :param layer: numpy array representation of the layer
    :param left_regions: specific features describing the left plate's subregions
    :param right_regions: specific features describing the right plate's subregions
    :param layer_index: y-axis index of the layer
    :param color_code: color coding of the cartilage (3 for tibia, 4 for femur)
    :param split_vector: the vector with which to split into left and right plate
    :return: layer thickness, number of normals
    """
    logging.debug(f'BEGIN {inspect.currentframe().f_code.co_name}')

    if len(layer) == 0:
        logging.debug(f'{inspect.currentframe().f_code.co_name} layer is empty')
        return [0, 0]

    arr = utility.build_array(layer, isolate=True, isolator=color_code)
    if len(arr) == 0:
        logging.debug(f'{inspect.currentframe().f_code.co_name} array is empty')
        return [0, 0]

    x, y = utility.get_x_y(arr[0])
    logging.debug(f'{inspect.currentframe().f_code.co_name} 1')
    sup_vectors = pd.DataFrame(list(zip(x, y)), columns=['x', 'y'])
    logging.debug(f'{inspect.currentframe().f_code.co_name} 2')
    try:
        z = np.polyfit(x, y, 2)
        fun = np.poly1d(z)
    except np.linalg.LinAlgError as e:
        logging.debug(f'{inspect.currentframe().f_code.co_name} {traceback.format_exc()}')
        return [0, 0]
    except Exception as e:
        logging.debug(f'{inspect.currentframe().f_code.co_name} {traceback.format_exc()}')
        return [0, 0]

    logging.debug(f'{inspect.currentframe().f_code.co_name} 3')
    x_new = np.linspace(x[0], x[-1], (x[-1] - x[0]) * 100)
    logging.debug(f'{inspect.currentframe().f_code.co_name} 4')
    normals = calculate_normals(x_new, sup_vectors, fun)
    logging.debug(f'{inspect.currentframe().f_code.co_name} 5')

    logging.debug(f'{inspect.currentframe().f_code.co_name} 6')
    max_vectors = sup_vectors.groupby(sup_vectors['x']).max().y.to_numpy()
    logging.debug(f'{inspect.currentframe().f_code.co_name} 7')
    min_vectors = sup_vectors.groupby(sup_vectors['x']).min().y.to_numpy()
    logging.debug(f'{inspect.currentframe().f_code.co_name} 8')

    if color_code == 4:
        logging.debug(f'END {inspect.currentframe().f_code.co_name}')
        return calculate_tibial_thickness(max_vectors, min_vectors, normals, layer_index, left_regions, right_regions, split_vector, sitk_image)
    elif color_code == 3:
        logging.debug(f'END {inspect.currentframe().f_code.co_name}')
        return calculate_femoral_thickness(max_vectors, min_vectors, normals, layer_index, left_regions, right_regions, split_vector, sitk_image)
    else:
        logging.debug(f'END {inspect.currentframe().f_code.co_name}')
        raise ValueError(f'Color code mismatch: {color_code}')


def calculate_femoral_thickness(max_vectors, min_vectors, normals, layer_index, left_regions, right_regions, split_vector, sitk_image):
    """
    Calculates the total thickness of a 2d femoral cartilage layer.

    :param max_vectors: Maximum z values of the cartilage data points
    :param min_vectors: Minimum z values of the cartilage data points
    :param normals: the normals along which to calculate the thickness
    :param layer_index: the y index of the current layer
    :param left_regions: corners a, b, c, d as well as radius and center of gravity of the left plate
    :param right_regions: corners a, b, c, d as well as radius and center of gravity of the right plate
    :param split_vector: the vector with which to split into left and right plate
    :return: total thickness, number of normals
    """
    logging.debug(f'BEGIN {inspect.currentframe().f_code.co_name}')

    average_thickness = dict()
    normal_count = dict()

    average_thickness['ecLF'] = 0
    average_thickness['ccLF'] = 0
    average_thickness['icLF'] = 0
    average_thickness['icMF'] = 0
    average_thickness['ccMF'] = 0
    average_thickness['ecMF'] = 0

    normal_count['ecLF'] = 0
    normal_count['ccLF'] = 0
    normal_count['icLF'] = 0
    normal_count['icMF'] = 0
    normal_count['ccMF'] = 0
    normal_count['ecMF'] = 0

    fail_count = 0
    x_val = 0

    for norm_vec in normals:
        try:
            x0 = norm_vec[0]
            norm = norm_vec[1]
            norm = norm[norm > 0]
            # set comparison context for current normal: x +- 1
            min_ext = min_vectors[x0 - 1:x0 + 2]
            max_ext = max_vectors[x0 - 1:x0 + 2]

            # find the closest minimum and maximum points to the normal, i.e. lowest and highest points where the normal
            # cuts the cartilage
            min_y_indices = np.argwhere([abs(norm - x) < 0.1 for x in min_ext])
            max_y_indices = np.argwhere([abs(norm - x) < 0.1 for x in max_ext])

            min_y_indices = [x[1] for x in min_y_indices]
            max_y_indices = [x[1] for x in max_y_indices]
            # get the x and y values for the maximum intersection point
            max_y = norm[max_y_indices].max()
            max_x = np.where(norm == max_y)[0][0]

            # get the x and y values for the minimum intersection point
            min_y = norm[min_y_indices].min()
            min_x = np.where(norm == min_y)[0][0]

            # assert that the right values were fetched
            assert max_y == norm[max_x]
            assert min_y == norm[min_x]

            max_x /= 100
            min_x /= 100

            x = np.array([max_x, max_y])
            y = np.array([min_x, min_y])

            # get the subregion of the current normal
            tmp = tuple([x_val, layer_index])
            label = utility.classify_femoral_point(tmp, left_regions, right_regions, split_vector)
            # calculate distance between the two vectors
            vec_dist = utility.vector_distance(x, y) * sitk_image.GetSpacing()[2]
            if not vec_dist > 0:
                raise ValueError

            average_thickness[label] += vec_dist
            normal_count[label] += 1
            x_val += 1
        except (ValueError, AssertionError, KeyError):
            fail_count += 1
            x_val += 1
            continue

    logging.debug(f'END {inspect.currentframe().f_code.co_name}')
    return [average_thickness, normal_count]


def calculate_tibial_thickness(max_vectors, min_vectors, normals, layer_index, left_regions, right_regions, split_vector, sitk_image):
    """
    Calculates the total thickness of a 2d tibial cartilage layer.

    :param max_vectors: Maximum z values of the cartilage data points
    :param min_vectors: Minimum z values of the cartilage data points
    :param normals: the normals along which to calculate the thickness
    :param layer_index: the y index of the current layer
    :param left_regions: corners a, b, c, d as well as radius and center of gravity of the left plate
    :param right_regions: corners a, b, c, d as well as radius and center of gravity of the right plate
    :param split_vector: the vector with which to split into left and right plate
    :return: total thickness, number of normals
    """
    logging.debug(f'BEGIN {inspect.currentframe().f_code.co_name}')

    average_thickness = dict()
    normal_count = dict()

    average_thickness['cLT'] = 0
    average_thickness['aLT'] = 0
    average_thickness['eLT'] = 0
    average_thickness['pLT'] = 0
    average_thickness['iLT'] = 0
    average_thickness['cMT'] = 0
    average_thickness['aMT'] = 0
    average_thickness['eMT'] = 0
    average_thickness['pMT'] = 0
    average_thickness['iMT'] = 0

    normal_count['cLT'] = 0
    normal_count['aLT'] = 0
    normal_count['eLT'] = 0
    normal_count['pLT'] = 0
    normal_count['iLT'] = 0
    normal_count['cMT'] = 0
    normal_count['aMT'] = 0
    normal_count['eMT'] = 0
    normal_count['pMT'] = 0
    normal_count['iMT'] = 0

    fail_count = 0
    x_val = 0

    for norm_vec in normals:
        try:
            x0 = norm_vec[0]
            norm = norm_vec[1]
            norm = norm[norm > 0]
            # set comparison context for current normal: x +- 1
            min_ext = min_vectors[x0 - 1:x0 + 2]
            max_ext = max_vectors[x0 - 1:x0 + 2]

            # find the closest minimum and maximum points to the normal, i.e. lowest and highest points where the normal
            # cuts the cartilage
            min_y_indices = np.argwhere([abs(norm - x) < 0.1 for x in min_ext])
            max_y_indices = np.argwhere([abs(norm - x) < 0.1 for x in max_ext])

            min_y_indices = [x[1] for x in min_y_indices]
            max_y_indices = [x[1] for x in max_y_indices]
            # get the x and y values for the maximum intersection point
            max_y = norm[max_y_indices].max()
            max_x = np.where(norm == max_y)[0][0]

            # get the x and y values for the minimum intersection point
            min_y = norm[min_y_indices].min()
            min_x = np.where(norm == min_y)[0][0]

            # assert that the right values were fetched
            assert max_y == norm[max_x]
            assert min_y == norm[min_x]

            max_x /= 100
            min_x /= 100

            x = np.array([max_x, max_y])
            y = np.array([min_x, min_y])

            # get the subregion of the current normal
            tmp = tuple([x_val, layer_index])
            label = utility.classify_tibial_point(tmp, left_regions, right_regions, split_vector)

            # calculate distance between the two vectors
            vec_dist = utility.vector_distance(x, y) * sitk_image.GetSpacing()[2]
            if not vec_dist > 0:
                raise ValueError

            average_thickness[label] += vec_dist
            normal_count[label] += 1
            x_val += 1
        except (ValueError, AssertionError, KeyError):
            fail_count += 1
            x_val += 1
            continue

    logging.debug(f'END {inspect.currentframe().f_code.co_name}')
    return [average_thickness, normal_count]


def calculate_normals(x, df: pd.DataFrame, fun) -> list:
    """
    Calculates all normal vectors for a single 2d cartilage layer.

    :param x: the x values for which to calculate the vectors at
    :param df: the dataframe representation of the layer
    :param fun: the function of which to calculate the normal vectors
    :return: the normal vectors of function fun at x
    """
    logging.debug(f'BEGIN {inspect.currentframe().f_code.co_name}')

    normals = list()

    for i in range(df.groupby(df['x']).max().shape[0]):
        normals.append(normal(x, i, fun))

    logging.debug(f'END {inspect.currentframe().f_code.co_name}')
    return normals


def normal(x, x0, fun):
    """
    Calculates the normal of a function at x0.

    :param x: x values
    :param x0: x value where to calculate the normal
    :param fun: the function of which to calculate the normal
    :return: the normal of function fun at x0
    """
    logging.debug(f'BEGIN {inspect.currentframe().f_code.co_name}')

    return [x0, fun(x0) - (1 / derivative(x0, fun)) * (x - x0)]


def derivative(x0, fun):
    """
    Calculates the derivative of a function at x0.

    :param x0: x value where to calculate the derivative
    :param fun: the function of which to calculate the derivative
    :return: the derivative of function fun at x0
    """
    logging.debug(f'BEGIN {inspect.currentframe().f_code.co_name}')

    h = 10**(-8)
    logging.debug(f'END {inspect.currentframe().f_code.co_name}')
    return (fun(x0 + h) - fun(x0)) / h


def helper(directory):
    logging.debug(f'BEGIN {inspect.currentframe().f_code.co_name}')

    sitk_image, np_image = utility.read_image(f'/images/Shape/Medical/Knees/OAI/Manual_Segmentations/{directory}/{directory}_segm.mhd')
    tib_res = average_tibial_thickness_per_region(np_image, sitk_image)
    fem_res = average_femoral_thickness_per_region(np_image, sitk_image)

    keys = set(tib_res[0][0].keys())
    tib_d = dict()
    for key in keys:
        tib_d[key] = np.array([x[key] for x in tib_res[0]], dtype='float') / np.array([x[key] for x in tib_res[1]],
                                                                                      dtype='float')
        value = tib_d[key]
        mask = value == 0
        # value[mask] = np.nan
        tib_d[key + '.aSD'] = np.nanstd(value)
        tib_d[key + '.aMav'] = np.nanmean(-np.sort(-value)[:math.ceil(len(value) * 0.01)])
        tib_d[key + '.aMiv'] = np.nanmean(np.sort(value)[:math.ceil(len(value) * 0.01)])
        tib_d[key] = np.nanmean(value)

    keys = set(fem_res[0][0].keys())
    fem_d = dict()
    for key in keys:
        fem_d[key] = np.array([x[key] for x in fem_res[0]], dtype='float') / np.array([x[key] for x in fem_res[1]],
                                                                                      dtype='float')
        value = fem_d[key]
        mask = value == 0
        # value[mask] = np.nan
        fem_d[key + '.aSD'] = np.nanstd(value)
        fem_d[key + '.aMav'] = np.nanmean(-np.sort(-value)[:math.ceil(len(value) * 0.01)])
        fem_d[key + '.aMiv'] = np.nanmean(np.sort(value)[:math.ceil(len(value) * 0.01)])
        fem_d[key] = np.nanmean(value)

    logging.debug(f'END {inspect.currentframe().f_code.co_name}')
    return {**{'dir': directory}, **fem_d, **tib_d}


def main():
    logging.basicConfig(filename='/work/scratch/westfechtel/pylogs/normals/2d_default.log', encoding='utf-8', level=logging.DEBUG, filemode='w')
    logging.debug('Entered main.')

    try:
        assert len(sys.argv) == 2
        chunk = np.load(f'/work/scratch/westfechtel/chunks/{sys.argv[1]}.npy')

        filehandler = logging.FileHandler(f'/work/scratch/westfechtel/pylogs/normals/{sys.argv[1]}.log', mode='w')
        filehandler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        root.addHandler(filehandler)
        logging.debug(f'Using chunk {sys.argv[1]} with length {len(chunk)}.')

        dirs = utility.get_subdirs(chunk)
        res = np.empty(len(dirs), dtype='object')
        for i, directory in enumerate(dirs):
            try:
                if i % 10 == 0:
                    logging.debug(f'Iteration #{i}')
                res[i] = helper(directory)
            except Exception:
                continue

        res = res[res != None]
        res = list(res)
        df = pd.DataFrame.from_dict(res)
        df.index = df['dir']
        df = df.drop('dir', axis=1)
        # df.to_excel('mesh.xlsx')
        df.to_pickle(f'/work/scratch/westfechtel/pickles/2d/{sys.argv[1]}')
    except Exception as e:
        logging.debug(traceback.format_exc())
        logging.debug(sys.argv)


if __name__ == '__main__':
    main()
