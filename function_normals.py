import logging
import math
import pprint
import sys
import traceback
from collections import defaultdict
from concurrent.futures import TimeoutError
from time import time

import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as poly
from pebble import ProcessPool
from pebble.common import ProcessExpired

import utility


def build_cwbz_layers(df):
    """
    Builds two-dimensional (y, z) layers in x-direction for the central weight-bearing zones of the femoral cartilage.

    :param df: A pandas dataframe representation of the central weight-bearing zone
    :return: A list containing the x coordinates of the layers, and a numpy array containing the layers
    """
    layers = np.zeros(df.nunique()['x'], dtype='object')
    xs = sorted(df['x'].unique())
    for i in range(len(layers)):
        layers[i] = df.loc[df['x'] == xs[i]][[
            'y', 'z']].sort_values(by='y').to_numpy()

    return xs, layers


def build_peripheral_layers(df):
    """
    Builds two-dimensional (x, z) layers in y-direction for the peripheral zones of the femoral cartilage.

    :param df: A pandas dataframe representation of the peripheral zone
    :return: A list containing the y coordinates of the layers, and a numpy array containing the layers
    """
    layers = np.zeros(df.nunique()['y'], dtype='object')
    ys = sorted(df['y'].unique())
    for i in range(len(layers)):
        layers[i] = df.loc[df['y'] == ys[i]][[
            'x', 'z']].sort_values(by='z').to_numpy()

    return ys, layers


def trace(p, v, tolerance, df):
    """
    Uses vector tracing to find the outermost intersection point in a search space.

    :param p: The origin point to trace from
    :param v: The direction vector
    :param tolerance: The tolerance for the intersection search
    :param df: A pandas dataframe representation of the search space

    :return: The outermost intersection point
    """
    point = None
    alpha = 1
    while True:
        g = p + alpha * v
        points = df.loc[abs(df['x'] - g[0]) <=
                        tolerance].loc[abs(df['y'] - g[1]) <= tolerance]
        if points.shape[0] == 0:
            break

        point = points.iloc[0].to_numpy()
        alpha += 1

    return point


def calculate_region_thickness(sitk_image, layers, dictionary, xs, left_landmarks, right_landmarks, cwbz=True,
                               left=True, label=None, tibia=False, split_vector=None, pdf=False):
    """
    Calculates the mean thickness per region for all layers of a cartilage.

    Fits a function through the middle of each layer and computes a number of function normals. For each normal, traces from the intercept in negative and positive 
    direction of the slope to find the outermost intersection points of the layer with the respective normal. For each pair of intersection points, calculates 
    their distance from each other and assigns the result to the corresponding subregion.

    :param layers: A numpy array containing all two-dimensional layers making up the cartilage (or part thereof)
    :param dictionary: A dictionary mapping thickness values to region labels
    :param xs: A list containing the x or y coordinates of the layers
    :param left_landmarks: Landmarks of the left plate, for regional classification
    :param right_landmarks: Landmarks of the right plate, for regional classification
    :param cwbz: Whether the layers belong to a central weight-bearing zone
    :param left: Whether the layers belong to the left or right plate
    :param label: Label to use for classification if layers belong to peripheral femoral region
    :param tibia: Whether the layers belong to the tibial cartilage
    :param split_vector: Vector splitting the tibial cartilage into left and right plate

    :return: Updated input dictionary with calculated thickness values
    """
    for layer_index, layer in enumerate(layers):
        if cwbz or tibia or pdf:
            x = np.array([x[0] for x in layer])
            y = np.array([x[1] for x in layer])
        else:
            # important to swap x and y here or we do not get a good fit
            x = np.array([x[1] for x in layer])
            y = np.array([x[0] for x in layer])

        try:
            z = poly.polyfit(x, y, 3)
            der = poly.polyder(z)
        except np.linalg.LinAlgError as e:
            logging.error(traceback.format_exc())
            logging.warning(f'Got error while trying to fit function. Return empty dict.')
            return dict()

        fun = poly.polyval(x, z)
        new_x = np.arange(min(x), max(x))
        normals = [poly.polyval(
            val, z) - (1 / poly.polyval(val, der)) * (new_x - val) for val in new_x]

        if len(normals) == 0:
            continue

        if cwbz or tibia:
            df = pd.DataFrame(layer, columns=['x', 'y'])
        else:
            # also swap x and y for search space or we don't get hits!
            df = pd.DataFrame({'x': x, 'y': y})

        outline_points = np.zeros(len(normals), dtype=object)

        for i in range(len(normals)):
            x0 = new_x[i]
            normal = normals[i]

            if len(normal) < 2:
                outline_points[i] = np.array([None, None])
                continue

            intercept = normal[i]
            slope = -(normal[0] - normal[1])
            p = np.array([x0, intercept])
            v1 = np.array([-1, -slope]) * .1
            v2 = np.array([1, slope]) * .1
            point_1 = None
            point_2 = None
            for j in range(10):
                point_1 = trace(p, v1, j, df)
                if point_1 is not None:
                    break

            t = time()
            if point_1 is not None:
                for j in range(10):
                    point_2 = trace(p, v2, j, df)
                    if point_2 is not None:
                        break

            if point_1 is None or point_2 is None:
                point_1 = None
                point_2 = None

            outline_points[i] = np.array([point_1, point_2])

        indices = []
        for i in range(len(outline_points)):
            if None in outline_points[i]:
                indices.append(i)

        outline_points = np.delete(outline_points, indices)

        layer_thickness = dict()
        if cwbz:
            if left:
                layer_thickness['ecLF'] = np.zeros(len(outline_points))
                layer_thickness['ccLF'] = np.zeros(len(outline_points))
                layer_thickness['icLF'] = np.zeros(len(outline_points))

                for i, point in enumerate(outline_points):
                    label = utility.classify_femoral_point(
                        np.array([xs[layer_index], point[0][0]]), left_landmarks, left=True)
                    layer_thickness[label][i] = utility.vector_distance(
                        point[0], point[1]) * sitk_image.GetSpacing()[1]

                keys = set(layer_thickness.keys())
                for key in keys:
                    dictionary[key] = np.hstack(
                        (dictionary[key], layer_thickness[key]))
            else:
                layer_thickness['ecMF'] = np.zeros(len(outline_points))
                layer_thickness['ccMF'] = np.zeros(len(outline_points))
                layer_thickness['icMF'] = np.zeros(len(outline_points))

                for i, point in enumerate(outline_points):
                    label = utility.classify_femoral_point(
                        np.array([xs[layer_index], point[0][0]]), right_landmarks, left=False)
                    layer_thickness[label][i] = utility.vector_distance(
                        point[0], point[1]) * sitk_image.GetSpacing()[1]

                keys = set(layer_thickness.keys())
                for key in keys:
                    dictionary[key] = np.hstack(
                        (dictionary[key], layer_thickness[key]))
        elif not cwbz and not tibia:
            layer_thickness[label] = np.zeros(len(outline_points))
            for i, point in enumerate(outline_points):
                layer_thickness[label][i] = utility.vector_distance(
                    point[0], point[1]) * sitk_image.GetSpacing()[1]

            keys = set(layer_thickness.keys())
            for key in keys:
                dictionary[key] = np.hstack(
                    (dictionary[key], layer_thickness[key]))
        else:
            if left:
                layer_thickness['eLT'] = np.zeros(len(outline_points))
                layer_thickness['pLT'] = np.zeros(len(outline_points))
                layer_thickness['iLT'] = np.zeros(len(outline_points))
                layer_thickness['aLT'] = np.zeros(len(outline_points))
                layer_thickness['cLT'] = np.zeros(len(outline_points))

                fails = 0
                for i, point in enumerate(outline_points):
                    label = utility.classify_tibial_point(np.array(
                        [xs[layer_index], point[0][0]]), left_landmarks, right_landmarks, split_vector)
                    if label in set(['cMT', 'aMT', 'eMT', 'pMT', 'iMT']):
                        fails += 1
                        continue
                    layer_thickness[label][i] = utility.vector_distance(
                        point[0], point[1]) * sitk_image.GetSpacing()[1]

                logging.info(f'{fails} failed classifications (of {i})')
                keys = set(layer_thickness.keys())
                for key in keys:
                    dictionary[key] = np.hstack(
                        (dictionary[key], layer_thickness[key]))
            else:
                layer_thickness['eMT'] = np.zeros(len(outline_points))
                layer_thickness['pMT'] = np.zeros(len(outline_points))
                layer_thickness['iMT'] = np.zeros(len(outline_points))
                layer_thickness['aMT'] = np.zeros(len(outline_points))
                layer_thickness['cMT'] = np.zeros(len(outline_points))

                fails = 0
                for i, point in enumerate(outline_points):
                    label = utility.classify_tibial_point(np.array(
                        [xs[layer_index], point[0][0]]), left_landmarks, right_landmarks, split_vector)
                    if label in set(['cLT', 'aLT', 'eLT', 'pLT', 'iLT']):
                        fails += 1           
                        continue
                    layer_thickness[label][i] = utility.vector_distance(
                        point[0], point[1]) * sitk_image.GetSpacing()[1]

                logging.info(f'{fails} failed classifications (of {i})')
                keys = set(layer_thickness.keys())
                for key in keys:
                    dictionary[key] = np.hstack(
                        (dictionary[key], layer_thickness[key]))

    return dictionary


def function_for_pool(directory):
    segmentation_directory = f'/images/Shape/Medical/Knees/OAI/Manual_Segmentations/{directory}/{directory}_segm.mhd'
    # segmentation_directory = f'/work/scratch/westfechtel/segmentations/{directory}'
    # segmentation_directory = '9144760_segm.mhd'
    sitk_image, np_image = utility.read_image(segmentation_directory)

    femoral_cartilage = utility.build_3d_cartilage_array(np_image, 3)
    tibial_cartilage = utility.build_3d_cartilage_array(np_image, 4)

    femoral_vectors = [list(element) for element in femoral_cartilage]
    tibial_vectors = [list(element) for element in tibial_cartilage]

    cwbzl, cwbzr = utility.extract_central_weightbearing_zone(
        femoral_vectors, tibial_vectors)
    lpdf, rpdf, adf = utility.extract_anterior_posterior_zones(
        femoral_vectors, cwbzl, cwbzr)

    ladf, radf = utility.split_anterior_part(adf)

    lower_mesh_left, upper_mesh_left = utility.build_femoral_meshes(cwbzl)
    lower_mesh_right, upper_mesh_right = utility.build_femoral_meshes(cwbzr)

    left_landmarks = utility.femoral_landmarks(upper_mesh_left.points)
    right_landmarks = utility.femoral_landmarks(upper_mesh_right.points)

    total_thickness = defaultdict()
    total_thickness['ecLF'] = np.zeros(1)
    total_thickness['ccLF'] = np.zeros(1)
    total_thickness['icLF'] = np.zeros(1)
    total_thickness['ecMF'] = np.zeros(1)
    total_thickness['ccMF'] = np.zeros(1)
    total_thickness['icMF'] = np.zeros(1)
    total_thickness['pLF'] = np.zeros(1)
    total_thickness['pMF'] = np.zeros(1)
    total_thickness['aLF'] = np.zeros(1)
    total_thickness['aMF'] = np.zeros(1)

    total_thickness['eLT'] = np.zeros(1)
    total_thickness['pLT'] = np.zeros(1)
    total_thickness['iLT'] = np.zeros(1)
    total_thickness['aLT'] = np.zeros(1)
    total_thickness['cLT'] = np.zeros(1)
    total_thickness['eMT'] = np.zeros(1)
    total_thickness['pMT'] = np.zeros(1)
    total_thickness['iMT'] = np.zeros(1)
    total_thickness['aMT'] = np.zeros(1)
    total_thickness['cMT'] = np.zeros(1)

    t = time()
    xs, layers = build_cwbz_layers(cwbzl)
    total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                 xs=xs, left_landmarks=left_landmarks,
                                                 right_landmarks=right_landmarks, cwbz=True, left=True, label=None,
                                                 tibia=False, split_vector=None)
    logging.info(f'{directory} finished calculations for cwbzl in {time() - t} seconds.')

    t = time()
    xs, layers = build_cwbz_layers(cwbzr)
    total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                 xs=xs, left_landmarks=left_landmarks,
                                                 right_landmarks=right_landmarks, cwbz=True, left=False, label=None,
                                                 tibia=False, split_vector=None)
    logging.info(f'{directory} finished calculations for cwbzr in {time() - t} seconds.')

    t = time()
    xs, layers = build_peripheral_layers(lpdf)
    total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                 xs=xs, left_landmarks=left_landmarks,
                                                 right_landmarks=right_landmarks, cwbz=False, left=False, label='pLF',
                                                 tibia=False, split_vector=None, pdf=True)
    logging.info(f'{directory} finished calculations for lpdf in {time() - t} seconds.')

    t = time()
    xs, layers = build_peripheral_layers(rpdf)
    total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                 xs=xs, left_landmarks=left_landmarks,
                                                 right_landmarks=right_landmarks, cwbz=False, left=False, label='pMF',
                                                 tibia=False, split_vector=None, pdf=True)
    logging.info(f'{directory} finished calculations for rpdf in {time() - t} seconds.')

    t = time()
    xs, layers = build_peripheral_layers(ladf)
    total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                 xs=xs, left_landmarks=left_landmarks,
                                                 right_landmarks=right_landmarks, cwbz=False, left=False, label='aLF',
                                                 tibia=False, split_vector=None)

    xs, layers = build_peripheral_layers(radf)
    total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                 xs=xs, left_landmarks=left_landmarks,
                                                 right_landmarks=right_landmarks, cwbz=False, left=False, label='aMF',
                                                 tibia=False, split_vector=None)
    logging.info(f'{directory} finished calculations for adf in {time() - t} seconds.')

    # lower_mesh, upper_mesh = utility.build_tibial_meshes(tibial_vectors)
    # left_landmarks, right_landmarks, split_vector = utility.tibial_landmarks(
    #     lower_mesh.points)
    df = pd.DataFrame(data=tibial_cartilage, columns=['x', 'y', 'z'])
    max_z = df.groupby(['x', 'y']).max()

    tmp1 = [np.array(item) for item in max_z.index]
    tmp2 = [item for item in max_z.to_numpy()]
    max_z = np.column_stack((tmp1, tmp2))

    left_landmarks, right_landmarks, split_vector = utility.tibial_landmarks(max_z)

    x, y, z, xy = utility.get_xyz(tibial_vectors)
    left_plate, right_plate = utility.split_into_plates(
        tibial_vectors, [0, np.mean(y)])

    ldf = pd.DataFrame(data=left_plate, columns=['x', 'y', 'z'])

    rdf = pd.DataFrame(data=right_plate, columns=['x', 'y', 'z'])

    try:
        t = time()
        xs, layers = build_cwbz_layers(ldf)
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=False, left=True, label=None,
                                                     tibia=True, split_vector=split_vector)
        logging.info(f'{directory} finished calculations for ldf in {time() - t} seconds.')

        t = time()
        xs, layers = build_cwbz_layers(rdf)
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=False, left=False,
                                                     label=None, tibia=True, split_vector=split_vector)
        logging.info(f'{directory} finished calculations for rdf in {time() - t} seconds.')
    except Exception:
        logging.error(traceback.format_exc())
        return dict()

    keys = set(total_thickness.keys())
    for key in keys:
        value = total_thickness[key]
        mask = value == 0
        value[mask] = np.nan
        total_thickness[key + '.aSD'] = np.nanstd(value)
        total_thickness[key +
                        '.aMav'] = np.nanmean(-np.sort(-value)[:math.ceil(len(value) * 0.01)])
        total_thickness[key + '.aMiv'] = np.nanmean(
            np.sort(value)[:math.ceil(len(value) * 0.01)])
        total_thickness[key] = np.nanmean(value)

    return {**{'dir': directory}, **total_thickness}


def main():
    logging.basicConfig(filename='/work/scratch/westfechtel/pylogs/function_normals/function_normals_default.log',
                        encoding='utf-8', level=logging.DEBUG, filemode='w')
    logging.debug('Entered main.')

    try:
        assert len(sys.argv) == 2
        chunk = np.load(f'/work/scratch/westfechtel/chunks/{sys.argv[1]}.npy')
        # chunk = sys.argv[1]

        filehandler = logging.FileHandler(
            f'/work/scratch/westfechtel/pylogs/function_normals/{sys.argv[1]}.log', mode='w')
        filehandler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        root.addHandler(filehandler)

        files = utility.get_subdirs(chunk)

        # debug !!
        # files = files[-100:-1]

        logging.info(f'Using chunk {sys.argv[1]} with length {len(files)}.')

        res_list = list()
        t = time()
        with ProcessPool() as pool:
            res = pool.map(function_for_pool, files, timeout=3600)
            # res = pool.map(function=function_for_pool, iterables=files, chunksize=int(len(files)/8), timeout=180)
            # pool.close()
            # pool.terminate()

            iterator = res.result()
            while True:
                try:
                    tmp = next(iterator)
                    res_list.append(tmp)
                    # logging.info(f'Adding {tmp} to result list.')
                except TimeoutError:
                    logging.error('Timeout error.')
                    continue
                except StopIteration:
                    logging.info('End of iterator.')
                    break
                except ProcessExpired as exp:
                    logging.error(exp)
                    continue

        logging.info(f'Elapsed time: {time() - t}')
        # df = pd.DataFrame.from_dict(res)
        df = pd.DataFrame.from_dict(res_list)
        df.index = df['dir']
        df = df.drop('dir', axis=1)
        df.to_pickle(
            f'/work/scratch/westfechtel/manpickles/function_normals/{sys.argv[1]}')
    except Exception as e:
        logging.debug(traceback.format_exc())
        logging.debug(sys.argv)


def test():
    logging.basicConfig(filename='function_normals_default.log',
                        encoding='utf-8', level=logging.DEBUG, filemode='w')
    t = time()
    pprint.PrettyPrinter().pprint(function_for_pool(1))
    print(time() - t)


if __name__ == '__main__':
    main()
    # test()
