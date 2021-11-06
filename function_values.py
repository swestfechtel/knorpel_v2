import logging
import math
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

import function_normals
import utility


def calculate_region_thickness(sitk_image, layers, dictionary, xs, left_landmarks, right_landmarks, cwbz=True,
                               left=True, label=None, tibia=False, split_vector=None):
    """
    Calculates the mean thickness per region for all layers of a cartilage.

    Fits two functions through the upper and lower outlines of each layer. For each layer, computes the respective function values for each value
    along the x axis, the difference of the two function values being the the thickness measure at that point. Assigns the results to the corresponding subregions.

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
        if cwbz or tibia:
            x = np.array([x[0] for x in layer])
            y = np.array([x[1] for x in layer])
        else:
            # important to swap x and y here or we do not get a good fit
            x = np.array([x[1] for x in layer])
            y = np.array([x[0] for x in layer])

        if cwbz or tibia:
            df = pd.DataFrame(layer, columns=['x', 'y'])
        else:
            # also swap x and y for search space or we don't get hits!
            df = pd.DataFrame({'x': x, 'y': y})

        upper_points = df.groupby(by='x').max().reset_index()
        lower_points = df.groupby(by='x').min().reset_index()
        try:
            upper_fit = poly.polyfit(upper_points['x'], upper_points['y'], 3)
            lower_fit = poly.polyfit(lower_points['x'], lower_points['y'], 3)
        except np.linalg.LinAlgError as e:
            logging.error(traceback.format_exc())
            logging.warning(f'Got error while trying to fit function. Return empty dict.')
            return dict()

        upper_fun = poly.polyval(upper_points['x'], upper_fit)
        lower_fun = poly.polyval(lower_points['x'], lower_fit)

        layer_thickness = dict()
        if cwbz:
            if left:
                layer_thickness['ecLF'] = np.zeros(len(x))
                layer_thickness['ccLF'] = np.zeros(len(x))
                layer_thickness['icLF'] = np.zeros(len(x))

                for i, val in enumerate(x):
                    label = utility.classify_femoral_point(
                        np.array([xs[layer_index], val]), left_landmarks, left=True)
                    layer_thickness[label][i] = (poly.polyval(val, upper_fit) - poly.polyval(val, lower_fit)) * \
                                                sitk_image.GetSpacing()[1]

                keys = set(layer_thickness.keys())
                for key in keys:
                    dictionary[key] = np.hstack(
                        (dictionary[key], layer_thickness[key]))
            else:
                layer_thickness['ecMF'] = np.zeros(len(x))
                layer_thickness['ccMF'] = np.zeros(len(x))
                layer_thickness['icMF'] = np.zeros(len(x))

                for i, val in enumerate(x):
                    label = utility.classify_femoral_point(
                        np.array([xs[layer_index], val]), right_landmarks, left=False)
                    layer_thickness[label][i] = (poly.polyval(val, upper_fit) - poly.polyval(val, lower_fit)) * \
                                                sitk_image.GetSpacing()[1]

                keys = set(layer_thickness.keys())
                for key in keys:
                    dictionary[key] = np.hstack(
                        (dictionary[key], layer_thickness[key]))
        elif not cwbz and not tibia:
            layer_thickness[label] = np.zeros(len(x))
            for i, val in enumerate(x):
                layer_thickness[label][i] = (poly.polyval(val, upper_fit) - poly.polyval(val, lower_fit)) * \
                                            sitk_image.GetSpacing()[1]

            keys = set(layer_thickness.keys())
            for key in keys:
                dictionary[key] = np.hstack(
                    (dictionary[key], layer_thickness[key]))
        else:
            if left:
                layer_thickness['eLT'] = np.zeros(len(x))
                layer_thickness['pLT'] = np.zeros(len(x))
                layer_thickness['iLT'] = np.zeros(len(x))
                layer_thickness['aLT'] = np.zeros(len(x))
                layer_thickness['cLT'] = np.zeros(len(x))

                for i, val in enumerate(x):
                    label = utility.classify_tibial_point(np.array(
                        [xs[layer_index], val]), left_landmarks, right_landmarks, split_vector)
                    layer_thickness[label][i] = (poly.polyval(val, upper_fit) - poly.polyval(val, lower_fit)) * \
                                                sitk_image.GetSpacing()[1]

                keys = set(layer_thickness.keys())
                for key in keys:
                    dictionary[key] = np.hstack(
                        (dictionary[key], layer_thickness[key]))
            else:
                layer_thickness['eMT'] = np.zeros(len(x))
                layer_thickness['pMT'] = np.zeros(len(x))
                layer_thickness['iMT'] = np.zeros(len(x))
                layer_thickness['aMT'] = np.zeros(len(x))
                layer_thickness['cMT'] = np.zeros(len(x))

                for i, val in enumerate(x):
                    label = utility.classify_tibial_point(np.array(
                        [xs[layer_index], val]), left_landmarks, right_landmarks, split_vector)
                    layer_thickness[label][i] = (poly.polyval(val, upper_fit) - poly.polyval(val, lower_fit)) * \
                                                sitk_image.GetSpacing()[1]

                keys = set(layer_thickness.keys())
                for key in keys:
                    dictionary[key] = np.hstack(
                        (dictionary[key], layer_thickness[key]))

    return dictionary


def function_for_pool(directory):
    segmentation_directory = f'/images/Shape/Medical/Knees/OAI/Manual_Segmentations/{directory}/{directory}_segm.mhd'
    # segmentation_directory = f'/work/scratch/westfechtel/segmentations/{directory}'
    # segmentation_directory = '9255535_segm.mhd'
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

    try:
        xs, layers = function_normals.build_cwbz_layers(cwbzl)
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=True, left=True, label=None,
                                                     tibia=False, split_vector=None)

        xs, layers = function_normals.build_cwbz_layers(cwbzr)
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=True, left=False, label=None,
                                                     tibia=False, split_vector=None)

        xs, layers = function_normals.build_peripheral_layers(lpdf)
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=False, left=False,
                                                     label='pLF', tibia=False, split_vector=None)

        xs, layers = function_normals.build_peripheral_layers(rpdf)
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=False, left=False,
                                                     label='pMF', tibia=False, split_vector=None)

        xs, layers = function_normals.build_peripheral_layers(ladf)
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=False, left=False,
                                                     label='aLF', tibia=False, split_vector=None)

        xs, layers = function_normals.build_peripheral_layers(radf)
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=False, left=False,
                                                     label='aMF', tibia=False, split_vector=None)
    except Exception:
        logging.error(traceback.format_exc())
        return dict()

    lower_mesh, upper_mesh = utility.build_tibial_meshes(tibial_vectors)
    left_landmarks, right_landmarks, split_vector = utility.tibial_landmarks(
        lower_mesh.points)

    x, y, z, xy = utility.get_xyz(tibial_vectors)
    left_plate, right_plate = utility.split_into_plates(
        tibial_vectors, [0, np.mean(y)])

    ldf = pd.DataFrame(data=left_plate, columns=['x', 'y', 'z'])

    rdf = pd.DataFrame(data=right_plate, columns=['x', 'y', 'z'])

    try:
        xs, layers = function_normals.build_cwbz_layers(ldf)
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=False, left=True, label=None,
                                                     tibia=True, split_vector=split_vector)

        xs, layers = function_normals.build_cwbz_layers(rdf)
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=False, left=False,
                                                     label=None, tibia=True, split_vector=split_vector)
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
    logging.basicConfig(filename='/work/scratch/westfechtel/pylogs/function_values/function_values_default.log',
                        encoding='utf-8', level=logging.DEBUG, filemode='w')
    logging.debug('Entered main.')

    try:
        assert len(sys.argv) == 2
        chunk = np.load(f'/work/scratch/westfechtel/chunks/{sys.argv[1]}.npy')
        # chunk = sys.argv[1]

        filehandler = logging.FileHandler(
            f'/work/scratch/westfechtel/pylogs/function_values/{sys.argv[1]}.log', mode='w')
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
            res = pool.map(function_for_pool, files, timeout=1200)
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
            f'/work/scratch/westfechtel/manpickles/function_values/{sys.argv[1]}')
    except Exception as e:
        logging.debug(traceback.format_exc())
        logging.debug(sys.argv)


if __name__ == '__main__':
    main()
