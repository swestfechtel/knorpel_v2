import logging
import math
import sys
import traceback
import function_normals
import utility
import time
import multiprocessing

import numpy as np
import pandas as pd

from collections import defaultdict
from concurrent.futures import TimeoutError
from time import time
from numpy.polynomial import polynomial as poly
from pebble import ProcessPool
from pebble.common import ProcessExpired
from scipy.optimize import fsolve
from logging.handlers import QueueHandler, QueueListener


def calculate_region_thickness(sitk_image, layers, dictionary, xs, left_landmarks, right_landmarks, cwbz=True,
                               left=True, label=None, tibia=False, split_vector=None, af=False, pdf=False):
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
        if cwbz or tibia or af or pdf:
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

        der = poly.polyder(lower_fit)
        # new_x = np.arange(min(lower_points['x']), max(lower_points['x']), step=.01)
        new_x = lower_points['x'].to_numpy()

        layer_thickness = dict()
        if cwbz:
            if left:
                layer_thickness['ecLF'] = np.zeros(len(new_x))
                layer_thickness['ccLF'] = np.zeros(len(new_x))
                layer_thickness['icLF'] = np.zeros(len(new_x))

                for i, val in enumerate(new_x):
                    label = utility.classify_femoral_point(
                        np.array([xs[layer_index], val]), left_landmarks, left=True)
                    der = poly.polyder(lower_fit)
                    der = poly.polyder(lower_fit)
                    normal = poly.polyval(val, lower_fit) - (1 / poly.polyval(val, der)) * (new_x - val)
                    idx = np.argwhere(np.diff(np.sign(normal - poly.polyval(new_x, upper_fit)))).flatten()
                    lower_intersection_x = new_x[i]
                    lower_intersection_y = poly.polyval(lower_intersection_x, lower_fit)
                    upper_intersection_x = new_x[idx]
                    if len(upper_intersection_x) == 0:
                        continue
                    upper_intersection_x = upper_intersection_x[0]
                    upper_intersection_y = poly.polyval(upper_intersection_x, upper_fit)

                    layer_thickness[label][i] = utility.vector_distance(np.array([lower_intersection_x, lower_intersection_y]), np.array([upper_intersection_x, upper_intersection_y])) * \
                                                sitk_image.GetSpacing()[1]

                keys = set(layer_thickness.keys())
                for key in keys:
                    dictionary[key] = np.hstack(
                        (dictionary[key], layer_thickness[key]))
            else:
                layer_thickness['ecMF'] = np.zeros(len(new_x))
                layer_thickness['ccMF'] = np.zeros(len(new_x))
                layer_thickness['icMF'] = np.zeros(len(new_x))

                for i, val in enumerate(new_x):
                    label = utility.classify_femoral_point(
                        np.array([xs[layer_index], val]), right_landmarks, left=False)
                    der = poly.polyder(lower_fit)
                    normal = poly.polyval(val, lower_fit) - (1 / poly.polyval(val, der)) * (new_x - val)
                    idx = np.argwhere(np.diff(np.sign(normal - poly.polyval(new_x, upper_fit)))).flatten()
                    lower_intersection_x = new_x[i]
                    lower_intersection_y = poly.polyval(lower_intersection_x, lower_fit)
                    upper_intersection_x = new_x[idx]
                    if len(upper_intersection_x) == 0:
                        continue
                    upper_intersection_x = upper_intersection_x[0]
                    upper_intersection_y = poly.polyval(upper_intersection_x, upper_fit)

                    layer_thickness[label][i] = utility.vector_distance(np.array([lower_intersection_x, lower_intersection_y]), np.array([upper_intersection_x, upper_intersection_y])) * \
                                                sitk_image.GetSpacing()[1]

                keys = set(layer_thickness.keys())
                for key in keys:
                    dictionary[key] = np.hstack(
                        (dictionary[key], layer_thickness[key]))
        elif not cwbz and not tibia:
            layer_thickness[label] = np.zeros(len(new_x))
            for i, val in enumerate(new_x):
                der = poly.polyder(lower_fit)
                normal = poly.polyval(val, lower_fit) - (1 / poly.polyval(val, der)) * (new_x - val)
                idx = np.argwhere(np.diff(np.sign(normal - poly.polyval(new_x, upper_fit)))).flatten()
                lower_intersection_x = new_x[i]
                lower_intersection_y = poly.polyval(lower_intersection_x, lower_fit)
                upper_intersection_x = new_x[idx]
                if len(upper_intersection_x) == 0:
                    continue
                upper_intersection_x = upper_intersection_x[0]
                upper_intersection_y = poly.polyval(upper_intersection_x, upper_fit)

                layer_thickness[label][i] = utility.vector_distance(np.array([lower_intersection_x, lower_intersection_y]), np.array([upper_intersection_x, upper_intersection_y])) * \
                                            sitk_image.GetSpacing()[1]

            keys = set(layer_thickness.keys())
            for key in keys:
                dictionary[key] = np.hstack(
                    (dictionary[key], layer_thickness[key]))
        else:
            if left:
                layer_thickness['eLT'] = np.zeros(len(new_x))
                layer_thickness['pLT'] = np.zeros(len(new_x))
                layer_thickness['iLT'] = np.zeros(len(new_x))
                layer_thickness['aLT'] = np.zeros(len(new_x))
                layer_thickness['cLT'] = np.zeros(len(new_x))

                fails = 0
                for i, val in enumerate(new_x):
                    label = utility.classify_tibial_point(np.array(
                        [xs[layer_index], val]), left_landmarks, right_landmarks, split_vector)
                    if label in set(['cMT', 'aMT', 'eMT', 'pMT', 'iMT']):
                        fails += 1
                        continue
                    der = poly.polyder(lower_fit)
                    normal = poly.polyval(val, lower_fit) - (1 / poly.polyval(val, der)) * (new_x - val)
                    idx = np.argwhere(np.diff(np.sign(normal - poly.polyval(new_x, upper_fit)))).flatten()
                    lower_intersection_x = new_x[i]
                    lower_intersection_y = poly.polyval(lower_intersection_x, lower_fit)
                    upper_intersection_x = new_x[idx]
                    if len(upper_intersection_x) == 0:
                        continue
                    upper_intersection_x = upper_intersection_x[0]
                    upper_intersection_y = poly.polyval(upper_intersection_x, upper_fit)

                    layer_thickness[label][i] = utility.vector_distance(np.array([lower_intersection_x, lower_intersection_y]), np.array([upper_intersection_x, upper_intersection_y])) * \
                                                sitk_image.GetSpacing()[1]
                
                # logging.info(f'{fails} failed classifications (of {i})')
                keys = set(layer_thickness.keys())
                for key in keys:
                    dictionary[key] = np.hstack(
                        (dictionary[key], layer_thickness[key]))
            else:
                
                layer_thickness['eMT'] = np.zeros(len(new_x))
                layer_thickness['pMT'] = np.zeros(len(new_x))
                layer_thickness['iMT'] = np.zeros(len(new_x))
                layer_thickness['aMT'] = np.zeros(len(new_x))
                layer_thickness['cMT'] = np.zeros(len(new_x))

                fails = 0
                for i, val in enumerate(new_x):
                    label = utility.classify_tibial_point(np.array(
                        [xs[layer_index], val]), left_landmarks, right_landmarks, split_vector)
                    if label in set(['cLT', 'aLT', 'eLT', 'pLT', 'iLT']):
                        fails += 1           
                        continue
                    der = poly.polyder(lower_fit)
                    normal = poly.polyval(val, lower_fit) - (1 / poly.polyval(val, der)) * (new_x - val)
                    idx = np.argwhere(np.diff(np.sign(normal - poly.polyval(new_x, upper_fit)))).flatten()
                    lower_intersection_x = new_x[i]
                    lower_intersection_y = poly.polyval(lower_intersection_x, lower_fit)
                    upper_intersection_x = new_x[idx]
                    if len(upper_intersection_x) == 0:
                        continue
                    upper_intersection_x = upper_intersection_x[0]
                    upper_intersection_y = poly.polyval(upper_intersection_x, upper_fit)

                    layer_thickness[label][i] = utility.vector_distance(np.array([lower_intersection_x, lower_intersection_y]), np.array([upper_intersection_x, upper_intersection_y])) * \
                                                sitk_image.GetSpacing()[1]
                
                # logging.info(f'{fails} failed classifications (of {i})')
                keys = set(layer_thickness.keys())
                for key in keys:
                    dictionary[key] = np.hstack(
                        (dictionary[key], layer_thickness[key]))

    return dictionary


def function_for_pool(directory):
    t = time()
    n_femur = 0
    n_tibia = 0
    segmentation_directory = f'../Manual_Segmentations/{directory}/{directory}_segm.mhd'
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
        for layer in layers:
            n_femur += len(np.arange(min([x[0] for x in layer]), max([x[0] for x in layer])))
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=True, left=True, label=None,
                                                     tibia=False, split_vector=None, af=False)
        

        xs, layers = function_normals.build_cwbz_layers(cwbzr)
        for layer in layers:
            n_femur += len(np.arange(min([x[0] for x in layer]), max([x[0] for x in layer])))
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=True, left=False, label=None,
                                                     tibia=False, split_vector=None, af=False)
        

        xs, layers = function_normals.build_peripheral_layers(lpdf)
        for layer in layers:
            n_femur += len(np.arange(min([x[0] for x in layer]), max([x[0] for x in layer])))
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=False, left=False,
                                                     label='pLF', tibia=False, split_vector=None, af=False, pdf=True)
        

        xs, layers = function_normals.build_peripheral_layers(rpdf)
        for layer in layers:
            n_femur += len(np.arange(min([x[0] for x in layer]), max([x[0] for x in layer])))
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=False, left=False,
                                                     label='pMF', tibia=False, split_vector=None, af=False, pdf=True)
        

        xs, layers = function_normals.build_peripheral_layers(ladf)
        for layer in layers:
            n_femur += len(np.arange(min([x[0] for x in layer]), max([x[0] for x in layer])))
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=False, left=False,
                                                     label='aLF', tibia=False, split_vector=None, af=True)
        

        xs, layers = function_normals.build_peripheral_layers(radf)
        for layer in layers:
            n_femur += len(np.arange(min([x[0] for x in layer]), max([x[0] for x in layer])))
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=False, left=False,
                                                     label='aMF', tibia=False, split_vector=None, af=True)
        

    except Exception:
        logging.error(traceback.format_exc())
        return dict()

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
        xs, layers = function_normals.build_cwbz_layers(ldf)
        for layer in layers:
            n_tibia += len(np.arange(min([x[0] for x in layer]), max([x[0] for x in layer])))
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=False, left=True, label=None,
                                                     tibia=True, split_vector=split_vector, af=False)
        

        xs, layers = function_normals.build_cwbz_layers(rdf)
        for layer in layers:
            n_tibia += len(np.arange(min([x[0] for x in layer]), max([x[0] for x in layer])))
        total_thickness = calculate_region_thickness(sitk_image=sitk_image, layers=layers, dictionary=total_thickness,
                                                     xs=xs, left_landmarks=left_landmarks,
                                                     right_landmarks=right_landmarks, cwbz=False, left=False,
                                                     label=None, tibia=True, split_vector=split_vector, af=False)
        
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

    logging.info(f'::{time() - t}::')
    logging.info(f'++{n_tibia}++')
    logging.info(f'<<{n_femur}>>')
    return {**{'dir': directory}, **total_thickness}


def worker_init(q):
    # all records from worker processes go to qh and then into q
    qh = QueueHandler(q)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)


def logger_init():
    q = multiprocessing.Queue()
    # this is the handler for all log records
    handler = logging.FileHandler(f'logs/an.log', mode='w')
    handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(process)s - %(message)s"))

    # ql gets records from the queue and sends them to the handler
    ql = QueueListener(q, handler)
    ql.start()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # add the handler to the logger so records from this process are handled
    logger.addHandler(handler)

    return ql, q


def main():
    logging.basicConfig(filename='logs/analytical_normals_default.log',
                        encoding='utf-8', level=logging.DEBUG, filemode='w')
    logging.debug('Entered main.')

    try:

        filehandler = logging.FileHandler(
            f'logs/an.log', mode='w')
        filehandler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        root.addHandler(filehandler)

        q_listener, q = logger_init()

        files = utility.get_subdirs(None)

        files = files[:50]

        # debug !!
        # files = files[-100:-1]


        res_list = list()
        t = time()
        with ProcessPool(initializer=worker_init, initargs=[q]) as pool:
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
            f'out/an')
    except Exception as e:
        logging.debug(traceback.format_exc())
        logging.debug(sys.argv)

    logging.info(f'total execution time: {time() - t}')


if __name__ == '__main__':
    main()
