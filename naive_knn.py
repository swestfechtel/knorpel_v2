import logging
import math
import sys
import traceback
import utility
import multiprocessing
import time

import numpy as np
import pandas as pd
import pyvista as pv

from time import time
from scipy import stats
from logging.handlers import QueueHandler, QueueListener
from concurrent.futures import TimeoutError
from pebble import ProcessPool
from pebble.common import ProcessExpired


def nearest_neighbor(voxel, search_space):
    nearest_neighbor_distance = 10000000
    nearest_neighbor = None
    for candidate in search_space:
        if (d := utility.vector_distance(voxel, candidate)) < nearest_neighbor_distance:
            nearest_neighbor_distance = d
            nearest_neighbor = candidate

    # return (voxel, nearest_neighbor, nearest_neighbor_distance)
    return nearest_neighbor_distance


def knn_distance(directory):
    t = time()
    n_tibia = 0
    n_femur = 0

    segmentation_directory = f'../Manual_Segmentations/{directory}/{directory}_segm.mhd'
    # segmentation_directory = f'/images/Shape/Medical/Knees/OAI/Manual_Segmentations/{directory}/{directory}_segm.mhd'
    sitk_image, np_image = utility.read_image(segmentation_directory)

    try:
        tibial_cartilage = utility.build_3d_cartilage_array(np_image, 4)
        femoral_cartilage = utility.build_3d_cartilage_array(np_image, 3)
    except Exception:
        logging.error(traceback.format_exc())
        logging.warning(f'Got error for file {directory} while trying to build 3d arrays. Return empty dict.')
        return dict()

    tibial_vectors = [list(element) for element in tibial_cartilage]
    femoral_vectors = [list(element) for element in femoral_cartilage]

    # tibial thickness
    try:
        lower_mesh, upper_mesh = utility.build_tibial_meshes(tibial_vectors)
    except Exception:
        logging.error(traceback.format_exc())
        logging.warning(
            f'Got delaunay error for file {directory} while trying to build tibial meshes. Return empty dict.')
        return dict()

    df = pd.DataFrame(data=tibial_cartilage, columns=['x', 'y', 'z'])
    max_z = df.groupby(['x', 'y']).max()

    tmp1 = [np.array(item) for item in max_z.index]
    tmp2 = [item for item in max_z.to_numpy()]
    max_z = np.column_stack((tmp1, tmp2))

    left_tibial_landmarks, right_tibial_landmarks, split_vector = utility.tibial_landmarks(max_z)
    left_plate, right_plate = utility.split_into_plates(tibial_vectors, split_vector)


    lower_mesh_left, upper_mesh_left = utility.build_tibial_meshes(left_plate)
    lower_mesh_right, upper_mesh_right = utility.build_tibial_meshes(right_plate)

    tibial_thickness = dict()
    tibial_thickness['cLT'] = np.zeros(lower_mesh.n_points)
    tibial_thickness['aLT'] = np.zeros(lower_mesh.n_points)
    tibial_thickness['eLT'] = np.zeros(lower_mesh.n_points)
    tibial_thickness['pLT'] = np.zeros(lower_mesh.n_points)
    tibial_thickness['iLT'] = np.zeros(lower_mesh.n_points)
    tibial_thickness['cMT'] = np.zeros(lower_mesh.n_points)
    tibial_thickness['aMT'] = np.zeros(lower_mesh.n_points)
    tibial_thickness['eMT'] = np.zeros(lower_mesh.n_points)
    tibial_thickness['pMT'] = np.zeros(lower_mesh.n_points)
    tibial_thickness['iMT'] = np.zeros(lower_mesh.n_points)

    mod = lambda l: l * sitk_image.GetSpacing()[1]

    distances = np.array([nearest_neighbor(x, upper_mesh_left.points) for x in lower_mesh_left.points])
    distances = mod(distances)

    for i, distance in enumerate(distances):
        key = utility.classify_tibial_point(lower_mesh_left.points[i][:2], left_tibial_landmarks, right_tibial_landmarks, split_vector)
        if key not in ('cLT', 'aLT', 'pLT', 'iLT', 'eLT'):
            logging.warning(f'Got wrong key {key} for lateral tibia.')
            continue

        tibial_thickness[key][i] = distance

    distances = np.array([nearest_neighbor(x, upper_mesh_right.points) for x in lower_mesh_right.points])
    distances = mod(distances)

    for i, distance in enumerate(distances):
        key = utility.classify_tibial_point(lower_mesh_right.points[i][:2], left_tibial_landmarks, right_tibial_landmarks, split_vector)
        if key not in ('cMT', 'aMT', 'pMT', 'iMT', 'eMT'):
            logging.warning(f'Got wrong key {key} for medial tibia.')
            continue

        tibial_thickness[key][i] = distance

    n_tibia = lower_mesh_left.n_points + lower_mesh_right.n_points

    # femoral thickness
    cwbzl, cwbzr = utility.extract_central_weightbearing_zone(femoral_vectors, tibial_vectors)
    lower_mesh_left, upper_mesh_left = utility.build_femoral_meshes(cwbzl)
    lower_mesh_right, upper_mesh_right = utility.build_femoral_meshes(cwbzr)

    left_landmarks = utility.femoral_landmarks(upper_mesh_left.points)
    right_landmarks = utility.femoral_landmarks(upper_mesh_right.points)

    left_thickness = dict()
    left_thickness['ecLF'] = np.zeros(lower_mesh_left.n_points)
    left_thickness['ccLF'] = np.zeros(lower_mesh_left.n_points)
    left_thickness['icLF'] = np.zeros(lower_mesh_left.n_points)

    right_thickness = dict()
    right_thickness['ecMF'] = np.zeros(lower_mesh_right.n_points)
    right_thickness['ccMF'] = np.zeros(lower_mesh_right.n_points)
    right_thickness['icMF'] = np.zeros(lower_mesh_right.n_points)

    distances = np.array([nearest_neighbor(x, upper_mesh_left.points) for x in lower_mesh_left.points])
    distances = mod(distances)

    for i, distance in enumerate(distances):
        key = utility.classify_femoral_point(lower_mesh_left.points[i][:2], left_landmarks, left=True)
        if key not in ('ecLF', 'ccLF', 'icLF'):
            logging.warning(f'Got wrong key {key} for lateral femur.')
            continue

        left_thickness[key][i] = distance

    distances = np.array([nearest_neighbor(x, upper_mesh_right.points) for x in lower_mesh_right.points])
    distances = mod(distances)

    for i, distance in enumerate(distances):
        key = utility.classify_femoral_point(lower_mesh_right.points[i][:2], right_landmarks, left=False)
        if key not in ('ecMF', 'ccMF', 'icMF'):
            logging.warning(f'Got wrong key {key} for medial femur.')
            continue

        right_thickness[key][i] = distance

    femoral_thickness = dict()
    femoral_thickness.update(left_thickness)
    femoral_thickness.update(right_thickness)

    lpdf, rpdf, adf = utility.extract_anterior_posterior_zones(femoral_vectors, cwbzl, cwbzr)
    ladf, radf = utility.split_anterior_part(adf)
    lp_lower_mesh, lp_upper_mesh = utility.build_tibial_meshes(lpdf.to_numpy())  # left (lateral) posterior region
    rp_lower_mesh, rp_upper_mesh = utility.build_tibial_meshes(rpdf.to_numpy())  # right (medial) posterior region
    la_lower_mesh, la_upper_mesh = utility.build_tibial_meshes(ladf.to_numpy())  # anterior region
    ra_lower_mesh, ra_upper_mesh = utility.build_tibial_meshes(radf.to_numpy())

    distances = np.array([nearest_neighbor(x, lp_upper_mesh.points) for x in lp_lower_mesh.points])
    distances = mod(distances)

    femoral_thickness['pLF'] = distances

    distances = np.array([nearest_neighbor(x, rp_upper_mesh.points) for x in rp_lower_mesh.points])
    distances = mod(distances)

    femoral_thickness['pMF'] = distances

    distances = np.array([nearest_neighbor(x, la_upper_mesh.points) for x in la_lower_mesh.points])
    distances = mod(distances)

    femoral_thickness['aLF'] = distances

    distances = np.array([nearest_neighbor(x, ra_upper_mesh.points) for x in ra_lower_mesh.points])
    distances = mod(distances)

    femoral_thickness['aMF'] = distances

    keys = set(tibial_thickness.keys())

    for key in keys:
        value = tibial_thickness[key]
        mask = value == 0
        value[mask] = np.nan

        tibial_thickness[key + '.aSD'] = np.nanstd(value)
        tibial_thickness[key + '.aMav'] = np.nanmean(-np.sort(-value)[:math.ceil(len(value) * 0.01)])
        tibial_thickness[key + '.aMiv'] = np.nanmean(np.sort(value)[:math.ceil(len(value) * 0.01)])
        tibial_thickness[key] = np.nanmean(value)

    keys = set(femoral_thickness.keys())

    for key in keys:
        value = femoral_thickness[key]
        mask = value == 0
        value[mask] = np.nan

        femoral_thickness[key + '.aSD'] = np.nanstd(value)
        femoral_thickness[key + '.aMav'] = np.nanmean(-np.sort(-value)[:math.ceil(len(value) * 0.01)])
        femoral_thickness[key + '.aMiv'] = np.nanmean(np.sort(value)[:math.ceil(len(value) * 0.01)])
        femoral_thickness[key] = np.nanmean(value)

    n_femur = lower_mesh_left.n_points + lower_mesh_right.n_points + lp_lower_mesh.n_points + rp_lower_mesh.n_points + la_lower_mesh.n_points + ra_lower_mesh.n_points

    logging.info(f'++{n_tibia}++')
    logging.info(f'<<{n_femur}>>')
    logging.info(f'::{time() - t}::')

    return {**{'dir': directory}, **femoral_thickness, **tibial_thickness}


def worker_init(q):
    # all records from worker processes go to qh and then into q
    qh = QueueHandler(q)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)


def logger_init(arg):
    q = multiprocessing.Queue()
    # this is the handler for all log records
    # handler = logging.FileHandler(f'/work/scratch/westfechtel/pylogs/naive_knn/{arg}.log', mode='w')
    handler = logging.FileHandler(f'logs/naive_knn.log', mode='w')
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
    # logging.basicConfig(filename='/work/scratch/westfechtel/pylogs/naive_knn/naive_knn_default.log', encoding='utf-8', level=logging.DEBUG, filemode='w')
    logging.basicConfig(filename='logs/naive_knn_default.log', encoding='utf-8', level=logging.DEBUG, filemode='w')
    logging.info('Entered main.')

    try:
        # assert len(sys.argv) == 2
        # chunk = np.load(f'/work/scratch/westfechtel/chunks/{sys.argv[1]}.npy')

        # filehandler = logging.FileHandler(f'/work/scratch/westfechtel/pylogs/naive_knn/{sys.argv[1]}.log', mode='w')
        filehandler = logging.FileHandler(f'logs/naive_knn.log', mode='w')
        filehandler.setLevel(logging.DEBUG)
        filehandler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        root.addHandler(filehandler)

        # q_listener, q = logger_init(sys.argv[1])
        q_listener, q = logger_init(None)

        files = utility.get_subdirs(None)
        # files = utility.get_subdirs(chunk)

        files = files[:50]

        res_list = list()
        t = time()
        with ProcessPool(initializer=worker_init, initargs=[q]) as pool:
            res = pool.map(knn_distance, files)

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
            df.to_pickle(f'out/naive_knn')
            # df.to_pickle(f'/work/scratch/westfechtel/manpickles/naive_knn/{sys.argv[1]}')

    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(sys.argv)

    logging.info(f'total execution time: {time() - t}')


if __name__ == '__main__':
    main()