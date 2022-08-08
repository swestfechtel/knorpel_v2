import logging
import math
import sys
import traceback
import utility

import numpy as np
import pandas as pd
import pyvista as pv

from time import time
from scipy import stats
from scipy.spatial import KDTree


def knn_distance(directory):
    n_tibia = 0
    n_femur = 0

    segmentation_directory = f'../Manual_Segmentations/{directory}/{directory}_segm.mhd'
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

    tree = KDTree(lower_mesh_left.points)
    distances, _ = tree.query(x=upper_mesh_left.points, k=1, workers=-1)
    distances = mod(distances)

    for i, distance in enumerate(distances):
        key = utility.classify_tibial_point(lower_mesh_left.points[i][:2], left_tibial_landmarks, right_tibial_landmarks, split_vector)
        if key not in ('cLT', 'aLT', 'pLT', 'iLT', 'eLT'):
            logging.warning(f'Got wrong key {key} for lateral tibia.')
            continue

        tibial_thickness[key][i] = distance

    tree = KDTree(lower_mesh_right.points)
    distances, _ = tree.query(x=upper_mesh_right.points, k=1, workers=-1)
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

    tree = KDTree(lower_mesh_left.points)
    distances, _ = tree.query(x=upper_mesh_left.points, k=1, workers=-1)
    distances = mod(distances)

    for i, distance in enumerate(distances):
        key = utility.classify_femoral_point(lower_mesh_left.points[i][:2], left_landmarks, left=True)
        if key not in ('ecLF', 'ccLF', 'icLF'):
            logging.warning(f'Got wrong key {key} for lateral femur.')
            continue

        left_thickness[key][i] = distance

    tree = KDTree(lower_mesh_right.points)
    distances, _ = tree.query(x=upper_mesh_right.points, k=1, workers=-1)
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

    

    tree = KDTree(lp_lower_mesh.points)
    distances, _ = tree.query(x=lp_upper_mesh.points, k=1, workers=-1)

    femoral_thickness['pLF'] = mod(distances)

    tree = KDTree(rp_lower_mesh.points)
    distances, _ = tree.query(x=rp_upper_mesh.points, k=1, workers=-1)

    femoral_thickness['pMF'] = mod(distances)

    tree = KDTree(la_lower_mesh.points)
    distances, _ = tree.query(x=la_upper_mesh.points, k=1, workers=-1)

    femoral_thickness['aLF'] = mod(distances)

    tree = KDTree(ra_lower_mesh.points)
    distances, _ = tree.query(x=ra_upper_mesh.points, k=1, workers=-1)

    femoral_thickness['aMF'] = mod(distances)

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

    return {**{'dir': directory}, **femoral_thickness, **tibial_thickness}


def main():
    logging.basicConfig(filename='logs/knn_default.log', encoding='utf-8',
                        level=logging.DEBUG, filemode='w')
    logging.info('Entered main.')

    try:
        filehandler = logging.FileHandler(f'logs/knn.log', mode='w')
        filehandler.setLevel(logging.DEBUG)
        filehandler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        root.addHandler(filehandler)

        files = utility.get_subdirs(None)

        # files = files[:50]

        res = np.empty(len(files), dtype='object')
        t = time()
        for i, directory in enumerate(files):
            tt = time()
            try:
                res[i] = knn_distance(directory)
                logging.info(f'::{time() - tt}::')
            except Exception:
                logging.error(traceback.format_exc())
                continue

        logging.info(f'Elapsed time: {time() - t}')
        res = res[res != None]
        res = list(res)
        df = pd.DataFrame.from_dict(res)
        df.index = df['dir']
        df = df.drop('dir', axis=1)
        # df.to_excel('mesh.xlsx')
        df.to_pickle(f'out/knn')

    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(sys.argv)

    logging.info(f'total execution time: {time() - t}')


if __name__ == '__main__':
    main()