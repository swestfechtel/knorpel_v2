import math
import os
import logging
import traceback
import sys
import utility

import numpy as np
import pandas as pd
import pyvista as pv

from scipy import stats
from multiprocessing import Pool
from pebble import ProcessPool
from pebble.common import ProcessExpired
from concurrent.futures import TimeoutError
from time import time
# from __future__ import division


def split_femoral_volume(vectors: list) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a femoral cartilage volume into three parts.

    Computes a central point from the cartilage volume and for every vector of the volume, calculates the vector
    distance to the central point.
    Then, the volume is split into three parts. First, the middle section of the volume is isolated by using the
    range between maximum and minimum z coordinate for every x coordinate. Every x for which the z range is below
    the median z range is added, and outliers are cleared by using z score. Now, that the x coordinates of the middle
    section are known, the volume can be split into three parts.

    :param vectors: An array of three-dimensional vectors (x, y, z) making up the cartilage volume
    :return: pandas dataframes of vectors belonging to the left, central and right part
    """
    x, y, z, xy = utility.get_xyz(vectors)
    df = pd.DataFrame(data={'x': z, 'y': y, 'z': x}, columns=['x', 'y', 'z']) # swap x and z
    center = np.array([df.x.min() + (df.x.max() - df.x.min()) / 2,
                       df.y.min() + (df.y.max() - df.y.min()) / 2,
                       df.y.min() + (df.y.max() - df.y.min()) / 2])

    df['dist_to_cog'] = np.zeros(df.shape[0])
    df['dist_to_cog'] = df.apply(lambda l: utility.vector_distance([l.x, l.y, l.z], center), axis=1)
    df = df.sort_values(by='x', ascending=True)
    zrange = df.groupby(by=['x'])['z'].max() - df.groupby(by=['x'])['z'].min()
    zmed = zrange.median()
    zindex = zrange.loc[zrange < zmed].index.to_numpy()
    mask = np.abs(stats.zscore(zindex)) < 2 # filter out all values deviating more than 2 stds
    lower_bound = zindex[mask].min()
    upper_bound = zindex[mask].max()

    right_portion = df.loc[df['x'] < lower_bound]
    right_portion = right_portion[['z', 'y', 'x', 'dist_to_cog']]
    right_portion.columns = ['x', 'y', 'z', 'dist_to_cog'] # swap x and z again because pyvista is weird

    middle_portion = df.loc[df['x'] > lower_bound].loc[df['x'] < upper_bound]

    left_portion = df.loc[df['x'] > upper_bound]
    left_portion = left_portion[['z', 'y', 'x', 'dist_to_cog']]
    left_portion.columns = ['x', 'y', 'z', 'dist_to_cog']

    return left_portion, middle_portion, right_portion


def build_portion_delaunay(portion: pd.DataFrame) -> [pv.core.pointset.PolyData, pv.core.pointset.PolyData]:
    """
    Builds upper and lower delaunay mesh of a femoral cartilage volume.

    Groups all vectors by (x, y) and adds the vector (x, y, max(distance to central point)) to the upper mesh, and
    the vector (x, y, min(distance to central point)) to the lower mesh, for every pair (x, y).

    :param portion: A pandas dataframe containing all vectors (x, y, z) making up the portion as well as the distance
    to the central point for each vector
    :return: A lower and upper mesh, by distance to central point
    """
    max_dist = portion[portion.groupby(['x', 'y'])['dist_to_cog'].transform(max) == portion['dist_to_cog']]
    min_dist = portion[portion.groupby(['x', 'y'])['dist_to_cog'].transform(min) == portion['dist_to_cog']]

    max_dist = [item[:3] for item in max_dist.to_numpy()]
    min_dist = [item[:3] for item in min_dist.to_numpy()]

    outer_cloud = pv.PolyData(max_dist)
    inner_cloud = pv.PolyData(min_dist)

    return outer_cloud.delaunay_2d(), inner_cloud.delaunay_2d()


def combine_to_cloud(left_mesh: pv.core.pointset.PolyData, middle_mesh: pv.core.pointset.PolyData, right_mesh: pv.core.pointset.PolyData) -> pd.DataFrame:
    """
    Re-combines three parts of a split femoral vector volume into a single dataframe.

    Use to build a coherent upper/lower mesh from split parts.

    :param left_mesh: An upper/lower delaunay mesh from the left part
    :param middle_mesh: An upper/lower delaunay mesh from the central part
    :param right_mesh: An upper/lower delaunay mesh from the right part
    :return: A pandas dataframe containing all the vectors from the three parts, combined and re-aligned
    """
    ldf = pd.DataFrame(data=left_mesh.points, columns=['x', 'y', 'z'])
    mdf = pd.DataFrame(data=middle_mesh.points, columns=['x', 'y', 'z'])
    rdf = pd.DataFrame(data=right_mesh.points, columns=['x', 'y', 'z'])

    ldf = ldf[['z', 'y', 'x']]
    ldf.columns = ['x', 'y', 'z']

    rdf = rdf[['z', 'y', 'x']]
    rdf.columns = ['x', 'y', 'z']

    return pd.concat([ldf, mdf, rdf])


def function_for_pool(directory):
    """
    Function to use for a multiprocessing pool.

    Reads a scan, builds the femoral and tibial delaunay meshes, landmarks, respective result dictionaries.
    Calculates the average thickness for each subregion for both femoral and tibial cartilage, as well as
    statistical measures.

    :param directory: The image file to read
    :return: A dictionary containing the file name and average thickness and statistical measures for each subregion
    """
    segmentation_directory = f'/images/Shape/Medical/Knees/OAI/Manual_Segmentations/{directory}/{directory}_segm.mhd'
    # segmentation_directory = f'/work/scratch/westfechtel/segmentations/{directory}'
    sitk_image, np_image = utility.read_image(segmentation_directory)
    try:
        tibial_cartilage = utility.build_3d_cartilage_array(np_image, 4)
        femoral_cartilage = utility.build_3d_cartilage_array(np_image, 3)
    except Exception:
        logging.error(traceback.format_exc())
        logging.warning(f'Got error for file {directory} while trying to build 3d arrays. Return empty dict.')
        # return {**{'dir': directory}, **{}, **{}}
        return dict()

    tibial_vectors = [list(element) for element in tibial_cartilage]
    femoral_vectors = [list(element) for element in femoral_cartilage]

    # tibial thickness
    try:
        lower_mesh, upper_mesh = utility.build_tibial_meshes(tibial_vectors)
    except Exception:
        logging.error(traceback.format_exc())
        logging.warning(f'Got delaunay error for file {directory} while trying to build tibial meshes. Return empty dict.')
        # return {**{'dir': directory}, **{}, **{}}
        return dict()


    # determine landmarks for tibial plates for subregion classification
    left_tibial_landmarks, right_tibial_landmarks, split_vector = utility.tibial_landmarks(lower_mesh.points)

    # calculate average thickness per region by ray tracing normal vectors from lower to upper surface
    try:
        lower_normals = lower_mesh.compute_normals(cell_normals=False)
    except Exception:
        logging.error(traceback.format_exc())
        logging.warning(f'Got error for file {directory} while trying to compute normals. Return empty dict.')
        # return {**{'dir': directory}, **{}, **{}}
        return dict()

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

    lower_normals['distances'] = np.zeros(lower_mesh.n_points)
    try:
        lower_normals, femoral_thickness = utility.calculate_distance(lower_normals, lower_mesh, upper_mesh, sitk_image,
                                                          left_tibial_landmarks, right_tibial_landmarks,
                                                          split_vector, tibial_thickness, femur=False)
    except Exception:
        logging.error(traceback.format_exc())
        logging.warning(f'Got error for file {directory} while trying to calculate distance. Return empty dict.')
        # return {**{'dir': directory}, **{}, **{}}
        return dict()

    # total average thickness
    mask = lower_normals['distances'] == 0
    lower_normals['distances'][mask] = np.nan
    total_avg_thickness = np.nanmean(lower_normals['distances'])

    keys = set(tibial_thickness.keys())
    for key in keys:
        value = tibial_thickness[key]
        mask = value == 0
        value[mask] = np.nan
        tibial_thickness[key + '.aSD'] = np.nanstd(value)
        tibial_thickness[key + '.aMav'] = np.nanmean(-np.sort(-value)[:math.ceil(len(value) * 0.01)])
        tibial_thickness[key + '.aMiv'] = np.nanmean(np.sort(value)[:math.ceil(len(value) * 0.01)])
        tibial_thickness[key] = np.nanmean(value)

    # femoral thickness
    try:
        left_portion, middle_portion, right_portion = split_femoral_volume(femoral_vectors)
        left_outer, left_inner = build_portion_delaunay(left_portion)
        middle_outer, middle_inner = build_portion_delaunay(middle_portion)
        right_outer, right_inner = build_portion_delaunay(right_portion)
        outer_cloud = combine_to_cloud(left_outer, middle_outer, right_outer)
    except Exception:
        logging.error(traceback.format_exc())
        logging.warning(f'Got delaunay error for file {directory} while trying to split femoral volume. Return empty dict.')
        # return {**{'dir': directory}, **{}, **{}}
        return dict()

    left_femoral_landmarks, right_femoral_landmarks, split_vector = utility.femoral_landmarks(outer_cloud.to_numpy())

    try:
        left_inner_normals = left_inner.compute_normals(cell_normals=False)
        middle_inner_normals = middle_inner.compute_normals(cell_normals=False)
        right_inner_normals = right_inner.compute_normals(cell_normals=False)
    except Exception:
        logging.error(traceback.format_exc())
        logging.warning(f'Got error for file {directory} while trying to compute normals (2). Return empty dict.')
        # return {**{'dir': directory}, **{}, **{}}
        return dict()

    left_thickness = dict()
    left_thickness['ecLF'] = np.zeros(left_inner.n_points)
    left_thickness['ccLF'] = np.zeros(left_inner.n_points)
    left_thickness['icLF'] = np.zeros(left_inner.n_points)
    left_thickness['icMF'] = np.zeros(left_inner.n_points)
    left_thickness['ccMF'] = np.zeros(left_inner.n_points)
    left_thickness['ecMF'] = np.zeros(left_inner.n_points)

    middle_thickness = left_thickness.copy()
    middle_thickness = {key: np.zeros(middle_inner.n_points) for key in middle_thickness.keys()}

    right_thickness = left_thickness.copy()
    right_thickness = {key: np.zeros(right_inner.n_points) for key in right_thickness.keys()}

    try:
        _, left_thickness = utility.calculate_distance(left_inner_normals, left_inner, left_outer, sitk_image,
                                               left_femoral_landmarks, right_femoral_landmarks,
                                               split_vector, left_thickness, femur=True)

        _, middle_thickness = utility.calculate_distance(middle_inner_normals, middle_inner, middle_outer, sitk_image,
                                                 left_femoral_landmarks, right_femoral_landmarks,
                                                 split_vector, middle_thickness, femur=True)

        _, right_thickness = utility.calculate_distance(right_inner_normals, right_inner, right_outer, sitk_image,
                                                left_femoral_landmarks, right_femoral_landmarks,
                                                split_vector, right_thickness, femur=True)
    except Exception:
        logging.error(traceback.format_exc())
        logging.warning(f'Got error for file {directory} while trying to calculate distance (2). Return empty dict.')
        # return {**{'dir': directory}, **{}, **{}}
        return dict()

    femoral_thickness = {key: np.hstack((left_thickness[key], middle_thickness[key], right_thickness[key])) for key in
                         left_thickness.keys()}

    keys = set(femoral_thickness.keys())
    for key in keys:
        value = femoral_thickness[key]
        mask = value == 0
        value[mask] = np.nan
        femoral_thickness[key + '.aSD'] = np.nanstd(value)
        femoral_thickness[key + '.aMav'] = np.nanmean(-np.sort(-value)[:math.ceil(len(value) * 0.01)])
        femoral_thickness[key + '.aMiv'] = np.nanmean(np.sort(value)[:math.ceil(len(value) * 0.01)])
        femoral_thickness[key] = np.nanmean(value)
    
    logging.info(f'File {directory} done.')
    return {**{'dir': directory}, **femoral_thickness, **tibial_thickness}


def main():
    logging.basicConfig(filename='/work/scratch/westfechtel/pylogs/mesh/mesh_default.log', encoding='utf-8', level=logging.DEBUG, filemode='w')
    logging.info('Entered main.')

    try:
        assert len(sys.argv) == 2
        chunk = np.load(f'/work/scratch/westfechtel/chunks/{sys.argv[1]}.npy')
        # chunk = sys.argv[1]

        filehandler = logging.FileHandler(f'/work/scratch/westfechtel/pylogs/mesh/{sys.argv[1]}.log', mode='w')
        filehandler.setLevel(logging.DEBUG)
        filehandler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
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
            res = pool.map(function_for_pool, files, timeout=180)
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
            df.to_pickle(f'/work/scratch/westfechtel/manpickles/mesh/{sys.argv[1]}')

    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(sys.argv)


if __name__ == '__main__':
    main()
