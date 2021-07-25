import logging
import sys
import traceback
import utility
import os
import math

import numpy as np
import pandas as pd
import pyvista as pv

from multiprocessing import Pool
from functools import partial


def vector_trace(sphere_points, sphere_normals, df):
    """
    Uses ray tracing to find an inner and outer vector along a normal vector from a central point against a point cloud.

    :param sphere_points: The central point
    :param sphere_normals: The normal vectors corresponding to the central point
    :param df: A pandas dataframe containing all vectors making up the point cloud
    :return: The inner and outer point along the normal vector, if exist
    """
    P = np.array(sphere_points)
    V = np.array(sphere_normals)
    alpha = 1
    max_iterations = 100
    inner_point = None
    outer_point = None
    tmp = P + V
    px = P[0]
    py = P[1]
    pz = P[2]
    local_df = df.copy()

    if tmp[0] < px:
        local_df = df.loc[df['x'] < px]
    else:
        local_df = df.loc[df['x'] >= px]

    if tmp[1] < py:
        local_df = local_df.loc[local_df['y'] < py]
    else:
        local_df = local_df.loc[local_df['y'] >= py]

    if tmp[2] < pz:
        local_df = local_df.loc[local_df['z'] < pz]
    else:
        local_df = local_df.loc[local_df['z'] >= pz]

    while True:
        g = P + alpha * V
        points = local_df.loc[abs(local_df['x'] - g[0]) <= 2].loc[abs(local_df['y'] - g[1]) <= 2].loc[abs(local_df['z'] - g[2]) <= 2]
        if points.shape[0] > 0:
            break

        alpha += 1
        if alpha > max_iterations:
            break

    if points.shape[0] == 0:
        return None

    inner_point = points.sort_values(by='dist', ascending=True).iloc[0:1].to_numpy()[0][0:3]
    P = inner_point
    alpha = 1

    while True:
        g = P + alpha * V
        points = local_df.loc[abs(local_df['x'] - g[0]) <= 2].loc[abs(local_df['y'] - g[1]) <= 2].loc[abs(local_df['z'] - g[2]) <= 2]
        if points.shape[0] == 0:
            break

        outer_point = points.sort_values(by='dist', ascending=False).iloc[0:1].to_numpy()[0][0:3]
        alpha += 1
        if alpha > max_iterations:
            break

    return [inner_point, outer_point]


def fun(directory):
    """
    Function to use for a single image file.

    Reads a scan, extracts the femoral and tibial cartilage volumes, calculates a central point for each volume.
    For each volume, uses ray tracing from the corresponding central point against the volume to calculate average
    thickness for each subregion, as well as statistical measures.

    :param directory: The image file to read
    :return: A dictionary containing the file name and average thickness and statistical measures for each subregion
    """
    sitk_image, np_image = utility.read_image(f'/images/Shape/Medical/Knees/OAI/Manual_Segmentations/{directory}/{directory}_segm.mhd')
    # sitk_image, np_image = utility.read_image('images/9001104/9001104_segm.mhd')
    tibial_cartilage = utility.build_3d_cartilage_array(np_image, 4)
    femoral_cartilage = utility.build_3d_cartilage_array(np_image, 3)

    femoral_vectors = [list(element) for element in femoral_cartilage]
    tibial_vectors = [list(element) for element in tibial_cartilage]
    x, y, z, xy = utility.get_xyz(femoral_vectors)
    df = pd.DataFrame(data={'x': z, 'y': y, 'z': x}, columns=['x', 'y', 'z'])
    center = np.array([df.x.min() + (df.x.max() - df.x.min()) / 2,
                       df.y.min() + (df.y.max() - df.y.min()) / 2,
                       df.y.min() + (df.y.max() - df.y.min()) / 2])

    # cloud = pv.PolyData(df.to_numpy())
    df['dist'] = np.zeros(df.shape[0])
    df['dist'] = df.apply(lambda l: utility.vector_distance([l.x, l.y, l.z], center), axis=1)

    num_sp = 60
    sphere = pv.Sphere(center=center, radius=1, theta_resolution=num_sp, phi_resolution=num_sp)
    sphere.compute_normals(point_normals=True, cell_normals=False, inplace=True)

    sphere_iter = np.array([[np.nan, np.nan]] * sphere.n_points, dtype='object')
    for i in range(sphere.n_points):
        sphere_iter[i][0] = tuple(sphere.points[i])
        sphere_iter[i][1] = tuple(sphere['Normals'][i])

    with Pool() as pool:
        res = pool.starmap(partial(vector_trace, df=df), iterable=sphere_iter)

    res = np.array(res, dtype='object')
    res = res[res != None]
    inner_points = [item[0] for item in res]
    outer_points = [item[1] for item in res]

    femoral_thickness = dict()
    femoral_thickness['ecLF'] = np.zeros(len(outer_points))
    femoral_thickness['ccLF'] = np.zeros(len(outer_points))
    femoral_thickness['icLF'] = np.zeros(len(outer_points))
    femoral_thickness['icMF'] = np.zeros(len(outer_points))
    femoral_thickness['ccMF'] = np.zeros(len(outer_points))
    femoral_thickness['ecMF'] = np.zeros(len(outer_points))

    left_landmarks, right_landmarks, split_vector = utility.femoral_landmarks(outer_points)
    for i in range(len(outer_points)):
        label = utility.classify_femoral_point(outer_points[i][:2], left_landmarks, right_landmarks, split_vector)
        femoral_thickness[label][i] = utility.vector_distance(outer_points[i], inner_points[i]) * \
                                      sitk_image.GetSpacing()[2]

    keys = set(femoral_thickness.keys())
    for key in keys:
        value = femoral_thickness[key]
        mask = value == 0
        value[mask] = np.nan
        femoral_thickness[key + '.aSD'] = np.nanstd(value)
        femoral_thickness[key + '.aMav'] = np.nanmean(-np.sort(-value)[:math.ceil(len(value) * 0.01)])
        femoral_thickness[key + '.aMiv'] = np.nanmean(np.sort(value)[:math.ceil(len(value) * 0.01)])
        femoral_thickness[key] = np.nanmean(value)

    x, y, z, xy = utility.get_xyz(tibial_vectors)
    df = pd.DataFrame(data={'x': z, 'y': y, 'z': x}, columns=['x', 'y', 'z'])
    center = np.array([df.x.min() + (df.x.max() - df.x.min()) / 2,
                       df.y.min() + (df.y.max() - df.y.min()) / 2,
                       df.y.min() + (df.y.max() - df.y.min()) / 2])

    # cloud = pv.PolyData(df.to_numpy())
    df['dist'] = np.zeros(df.shape[0])
    df['dist'] = df.apply(lambda l: utility.vector_distance([l.x, l.y, l.z], center), axis=1)

    num_sp = 60
    sphere = pv.Sphere(center=center, radius=1, theta_resolution=num_sp, phi_resolution=num_sp)
    sphere.compute_normals(point_normals=True, cell_normals=False, inplace=True)

    sphere_iter = np.array([[np.nan, np.nan]] * sphere.n_points, dtype='object')
    for i in range(sphere.n_points):
        sphere_iter[i][0] = tuple(sphere.points[i])
        sphere_iter[i][1] = tuple(sphere['Normals'][i])

    with Pool() as pool:
        res = pool.starmap(partial(vector_trace, df=df), iterable=sphere_iter)

    res = np.array(res, dtype='object')
    res = res[res != None]
    inner_points = [item[0] for item in res]
    outer_points = [item[1] for item in res]

    tibial_thickness = dict()
    tibial_thickness['cLT'] = np.zeros(len(outer_points))
    tibial_thickness['aLT'] = np.zeros(len(outer_points))
    tibial_thickness['eLT'] = np.zeros(len(outer_points))
    tibial_thickness['pLT'] = np.zeros(len(outer_points))
    tibial_thickness['iLT'] = np.zeros(len(outer_points))
    tibial_thickness['cMT'] = np.zeros(len(outer_points))
    tibial_thickness['aMT'] = np.zeros(len(outer_points))
    tibial_thickness['eMT'] = np.zeros(len(outer_points))
    tibial_thickness['pMT'] = np.zeros(len(outer_points))
    tibial_thickness['iMT'] = np.zeros(len(outer_points))

    left_landmarks, right_landmarks, split_vector = utility.tibial_landmarks(outer_points)
    for i in range(len(outer_points)):
        label = utility.classify_tibial_point(outer_points[i][:2], left_landmarks, right_landmarks, split_vector)
        tibial_thickness[label][i] = utility.vector_distance(outer_points[i], inner_points[i]) * \
                                      sitk_image.GetSpacing()[2]

    keys = set(tibial_thickness.keys())
    for key in keys:
        value = tibial_thickness[key]
        mask = value == 0
        value[mask] = np.nan
        tibial_thickness[key + '.aSD'] = np.nanstd(value)
        tibial_thickness[key + '.aMav'] = np.nanmean(-np.sort(-value)[:math.ceil(len(value) * 0.01)])
        tibial_thickness[key + '.aMiv'] = np.nanmean(np.sort(value)[:math.ceil(len(value) * 0.01)])
        tibial_thickness[key] = np.nanmean(value)

    return {**{'dir': directory}, **femoral_thickness, **tibial_thickness}


def main():
    if os.path.exists('/work/scratch/westfechtel/pylogs/sphere.log'):
        os.remove('/work/scratch/westfechtel/pylogs/sphere.log')

    logging.basicConfig(filename='/work/scratch/westfechtel/pylogs/sphere.log', encoding='utf-8', level=logging.DEBUG)
    logging.debug('Entered main.')

    try:
        assert len(sys.argv) == 2
        chunk = np.load(f'/work/scratch/westfechtel/chunks/{sys.argv[1]}.npy')
        logging.debug(f'Using chunk {sys.argv[1]} with length {len(chunk)}.')

        dirs = utility.get_subdirs(chunk)
        dirs = dirs[:1]
        res = np.empty(len(dirs), dtype='object')
        for i, directory in enumerate(dirs):
            try:
                res[i] = fun(directory)
                if i % 10 == 0:
                    logging.debug(f'Iteration #{i}')
            except Exception:
                continue

        res = res[res != None]
        df = pd.DataFrame.from_dict(res)
        df.index = df['dir']
        df = df.drop('dir', axis=1)
        # df.to_excel('mesh.xlsx')
        df.to_pickle(f'/work/scratch/westfechtel/pickles/sphere/{sys.argv[1]}')
    except Exception as e:
        logging.debug(traceback.format_exc())
        logging.debug(sys.argv)


if __name__ == '__main__':
    main()
