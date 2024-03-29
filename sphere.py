import logging
import math
import sys
import traceback
from functools import partial
from multiprocessing import Pool
from time import time

import numpy as np
import pandas as pd
import pyvista as pv
from sklearn.cluster import KMeans
from logging.handlers import QueueHandler, QueueListener

import utility


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
        points = local_df.loc[abs(local_df['x'] - g[0]) <= 2].loc[abs(local_df['y'] - g[1]) <= 2].loc[
            abs(local_df['z'] - g[2]) <= 2]
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
        points = local_df.loc[abs(local_df['x'] - g[0]) <= 2].loc[abs(local_df['y'] - g[1]) <= 2].loc[
            abs(local_df['z'] - g[2]) <= 2]
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
    segmentation_directory = f'../Manual_Segmentations/{directory}/{directory}_segm.mhd'
    # segmentation_directory = f'/work/scratch/westfechtel/segmentations/{directory}'
    sitk_image, np_image = utility.read_image(segmentation_directory)

    try:
        tibial_cartilage = utility.build_3d_cartilage_array(np_image, 4)
        femoral_cartilage = utility.build_3d_cartilage_array(np_image, 3)
    except Exception:
        logging.debug(traceback.format_exc())
        return {**{'dir': directory}, **{}, **{}}

    femoral_vectors = [list(element) for element in femoral_cartilage]
    tibial_vectors = [list(element) for element in tibial_cartilage]

    """
    x, y, z, xy = utility.get_xyz(femoral_vectors)
    df = pd.DataFrame(data={'x': z, 'y': y, 'z': x}, columns=['x', 'y', 'z'])
    center = np.array([df.x.min() + (df.x.max() - df.x.min()) / 2,
                       df.y.min() + (df.y.max() - df.y.min()) / 2,
                       df.z.min() + (df.z.max() - df.z.min()) / 2])

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
                                      sitk_image.GetSpacing()[1]

    keys = set(femoral_thickness.keys())
    for key in keys:
        value = femoral_thickness[key]
        mask = value == 0
        value[mask] = np.nan
        femoral_thickness[key + '.aSD'] = np.nanstd(value)
        femoral_thickness[key + '.aMav'] = np.nanmean(-np.sort(-value)[:math.ceil(len(value) * 0.01)])
        femoral_thickness[key + '.aMiv'] = np.nanmean(np.sort(value)[:math.ceil(len(value) * 0.01)])
        femoral_thickness[key] = np.nanmean(value)
    """
    cwbzl, cwbzr = utility.extract_central_weightbearing_zone(femoral_vectors, tibial_vectors)
    center_left = np.array([cwbzl.x.min() + (cwbzl.x.max() - cwbzl.x.min()) / 2,
                            cwbzl.y.min() + (cwbzl.y.max() - cwbzl.y.min()) / 2,
                            cwbzl.z.min() - (cwbzl.z.max() - cwbzl.z.min()) / 4])

    center_right = np.array([cwbzr.x.min() + (cwbzr.x.max() - cwbzr.x.min()) / 2,
                             cwbzr.y.min() + (cwbzr.y.max() - cwbzr.y.min()) / 2,
                             cwbzr.z.min() - (cwbzr.z.max() - cwbzr.z.min()) / 4])

    lower_mesh_left, upper_mesh_left = utility.build_femoral_meshes(cwbzl)
    lower_mesh_right, upper_mesh_right = utility.build_femoral_meshes(cwbzr)

    left_landmarks = utility.femoral_landmarks(upper_mesh_left.points)
    right_landmarks = utility.femoral_landmarks(upper_mesh_right.points)

    sphere_left = pv.Sphere(center=center_left, radius=1, theta_resolution=60, phi_resolution=60)
    sphere_right = pv.Sphere(center=center_right, radius=1, theta_resolution=60, phi_resolution=60)

    cwbzl['dist'] = np.zeros(cwbzl.shape[0])
    cwbzl['dist'] = cwbzl.apply(lambda l: utility.vector_distance([l.x, l.y, l.z], center_left), axis=1)

    cwbzr['dist'] = np.zeros(cwbzr.shape[0])
    cwbzr['dist'] = cwbzr.apply(lambda l: utility.vector_distance([l.x, l.y, l.z], center_left), axis=1)

    sphere_left.compute_normals(point_normals=True, cell_normals=False, inplace=True)
    sphere_right.compute_normals(point_normals=True, cell_normals=False, inplace=True)

    sphere_left_iter = np.array([[np.nan, np.nan]] * sphere_left.n_points, dtype='object')
    sphere_right_iter = np.array([[np.nan, np.nan]] * sphere_right.n_points, dtype='object')

    for i in range(sphere_left.n_points):
        sphere_left_iter[i, 0] = tuple(sphere_left.points[i])
        sphere_left_iter[i, 1] = tuple(sphere_left['Normals'][i])

    for i in range(sphere_right.n_points):
        sphere_right_iter[i, 0] = tuple(sphere_right.points[i])
        sphere_right_iter[i, 1] = tuple(sphere_right['Normals'][i])

    with Pool() as pool:
        res = pool.starmap(partial(vector_trace, df=cwbzl), iterable=sphere_left_iter)

    res = np.array(res, dtype='object')
    res = res[res != None]

    inner_points = [item[0] for item in res]
    outer_points = [item[1] for item in res]

    n_femur = len(outer_points)

    left_thickness = dict()
    left_thickness['ecLF'] = np.zeros(len(outer_points))
    left_thickness['ccLF'] = np.zeros(len(outer_points))
    left_thickness['icLF'] = np.zeros(len(outer_points))

    for i in range(len(outer_points)):
        label = utility.classify_femoral_point(outer_points[i][:2], left_landmarks, left=True)
        left_thickness[label][i] = utility.vector_distance(inner_points[i], outer_points[i]) * sitk_image.GetSpacing()[
            1]

    with Pool() as pool:
        res = pool.starmap(partial(vector_trace, df=cwbzr), iterable=sphere_right_iter)

    res = np.array(res, dtype='object')
    res = res[res != None]

    inner_points = [item[0] for item in res]
    outer_points = [item[1] for item in res]

    n_femur += len(outer_points)

    right_thickness = dict()
    right_thickness['ecMF'] = np.zeros(len(outer_points))
    right_thickness['ccMF'] = np.zeros(len(outer_points))
    right_thickness['icMF'] = np.zeros(len(outer_points))

    for i in range(len(outer_points)):
        label = utility.classify_femoral_point(outer_points[i][:2], right_landmarks, left=False)
        right_thickness[label][i] = utility.vector_distance(inner_points[i], outer_points[i]) * sitk_image.GetSpacing()[
            1]

    femoral_thickness = dict()
    femoral_thickness.update(left_thickness)
    femoral_thickness.update(right_thickness)

    lpdf, rpdf, adf = utility.extract_anterior_posterior_zones(femoral_vectors, cwbzl, cwbzr)
    ladf, radf = utility.split_anterior_part(adf)

    center_lp = np.array([lpdf.x.min() + (lpdf.x.max() - lpdf.x.min()) / 4,
                          lpdf.y.min() + (lpdf.y.max() - lpdf.y.min()) / 2,
                          lpdf.z.min() + (lpdf.z.max() - lpdf.z.min()) / 2])

    center_rp = np.array([rpdf.x.min() + (rpdf.x.max() - rpdf.x.min()) / 4,
                          rpdf.y.min() + (rpdf.y.max() - rpdf.y.min()) / 2,
                          rpdf.z.min() + (rpdf.z.max() - rpdf.z.min()) / 2])

    center_la = np.array([ladf.x.min() + ((ladf.x.max() - ladf.x.min()) / 4) * 3,
                          ladf.y.min() + (ladf.y.max() - ladf.y.min()) / 2,
                          ladf.z.min() + (ladf.z.max() - ladf.z.min()) / 4])

    center_ra = np.array([radf.x.min() + ((radf.x.max() - radf.x.min()) / 4) * 3,
                          radf.y.min() + (radf.y.max() - radf.y.min()) / 2,
                          radf.z.min() + (radf.z.max() - radf.z.min()) / 4])

    sphere_lp = pv.Sphere(center=center_lp, radius=1, theta_resolution=60, phi_resolution=60)
    sphere_rp = pv.Sphere(center=center_rp, radius=1, theta_resolution=60, phi_resolution=60)
    sphere_la = pv.Sphere(center=center_la, radius=1, theta_resolution=60, phi_resolution=60)
    sphere_ra = pv.Sphere(center=center_ra, radius=1, theta_resolution=60, phi_resolution=60)

    lpdf['dist'] = np.zeros(lpdf.shape[0])
    lpdf['dist'] = lpdf.apply(lambda l: utility.vector_distance([l.x, l.y, l.z], center_lp), axis=1)

    rpdf['dist'] = np.zeros(rpdf.shape[0])
    rpdf['dist'] = rpdf.apply(lambda l: utility.vector_distance([l.x, l.y, l.z], center_rp), axis=1)

    ladf['dist'] = np.zeros(ladf.shape[0])
    ladf['dist'] = ladf.apply(lambda l: utility.vector_distance([l.x, l.y, l.z], center_la), axis=1)

    radf['dist'] = np.zeros(radf.shape[0])
    radf['dist'] = radf.apply(lambda l: utility.vector_distance([l.x, l.y, l.z], center_ra), axis=1)

    sphere_lp.compute_normals(point_normals=True, cell_normals=False, inplace=True)
    sphere_rp.compute_normals(point_normals=True, cell_normals=False, inplace=True)
    sphere_la.compute_normals(point_normals=True, cell_normals=False, inplace=True)
    sphere_ra.compute_normals(point_normals=True, cell_normals=False, inplace=True)

    sphere_lp_iter = np.array([[np.nan, np.nan]] * sphere_lp.n_points, dtype='object')
    sphere_rp_iter = np.array([[np.nan, np.nan]] * sphere_rp.n_points, dtype='object')
    sphere_la_iter = np.array([[np.nan, np.nan]] * sphere_la.n_points, dtype='object')
    sphere_ra_iter = np.array([[np.nan, np.nan]] * sphere_ra.n_points, dtype='object')

    for i in range(sphere_lp.n_points):
        sphere_lp_iter[i, 0] = tuple(sphere_lp.points[i])
        sphere_lp_iter[i, 1] = tuple(sphere_lp['Normals'][i])

    for i in range(sphere_rp.n_points):
        sphere_rp_iter[i, 0] = tuple(sphere_rp.points[i])
        sphere_rp_iter[i, 1] = tuple(sphere_rp['Normals'][i])

    for i in range(sphere_la.n_points):
        sphere_la_iter[i, 0] = tuple(sphere_la.points[i])
        sphere_la_iter[i, 1] = tuple(sphere_la['Normals'][i])

    for i in range(sphere_ra.n_points):
        sphere_ra_iter[i, 0] = tuple(sphere_ra.points[i])
        sphere_ra_iter[i, 1] = tuple(sphere_ra['Normals'][i])

    with Pool() as pool:
        res = pool.starmap(partial(vector_trace, df=lpdf), iterable=sphere_lp_iter)

    res = np.array(res, dtype='object')
    res = res[res != None]

    inner_points = [item[0] for item in res]
    outer_points = [item[1] for item in res]

    n_femur += len(outer_points)

    femoral_thickness['pLF'] = np.zeros(len(outer_points))

    for i in range(len(outer_points)):
        femoral_thickness['pLF'][i] = utility.vector_distance(inner_points[i], outer_points[i]) * \
                                      sitk_image.GetSpacing()[1]

    with Pool() as pool:
        res = pool.starmap(partial(vector_trace, df=rpdf), iterable=sphere_rp_iter)

    res = np.array(res, dtype='object')
    res = res[res != None]

    inner_points = [item[0] for item in res]
    outer_points = [item[1] for item in res]

    n_femur += len(outer_points)

    femoral_thickness['pMF'] = np.zeros(len(outer_points))

    for i in range(len(outer_points)):
        femoral_thickness['pMF'][i] = utility.vector_distance(inner_points[i], outer_points[i]) * \
                                      sitk_image.GetSpacing()[1]

    with Pool() as pool:
        res = pool.starmap(partial(vector_trace, df=ladf), iterable=sphere_la_iter)

    res = np.array(res, dtype='object')
    res = res[res != None]

    inner_points = [item[0] for item in res]
    outer_points = [item[1] for item in res]

    n_femur += len(outer_points)

    femoral_thickness['aLF'] = np.zeros(len(outer_points))

    for i in range(len(outer_points)):
        femoral_thickness['aLF'][i] = utility.vector_distance(inner_points[i], outer_points[i]) * \
                                      sitk_image.GetSpacing()[1]

    with Pool() as pool:
        res = pool.starmap(partial(vector_trace, df=radf), iterable=sphere_ra_iter)

    res = np.array(res, dtype='object')
    res = res[res != None]

    inner_points = [item[0] for item in res]
    outer_points = [item[1] for item in res]

    n_femur += len(outer_points)

    femoral_thickness['aMF'] = np.zeros(len(outer_points))

    for i in range(len(outer_points)):
        femoral_thickness['aMF'][i] = utility.vector_distance(inner_points[i], outer_points[i]) * \
                                      sitk_image.GetSpacing()[1]

    keys = set(femoral_thickness.keys())
    for key in keys:
        value = femoral_thickness[key]
        mask = value == 0
        value[mask] = np.nan
        femoral_thickness[key + '.aSD'] = np.nanstd(value)
        femoral_thickness[key + '.aMav'] = np.nanmean(-np.sort(-value)[:math.ceil(len(value) * 0.01)])
        femoral_thickness[key + '.aMiv'] = np.nanmean(np.sort(value)[:math.ceil(len(value) * 0.01)])
        femoral_thickness[key] = np.nanmean(value)

    # Tibia
    tmp_df = pd.DataFrame(data=tibial_vectors, columns=['x', 'y', 'z'])
    max_z = tmp_df.groupby(['x', 'y']).max()

    # extract max and min vectors by z coordinate
    tmp1 = [np.array(item) for item in max_z.index]
    tmp2 = [item for item in max_z.to_numpy()]
    max_z = np.column_stack((tmp1, tmp2))

    left_landmarks, right_landmarks, split_vector = utility.tibial_landmarks(max_z)

    cluster = KMeans(n_clusters=1, random_state=0).fit(tibial_vectors)
    split_vector = cluster.cluster_centers_[0]
    left_plate, right_plate = utility.split_into_plates(tibial_vectors, split_vector)

    ldf = pd.DataFrame(data=left_plate, columns=['x', 'y', 'z'])
    center = np.array([ldf.x.min() + (ldf.x.max() - ldf.x.min()) / 2,
                       ldf.y.min() + (ldf.y.max() - ldf.y.min()) / 2,
                       ldf.z.max() * 1.25])
    # df.z.min() + (df.z.max() - df.z.min()) / 2])

    # cloud = pv.PolyData(df.to_numpy())
    ldf['dist'] = np.zeros(ldf.shape[0])
    ldf['dist'] = ldf.apply(lambda l: utility.vector_distance([l.x, l.y, l.z], center), axis=1)

    num_sp = 60
    sphere = pv.Sphere(center=center, radius=1, theta_resolution=num_sp, phi_resolution=num_sp)
    sphere.compute_normals(point_normals=True, cell_normals=False, inplace=True)

    sphere_iter = np.array([[np.nan, np.nan]] * sphere.n_points, dtype='object')
    for i in range(sphere.n_points):
        sphere_iter[i][0] = tuple(sphere.points[i])
        sphere_iter[i][1] = tuple(sphere['Normals'][i])

    with Pool() as pool:
        res = pool.starmap(partial(vector_trace, df=ldf), iterable=sphere_iter)

    res = np.array(res, dtype='object')
    res = res[res != None]
    inner_points = [item[0] for item in res]
    outer_points = [item[1] for item in res]

    n_tibia = len(outer_points)

    tibial_thickness = dict()
    
    tibial_thickness['cLT'] = np.zeros(len(outer_points))
    tibial_thickness['aLT'] = np.zeros(len(outer_points))
    tibial_thickness['eLT'] = np.zeros(len(outer_points))
    tibial_thickness['pLT'] = np.zeros(len(outer_points))
    tibial_thickness['iLT'] = np.zeros(len(outer_points))

    for i in range(len(outer_points)):
        label = utility.classify_tibial_point(outer_points[i][:2], left_landmarks, right_landmarks, split_vector)
        if label in set(['cMT', 'aMT', 'eMT', 'pMT', 'iMT']):
            continue

        tibial_thickness[label][i] = utility.vector_distance(outer_points[i], inner_points[i]) * \
                                     sitk_image.GetSpacing()[1]

    rdf = pd.DataFrame(data=right_plate, columns=['x', 'y', 'z'])
    center = np.array([rdf.x.min() + (rdf.x.max() - rdf.x.min()) / 2,
                       rdf.y.min() + (rdf.y.max() - rdf.y.min()) / 2,
                       rdf.z.max() * 1.25])
    # df.z.min() + (df.z.max() - df.z.min()) / 2])

    # cloud = pv.PolyData(df.to_numpy())
    rdf['dist'] = np.zeros(rdf.shape[0])
    rdf['dist'] = rdf.apply(lambda l: utility.vector_distance([l.x, l.y, l.z], center), axis=1)

    num_sp = 60
    sphere = pv.Sphere(center=center, radius=1, theta_resolution=num_sp, phi_resolution=num_sp)
    sphere.compute_normals(point_normals=True, cell_normals=False, inplace=True)

    sphere_iter = np.array([[np.nan, np.nan]] * sphere.n_points, dtype='object')
    for i in range(sphere.n_points):
        sphere_iter[i][0] = tuple(sphere.points[i])
        sphere_iter[i][1] = tuple(sphere['Normals'][i])

    with Pool() as pool:
        res = pool.starmap(partial(vector_trace, df=rdf), iterable=sphere_iter)

    res = np.array(res, dtype='object')
    res = res[res != None]
    inner_points = [item[0] for item in res]
    outer_points = [item[1] for item in res]

    n_tibia += len(outer_points)

    tibial_thickness['cMT'] = np.zeros(len(outer_points))
    tibial_thickness['aMT'] = np.zeros(len(outer_points))
    tibial_thickness['eMT'] = np.zeros(len(outer_points))
    tibial_thickness['pMT'] = np.zeros(len(outer_points))
    tibial_thickness['iMT'] = np.zeros(len(outer_points))

    for i in range(len(outer_points)):
        label = utility.classify_tibial_point(outer_points[i][:2], left_landmarks, right_landmarks, split_vector)
        if label in set(['cLT', 'aLT', 'eLT', 'pLT', 'iLT']):
            continue
            
        tibial_thickness[label][i] = utility.vector_distance(outer_points[i], inner_points[i]) * \
                                     sitk_image.GetSpacing()[1]

    keys = set(tibial_thickness.keys())
    for key in keys:
        value = tibial_thickness[key]
        mask = value == 0
        value[mask] = np.nan
        tibial_thickness[key + '.aSD'] = np.nanstd(value)
        tibial_thickness[key + '.aMav'] = np.nanmean(-np.sort(-value)[:math.ceil(len(value) * 0.01)])
        tibial_thickness[key + '.aMiv'] = np.nanmean(np.sort(value)[:math.ceil(len(value) * 0.01)])
        tibial_thickness[key] = np.nanmean(value)

    logging.info(f'++{n_tibia}++')
    logging.info(f'<<{n_femur}>>')

    return {**{'dir': directory}, **femoral_thickness, **tibial_thickness}


def main():
    logging.basicConfig(filename='logs/sphere_default.log', encoding='utf-8',
                        level=logging.DEBUG, filemode='w')
    logging.debug('Entered main.')

    try:

        filehandler = logging.FileHandler(f'logs/sphere.log', mode='w')
        filehandler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        root.addHandler(filehandler)

        dirs = utility.get_subdirs(None)

        dirs = dirs[:50]

        res = np.empty(len(dirs), dtype='object')
        t = time()
        for i, directory in enumerate(dirs):
            tt = time()
            try:
                res[i] = fun(directory)
                logging.info(f'::{time() - tt}::')
                if i % 10 == 0:
                    logging.debug(f'Iteration #{i}')
                    print(f'Iteration #{i}')
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
        df.to_pickle(f'out/sphere')
    except Exception as e:
        logging.debug(traceback.format_exc())
        logging.debug(sys.argv)

    logging.info(f'total execution time: {time() - t}')


if __name__ == '__main__':
    main()
