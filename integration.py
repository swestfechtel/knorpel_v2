import math
import sys
import traceback
import utility
import logging
import meshing

import numpy as np
import pandas as pd

from multiprocessing import Pool
from time import time


def fit_function(df, degree):
    try:
        z = np.polyfit(df.index.to_numpy(), df.y.to_numpy(), degree)
        return np.poly1d(z)
    except np.linalg.LinAlgError:
        return None
    except TypeError:
        return None
    except ValueError:
        return None
    except Exception:
        return None


def function_for_pool(directory):
    segmentation_directory = f'/images/Shape/Medical/Knees/OAI/Manual_Segmentations/{directory}/{directory}_segm.mhd'
    # segmentation_directory = f'/work/scratch/westfechtel/segmentations/{directory}'
    sitk_image, np_image = utility.read_image(segmentation_directory)
    try:
        femoral_cartilage = utility.build_3d_cartilage_array(np_image, 3)
        tibial_cartilage = utility.build_3d_cartilage_array(np_image, 4)
    except Exception:
        logging.error(traceback.format_exc())
        return {**{'dir': directory}, **{}, **{}}

    femoral_vectors = [list(elem) for elem in femoral_cartilage]
    tibial_vectors = [list(elem) for elem in tibial_cartilage]

    left_portion, middle_portion, right_portion = meshing.split_femoral_volume(femoral_vectors)
    left_outer, left_inner = meshing.build_portion_delaunay(left_portion)
    middle_outer, middle_inner = meshing.build_portion_delaunay(middle_portion)
    right_outer, right_inner = meshing.build_portion_delaunay(right_portion)
    outer_cloud = meshing.combine_to_cloud(left_outer, middle_outer, right_outer)
    left_femoral_regions, right_femoral_regions, femoral_split_vector = utility.femoral_landmarks(outer_cloud.to_numpy())

    _, upper_tibial_mesh = utility.build_tibial_meshes(tibial_vectors)
    left_tibial_regions, right_tibial_regions, tibial_split_vector = utility.tibial_landmarks(upper_tibial_mesh.points)

    femoral_thickness = dict()
    femoral_thickness['ecLF'] = np.zeros(1)
    femoral_thickness['ccLF'] = np.zeros(1)
    femoral_thickness['icLF'] = np.zeros(1)
    femoral_thickness['icMF'] = np.zeros(1)
    femoral_thickness['ccMF'] = np.zeros(1)
    femoral_thickness['ecMF'] = np.zeros(1)
    for i in range(len(np_image)):
        layer = np_image[i]
        layer = utility.isolate_cartilage(layer, 3)
        if len(layer) == 0:
            continue

        femoral_cartilage = utility.build_array(layer, isolate=True, isolator=3)
        x, y = utility.get_x_y(femoral_cartilage)
        df = pd.DataFrame(data={'x': x, 'y': y})
        max_y = df.groupby(by='x').max()
        min_y = df.groupby(by='x').min()
        upper_fun = fit_function(max_y, 4)
        lower_fun = fit_function(min_y, 4)
        if upper_fun is None or lower_fun is None:
            continue

        # upper_integration = integrate.quad(upper_fun, min(x), max(x))
        # lower_integration = integrate.quad(lower_fun, min(x), max(x))
        # res = (upper_integration[0] - lower_integration[0]) / max(x)
        # total_femoral_thickness += res
        for j in range(min(x), max(x)):
            label = utility.classify_femoral_point([j, i], left_femoral_regions, right_femoral_regions, femoral_split_vector)
            dist = (upper_fun(j) - lower_fun(j)) * sitk_image.GetSpacing()[1]
            femoral_thickness[label] = np.append(femoral_thickness[label], dist)

        # num_it += 1

    # mean_femoral_thickness = (total_femoral_thickness / num_it) * sitk_image.GetSpacing()[1]
    keys = set(femoral_thickness.keys())
    for key in keys:
        value = femoral_thickness[key]
        mask = value == 0
        value[mask] = np.nan
        femoral_thickness[key + '.aSD'] = np.nanstd(value)
        femoral_thickness[key + '.aMav'] = np.nanmean(-np.sort(-value)[:math.ceil(len(value) * 0.01)])
        femoral_thickness[key + '.aMiv'] = np.nanmean(np.sort(value)[:math.ceil(len(value) * 0.01)])
        femoral_thickness[key] = np.nanmean(value)

    num_it = 0

    tibial_thickness = dict()
    tibial_thickness['cLT'] = np.zeros(1)
    tibial_thickness['aLT'] = np.zeros(1)
    tibial_thickness['eLT'] = np.zeros(1)
    tibial_thickness['pLT'] = np.zeros(1)
    tibial_thickness['iLT'] = np.zeros(1)
    tibial_thickness['cMT'] = np.zeros(1)
    tibial_thickness['aMT'] = np.zeros(1)
    tibial_thickness['eMT'] = np.zeros(1)
    tibial_thickness['pMT'] = np.zeros(1)
    tibial_thickness['iMT'] = np.zeros(1)
    for i in range(len(np_image)):
        layer = np_image[i]
        layer = utility.isolate_cartilage(layer, 4)
        if len(layer) == 0:
            continue

        tibial_cartilage = utility.build_array(layer, isolate=True, isolator=4)
        x, y = utility.get_x_y(tibial_cartilage)
        df = pd.DataFrame(data={'x': x, 'y': y})
        max_y = df.groupby(by='x').max()
        min_y = df.groupby(by='x').min()
        upper_fun = fit_function(max_y, 4)
        lower_fun = fit_function(min_y, 4)
        if upper_fun is None or lower_fun is None:
            continue

        # upper_integration = integrate.quad(upper_fun, min(x), max(x))
        # lower_integration = integrate.quad(lower_fun, min(x), max(x))
        # res = (upper_integration[0] - lower_integration[0]) / max(x)
        # total_tibial_thickness += res
        # num_it += 1
        for j in range(min(x), max(x)):
            label = utility.classify_tibial_point([j, i], left_tibial_regions, right_tibial_regions, tibial_split_vector)
            dist = (upper_fun(j) - lower_fun(j)) * sitk_image.GetSpacing()[1]
            tibial_thickness[label] = np.append(tibial_thickness[label], dist)

    keys = set(tibial_thickness.keys())
    for key in keys:
        value = tibial_thickness[key]
        mask = value == 0
        value[mask] = np.nan
        tibial_thickness[key + '.aSD'] = np.nanstd(value)
        tibial_thickness[key + '.aMav'] = np.nanmean(-np.sort(-value)[:math.ceil(len(value) * 0.01)])
        tibial_thickness[key + '.aMiv'] = np.nanmean(np.sort(value)[:math.ceil(len(value) * 0.01)])
        tibial_thickness[key] = np.nanmean(value)

    # mean_tibial_thickness = (total_tibial_thickness / num_it) * sitk_image.GetSpacing()[1]

    # return {'dir': directory, 'fem': mean_femoral_thickness, 'tib': mean_tibial_thickness}
    return {**{'dir': directory}, **femoral_thickness, **tibial_thickness}


def main():
    logging.basicConfig(filename='/work/scratch/westfechtel/pylogs/integration/integration_default.log', encoding='utf-8', level=logging.DEBUG, filemode='w')
    logging.info('Entered main')
    try:
        assert len(sys.argv) == 2
        chunk = np.load(f'/work/scratch/westfechtel/chunks/{sys.argv[1]}.npy')
        # chunk = sys.argv[1]

        filehandler = logging.FileHandler(f'/work/scratch/westfechtel/pylogs/integration/{sys.argv[1]}.log', mode='w')
        filehandler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        root.addHandler(filehandler)
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        files = utility.get_subdirs(chunk)
        logging.info(f'Using chunk {sys.argv[1]} with length {len(files)}.')

        t = time()
        with Pool() as pool:
            res = pool.map(func=function_for_pool, iterable=files)

        logging.info(f'Elapsed time: {time() - t}')
        df = pd.DataFrame.from_dict(res)
        df.index = df['dir']
        df = df.drop('dir', axis=1)
        # df.to_excel('integration.xlsx')
        df.to_pickle(f'/work/scratch/westfechtel/manpickles/twodim_integration/{sys.argv[1]}')
    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(sys.argv)


if __name__ == '__main__':
    main()
