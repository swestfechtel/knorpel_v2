import functools
import math
import os
import string
import traceback
import warnings
import pickle
from collections import defaultdict

import SimpleITK as sitk
import numpy as np
import pandas as pd
import pyvista as pv
from scipy import stats
from sklearn.cluster import KMeans


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def get_subdirs(chunk):
    f = open('/work/scratch/westfechtel/knorpel_v2/stage_01.pickle', 'rb')
    files = pickle.load(f)
    return [f.name for f in os.scandir('/images/Shape/Medical/Knees/OAI/Manual_Segmentations/') if
            f.is_dir() and f.name in chunk and int(f.name) in files]


"""
def get_subdirs(chunk):
    return np.load(f'/work/scratch/westfechtel/segmented_chunks/{chunk}.npy', allow_pickle=True)
"""


@deprecated
def get_x_y(array: list):
    """
    Extracts x and y values from an array [[x1, y1], [x2, y2], ...] sorted by x.

    :param array:
    :return: [x1, x2, ...], [y1, y2, ...] | x1 < x2 < ...
    """
    array += [None]
    tmp = np.array(array, dtype='object')
    tmp = tmp[tmp != None]

    try:
        tmp = np.sort(tmp)
        x = [item[0] for item in tmp]
        y = [item[1] for item in tmp]
    except TypeError as e:
        return None, [tmp, traceback.format_exc()]

    return x, y


def vector_distance(x: np.array, y: np.array) -> float:
    """
    Calculates the distance between two vectors.

    :param x: vector x
    :param y: vector y
    :return: the distance between x and y
    """
    tmp = x - y
    return math.sqrt(sum([item ** 2 for item in tmp]))


def split_into_plates(vectors, split_vector) -> [list, list]:
    """
    Splits the tibial or femoral cartilage into two plates.

    :param vectors: all vectors making up the cartilage
    :param split_vector: the vector to split by
    :return: two lists containing vectors making up the left and right plates, respectively
    """
    left_plate = list()
    right_plate = list()

    for vector in vectors:
        if vector[1] <= split_vector[1]:  # if y-coord(vector) is lower than y-coord(split_vector)
            left_plate.append(vector)
        else:
            right_plate.append(vector)

    return left_plate, right_plate


def is_in_ellipse(vector, center, radius) -> bool:
    """
    Checks whether a vector lies within a circle.

    :param vector: The vector to check
    :param center: The center of the circle
    :param radius: The radius of the circle
    :return: true if the vector lies within the circle, false otherwise
    """
    return vector_distance(vector, center) <= radius


def calculate_ellipse(vectors, center) -> [int, list]:
    """
    Calculates a circle which covers 20% of the total surface area.

    :param vectors: All vectors making up the surface area
    :param center: The center of gravity of the vectors
    :return: Radius of the circle, all vectors within the circle
    """
    r = 20
    points_in_ellipse = list()
    num_it = 0

    while not (abs(len(points_in_ellipse) / len(vectors) - .2) < .01):
        if num_it > 100:
            break

        points_in_ellipse = list()
        for point in vectors:
            if is_in_ellipse(point[:2], center[:2], r):
                points_in_ellipse.append(point)

        if len(points_in_ellipse) / len(vectors) > 0.2:
            r /= 2
        else:
            r += .5

        num_it += 1

    return r, points_in_ellipse


def get_plate_corners(plate) -> [int, int, int, int]:
    """
    Gets the four corner vectors a, b, c and d of a tibia plate.

    :param plate: the tibia plate from which to extract the corner vectors
    :return: the corner vectors
    """
    xmin = min([item[0] for item in plate])
    xmax = max([item[0] for item in plate])
    ymin = min([item[1] for item in plate])
    ymax = max([item[1] for item in plate])

    a = [xmin, ymin]
    b = [xmax, ymin]
    c = [xmax, ymax]
    d = [xmin, ymax]

    return a, b, c, d


def get_femoral_thirds(plate) -> [int, int]:
    """
    Splits a plate into three subregions, each one containing 33% of all points.

    :param plate: all vectors making up the plate
    :return: two split indices for the y-axis
    """
    ymin = min(item[1] for item in plate)
    ymax = max(item[1] for item in plate)
    yrange = ymax - ymin
    first_split = ymin + int(yrange / 3)
    second_split = ymin + 2 * int(yrange / 3)

    points_in_first_third = list()
    points_in_second_third = list()
    num_it = 0

    while not (abs(len(points_in_first_third) / len(plate) - .33) < .02):
        if num_it > 30:
            break

        points_in_first_third = list()
        for point in plate:
            if point[1] < first_split:
                points_in_first_third.append(point)

        if len(points_in_first_third) / len(plate) > 0.33:
            first_split -= 1
        else:
            first_split += 1

        num_it += 1

    num_it = 0

    while not (abs(len(points_in_second_third) / len(plate) - .33) < .02):
        if num_it > 30:
            break

        points_in_second_third = list()
        for point in plate:
            if first_split <= point[1] < second_split:
                points_in_second_third.append(point)

        if len(points_in_second_third) / len(plate) > 0.33:
            second_split -= 1
        else:
            second_split += 1

        num_it += 1

    return first_split, second_split


def classify_tibial_point(vector, left_regions, right_regions, split_vector) -> string:
    """
    Classifies a vector's subregion from the tibial cartilage according to its position.

    :param vector: the vector to classify
    :param left_regions: corners a, b, c, d as well as radius and center of gravity of the left plate
    :param right_regions: corners a, b, c, d as well as radius and center of gravity of the right plate
    :param split_vector: the vector with which to split into left and right plate
    :return: a classification label for the vector
    """

    al, bl, cl, dl, l_rad, l_center = left_regions
    ar, br, cr, dr, r_rad, r_center = right_regions
    # print(vector)
    if vector[1] > split_vector[1]: #right - lateral
        if is_in_ellipse(vector, r_center[:2], r_rad):
            return 'cLT'

        ac = np.array(cr) - np.array(ar)
        db = np.array(br) - np.array(dr)
        xc = np.array(cr) - np.array([vector[0], vector[1]])
        xb = np.array(br) - np.array([vector[0], vector[1]])

        x_cross_ac = np.cross(ac, xc)
        x_cross_db = np.cross(db, xb)
        if x_cross_ac > 0:
            if x_cross_db > 0:
                return 'iLT'
            else:
                return 'pLT'
        else:
            if x_cross_db > 0:
                return 'aLT'
            else:
                return 'eLT'
    else: # left - medial

        if is_in_ellipse(vector, l_center[:2], l_rad):
            return 'cMT'

        ac = np.array(cl) - np.array(al)
        db = np.array(bl) - np.array(dl)
        xc = np.array(cl) - np.array([vector[0], vector[1]])
        xb = np.array(bl) - np.array([vector[0], vector[1]])

        x_cross_ac = np.cross(ac, xc)
        x_cross_db = np.cross(db, xb)
        if x_cross_ac > 0:
            if x_cross_db > 0:
                return 'eMT'
            else:
                return 'pMT'
        else:
            if x_cross_db > 0:
                return 'aMT'
            else:
                return 'iMT'


"""
def classify_femoral_point(vector, left_regions, right_regions, split_vector) -> string:
    ""
    Classifies a vector's subregion from the femoral cartilage according to its position.

    :param vector: the vector to classify
    :param left_regions: split indices for the left plate
    :param right_regions: split indices for the right plate
    :param split_vector: the vector with which to split into left and right plate
    :return: a classification label for the vector
    ""
    l_first_split, l_second_split = left_regions
    r_first_split, r_second_split = right_regions

    if vector[1] < split_vector[1]:
        if vector[1] < l_first_split:
            return 'ecLF'
        elif l_first_split <= vector[1] <= l_second_split:
            return 'ccLF'
        else:
            return 'icLF'
    else:
        if vector[1] < r_first_split:
            return 'icMF'
        elif r_first_split <= vector[1] <= r_second_split:
            return 'ccMF'
        else:
            return 'ecMF'
"""


def classify_femoral_point(vector, landmarks, left=True):
    first_split, second_split = landmarks
    if not left:
        if vector[1] < first_split:
            return 'icLF'
        elif first_split <= vector[1] <= second_split:
            return 'ccLF'
        else:
            return 'ecLF'
    else:
        if vector[1] < first_split:
            return 'ecMF'
        elif first_split <= vector[1] <= second_split:
            return 'ccMF'
        else:
            return 'icMF'


def read_image(path: string) -> [sitk.Image, np.array]:
    """
    Reads a mhd/nifti file into a numpy array

    :param path: path to the file
    :return: numpy array representation of the mhd/nifti file
    """
    sitk_image = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(sitk_image)
    return sitk_image, arr


"""
def build_3d_cartilage_array(image, color_code=3) -> np.array:
    ""
    Extracts all vectors making up a cartilage according to color coding and packs them into a numpy array.

    :param image: the 3d image representation of a mri scan
    :param color_code: the color coding of the cartilage to extract
    :return: a numpy array containing all vectors making up the cartilage
    ""
    cartilage = np.where(image == color_code, image, 0)

    ys = np.sort(np.where(cartilage == color_code)[0])
    xs = np.sort(np.where(cartilage == color_code)[1])
    zs = np.sort(np.where(cartilage == color_code)[2])

    cartilage = cartilage[ys[0]:ys[-1], xs[0]:xs[-1], zs[0]:zs[-1]]

    X = [0] * (cartilage.shape[0] * cartilage.shape[1] * cartilage.shape[2])

    i = 0
    for y in range(cartilage.shape[0]):
        for x in range(cartilage.shape[1]):
            for z in range(cartilage.shape[2]):
                if cartilage[y][x][z] == color_code:
                    X[i] = [x, y, z]
                    i += 1

    tmp = np.array(X, dtype=object)
    tmp = tmp[tmp != 0]
    return tmp
"""


def build_3d_cartilage_array(image, color_code=3) -> np.array:
    """
    cartilage = [0] * (image.shape[0] * image.shape[1] * image.shape[2])
    indx = 0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for z in range(image.shape[2]):
                if image[y,x,z] == color_code:
                    cartilage[indx] = [x, y, z]
                    indx += 1

    cartilage = np.array(cartilage, dtype=object)
    cartilage = cartilage[cartilage != 0]
    return np.array([list(element) for element in cartilage])
    """
    return np.argwhere(image == color_code)[:, [2, 0, 1]]


def extract_central_weightbearing_zone(femoral_cartilage, tibial_cartilage):
    df = pd.DataFrame(data=tibial_cartilage, columns=['x', 'y', 'z'])
    max_z = df.groupby(['x', 'y']).max()

    tmp1 = [np.array(item) for item in max_z.index]
    tmp2 = [item for item in max_z.to_numpy()]
    max_z = np.column_stack((tmp1, tmp2))

    dd = defaultdict(list)
    left_tibial_regions, right_tibial_regions, split_vector = tibial_landmarks(max_z)
    for v in max_z:
        vector = np.array(v)
        label = classify_tibial_point(vector[:2], left_tibial_regions, right_tibial_regions, split_vector)
        dd[label].append(vector)

    # extract external, central and internal subregions for each tibial cartilage plate
    wbr = np.vstack((np.array(dd['eLT']), np.array(dd['cLT']), np.array(dd['iLT'])))
    wbl = np.vstack((np.array(dd['iMT']), np.array(dd['cMT']), np.array(dd['eMT'])))

    # pack them into a dataframe and cut the dimensions along the x axis to the min, max x expansion of the respective central subregion
    tdfr = pd.DataFrame(data={'x': wbr[:, 0], 'y': wbr[:, 1], 'z': wbr[:, 2]})
    tdfl = pd.DataFrame(data={'x': wbl[:, 0], 'y': wbl[:, 1], 'z': wbl[:, 2]})
    tdfr = tdfr.loc[tdfr['x'] < max(np.array(dd['cLT'])[:, 0])].loc[tdfr['x'] > min(np.array(dd['cLT'])[:, 0])]
    tdfl = tdfl.loc[tdfl['x'] < max(np.array(dd['cMT'])[:, 0])].loc[tdfl['x'] > min(np.array(dd['cMT'])[:, 0])]

    # pack the femoral cartilage into a dataframe and split into two plates
    df = pd.DataFrame(data=femoral_cartilage, columns=['x', 'y', 'z'])
    fdfl = df.loc[df['y'] < df['y'].mean()]
    fdfr = df.loc[df['y'] >= df['y'].mean()]

    # for each femoral plate, extract the central weight bearing zone according to the min, max x, y dimensions of the previously extracted tibial subregions
    # aka extract the part of the femoral cartilage that is in contact with the tibial cartilage when standing
    # left side first
    minx = tdfl['x'].min()
    maxx = tdfl['x'].max()
    miny = tdfl['y'].min()
    maxy = tdfl['y'].max()
    cwbzl = fdfl.loc[fdfl['x'] > minx].loc[fdfl['x'] < maxx].loc[fdfl['y'] > miny].loc[fdfl['y'] < maxy]
    cwbzl = cwbzl[(np.abs(stats.zscore(cwbzl)) < 2).all(axis=1)]

    # then right side
    minx = tdfr['x'].min()
    maxx = tdfr['x'].max()
    miny = tdfr['y'].min()
    maxy = tdfr['y'].max()
    cwbzr = fdfr.loc[fdfr['x'] > minx].loc[fdfr['x'] < maxx].loc[fdfr['y'] > miny].loc[fdfr['y'] < maxy]
    cwbzr = cwbzr[(np.abs(stats.zscore(cwbzr)) < 2).all(axis=1)]

    return cwbzl, cwbzr


def extract_anterior_posterior_zones(femoral_cartilage, cwbzl, cwbzr):
    df = pd.DataFrame(data=femoral_cartilage, columns=['x', 'y', 'z'])

    left_plate = df.loc[df['y'] < df['y'].mean()]
    right_plate = df.loc[df['y'] >= df['y'].mean()]

    lpdf = left_plate.loc[left_plate['x'] > cwbzl['x'].max()]
    rpdf = right_plate.loc[right_plate['x'] > cwbzr['x'].max()]

    lpdf = lpdf[['z', 'y', 'x']] # rotate
    lpdf.columns = ['x', 'y', 'z']

    rpdf = rpdf[['z', 'y', 'x']]
    rpdf.columns = ['x', 'y', 'z']

    ladf = left_plate.loc[left_plate['x'] < cwbzl['x'].min()]
    radf = right_plate.loc[right_plate['x'] < cwbzr['x'].min()]
    adf = pd.concat([ladf, radf])

    return lpdf, rpdf, adf
    # return lpdf, rpdf, ladf, radf


def split_anterior_part(adf):
    a_lower_mesh, a_upper_mesh = build_tibial_meshes(adf.to_numpy())
    df = pd.DataFrame(a_upper_mesh.points, columns=['x', 'y', 'z'])

    tmp = df.groupby(by='x').min().reset_index()[['x', 'z']]
    indices = []
    for index, row in tmp.iterrows():
        if (a := df.loc[df['x'] == row['x']].loc[df['z'] == row['z']]).shape[0] > 0:
            indices.append(list(a.index))

    indices = np.hstack((indices))
    valley = df.iloc[indices]
    valley = valley[np.abs(stats.zscore(valley['y'])) < 1]
    left_part = adf.loc[adf['y'] < valley['y'].median()]
    right_part = adf.loc[adf['y'] > valley['y'].median()]

    return left_part, right_part


def get_xyz(regions):
    x = [x[0] for x in regions]
    y = [x[1] for x in regions]
    xy = [[x[0], x[1]] for x in regions]
    z = [x[2] for x in regions]
    return x, y, z, xy


def build_tibial_meshes(vectors: list) -> [pv.core.pointset.PolyData, pv.core.pointset.PolyData]:
    """
    Builds upper and lower delaunay mesh of a tibial cartilage volume.

    Groups all vectors by (x, y) and adds the vector (x, y, max(z)) to the upper mesh, and
    the vector (x, y, min(z)) to the lower mesh, for every pair (x, y).

    :param vectors: An array of three-dimensional vectors (x, y, z) making up the cartilage volume
    :return: A lower and upper mesh, by z coordinate
    """
    df = pd.DataFrame(data=vectors, columns=['x', 'y', 'z'])
    max_z = df.groupby(['x', 'y']).max()
    min_z = df.groupby(['x', 'y']).min()

    # extract max and min vectors by z coordinate
    tmp1 = [np.array(item) for item in max_z.index]
    tmp2 = [item for item in max_z.to_numpy()]
    max_z = np.column_stack((tmp1, tmp2))

    tmp1 = [np.array(item) for item in min_z.index]
    tmp2 = [item for item in min_z.to_numpy()]
    min_z = np.column_stack((tmp1, tmp2))

    # build two point clouds for the upper and lower vectors
    upper_cloud = pv.PolyData(max_z)
    lower_cloud = pv.PolyData(min_z)

    # build polygon meshes for both point clouds using delaunay
    lower_mesh = lower_cloud.delaunay_2d()
    upper_mesh = upper_cloud.delaunay_2d()

    return lower_mesh, upper_mesh


def build_femoral_meshes(vectors: pd.DataFrame) -> [pv.core.pointset.PolyData, pv.core.pointset.PolyData]:
    """
    Builds upper and lower delaunay mesh of a tibial cartilage volume.

    Groups all vectors by (x, y) and adds the vector (x, y, max(z)) to the upper mesh, and
    the vector (x, y, min(z)) to the lower mesh, for every pair (x, y).

    :param vectors: An array of three-dimensional vectors (x, y, z) making up the cartilage volume
    :return: A lower and upper mesh, by z coordinate
    """
    df = vectors
    max_z = df.groupby(['x', 'y']).max()
    min_z = df.groupby(['x', 'y']).min()

    # extract max and min vectors by z coordinate
    tmp1 = [np.array(item) for item in max_z.index]
    tmp2 = [item for item in max_z.to_numpy()]
    max_z = np.column_stack((tmp1, tmp2))

    tmp1 = [np.array(item) for item in min_z.index]
    tmp2 = [item for item in min_z.to_numpy()]
    min_z = np.column_stack((tmp1, tmp2))

    # build two point clouds for the upper and lower vectors
    upper_cloud = pv.PolyData(max_z)
    lower_cloud = pv.PolyData(min_z)

    # build polygon meshes for both point clouds using delaunay
    lower_mesh = lower_cloud.delaunay_2d()
    upper_mesh = upper_cloud.delaunay_2d()

    return lower_mesh, upper_mesh


def tibial_landmarks(vectors) -> [list, list, np.ndarray]:
    """
    Computes the landmarks of a tibial cartilage volume which can be used to split the volume into subregions.

    Splits the volume into a left and a right plate using KMeans clustering.
    For each plate, calculates an ellipse which contains ~20% of the plate's vectors.
    Gets the four corners of each plate.

    :param vectors: Array-like structure containing the vectors making up the cartilage volume
    :return: A list containing the landmarks of the left plate, right plate and the split vector between the plates.
    Landmark format is [lower left corner, lower right corner, upper right corner, upper left corner, circle radius, circle center]
    """
    cluster = KMeans(n_clusters=1, random_state=0).fit(vectors)
    split_vector = cluster.cluster_centers_[0]
    left_plate, right_plate = split_into_plates(vectors, split_vector)

    left_plate_cog = KMeans(n_clusters=1, random_state=0).fit([x[:2] for x in left_plate]).cluster_centers_[0]
    right_plate_cog = KMeans(n_clusters=1, random_state=0).fit([x[:2] for x in right_plate]).cluster_centers_[0]

    left_plate_radius, left_plate_circle = calculate_ellipse(left_plate, left_plate_cog)
    right_plate_radius, right_plate_circle = calculate_ellipse(right_plate, right_plate_cog)

    la, lb, lc, ld = get_plate_corners(plate=left_plate)
    ra, rb, rc, rd = get_plate_corners(plate=right_plate)

    left_tibial_landmarks = [la, lb, lc, ld, left_plate_radius, left_plate_cog]
    right_tibial_landmarks = [ra, rb, rc, rd, right_plate_radius, right_plate_cog]

    return left_tibial_landmarks, right_tibial_landmarks, split_vector


"""
def femoral_landmarks(vectors) -> [list, list, np.ndarray]:
    ""
    Computes the landmarks of a femoral cartilage volume which can be used to split the volume into subregions.

    Splits the volume into a left and right plate using KMeans clustering.
    For each plate, splits the plate into equal thirds along the y axis.

    :param vectors: Array-like structure containing the vectors making up the cartilage volume
    :return: A list containing the landmarks of the left plate, right plate and the split vector between the plates.
    Landmark format is [first split coordinate, second split coordinate] such that #points left of first split ~=
    #points between first and second split ~= #points right of second split
    ""
    cluster = KMeans(n_clusters=1, random_state=0).fit(vectors)
    split_vector = cluster.cluster_centers_[0]
    left_plate, right_plate = split_into_plates(vectors, split_vector)

    first_split, second_split = get_femoral_thirds(left_plate)
    left_femoral_landmarks = [first_split, second_split]

    first_split, second_split = get_femoral_thirds(right_plate)
    right_femoral_landmarks = [first_split, second_split]

    return left_femoral_landmarks, right_femoral_landmarks, split_vector
"""


def femoral_landmarks(vectors):
    first_split, second_split = get_femoral_thirds(vectors)
    return [first_split, second_split]


def calculate_distance(lower_normals, lower_mesh, upper_mesh, sitk_image, left_landmarks, right_landmarks, split_vector,
                       dictionary, femur=False):
    """
    Calculates the distances between an upper and lower delaunay mesh using ray tracing.

    For each vector in the lower mesh, finds the corresponding vector from the upper mesh and calculates the
    vector distance between the two. Also labels the vector according to its corresponding subregion.

    :param lower_normals: The normal vectors of the lower delaunay mesh
    :param lower_mesh: The lower delaunay mesh
    :param upper_mesh: The upper delaunay mesh
    :param sitk_image: Meta information containing spacing of the scan
    :param left_landmarks: The subregion landmarks of the left cartilage plate
    :param right_landmarks: The subregion landmarks of the right cartilage plate
    :param split_vector: The split vector between the two plates
    :param dictionary: A dictionary containing the appropriate subregion keys
    :param femur: Whether the meshes belong to a femoral cartilage volume
    :return: the lower normals with their corresponding distance measures, and a dictionary containing a list
    of distances for each subregion label
    """
    lower_normals['distances'] = np.zeros(lower_mesh.n_points)
    for i in range(lower_normals.n_points):
        v = lower_mesh.points[i]
        vec = lower_normals['Normals'][i] * lower_normals.length
        v0 = v - vec
        v1 = v + vec
        iv, ic = upper_mesh.ray_trace(v0, v1, first_point=True)
        dist = np.sqrt(np.sum((iv - v) ** 2)) * sitk_image.GetSpacing()[1]
        lower_normals['distances'][i] = dist
        if not femur:
            label = classify_tibial_point(v[:2], left_landmarks, right_landmarks, split_vector)
        else:
            label = classify_femoral_point(v[:2], left_landmarks, right_landmarks, split_vector)

        dictionary[label][i] = dist

    return lower_normals, dictionary


def calculate_femoral_thickness(lower_normals, lower_mesh, upper_mesh, sitk_image, landmarks, dictionary, left=True):
    lower_normals['distances'] = np.zeros(lower_mesh.n_points)
    for i in range(lower_normals.n_points):
        v = lower_mesh.points[i]
        vec = lower_normals['Normals'][i] * lower_normals.length
        v0 = v - vec
        v1 = v + vec
        iv, ic = upper_mesh.ray_trace(v0, v1, first_point=True)
        dist = np.sqrt(np.sum((iv - v) ** 2)) * sitk_image.GetSpacing()[1]
        lower_normals['distances'][i] = dist
        label = classify_femoral_point(v[:2], landmarks, left)

        dictionary[label][i] = dist

    return lower_normals, dictionary


def calculate_distance_without_classification(lower_normals, lower_mesh, upper_mesh, sitk_image):
    lower_normals['distances'] = np.zeros(lower_mesh.n_points)
    for i in range(lower_normals.n_points):
        v = lower_mesh.points[i]
        vec = lower_normals['Normals'][i] * lower_normals.length
        v0 = v - vec
        v1 = v + vec
        iv, ic = upper_mesh.ray_trace(v0, v1, first_point=True)
        dist = np.sqrt(np.sum((iv - v) ** 2)) * sitk_image.GetSpacing()[1]
        lower_normals['distances'][i] = dist

    return lower_normals


@deprecated
def isolate_cartilage(layer: np.array, color_code: int = 3) -> np.array:
    """
    Crops a single layer to only include x, y ranges where cartilage is present.

    :param layer: numpy array representation of the layer to crop
    :param color_code: color code of the cartilage segment
    :return: cropped numpy array representation of the layer
    """
    cartilage = np.where(layer == color_code, layer, 0)
    cartilage = cartilage[:, np.any(cartilage == color_code, axis=0)]

    return cartilage[np.any(cartilage == color_code, axis=1), :]


@deprecated
def build_array(layer: np.array, isolate: bool = False, isolator: int = 3) -> list:
    """
    Builds an array for a single layer.

    :param layer: numpy array representation of the layer
    :param isolate: whether to isolate cartilage points
    :param isolator: the color value of the points to isolate
    :return: descriptive values X, class values Y
    """
    X = [None] * (layer.shape[0] * layer.shape[1])

    i = 0
    for y in range(layer.shape[0]):
        for x in range(layer.shape[1]):
            if not isolate:
                X[i] = [x, y]
                i += 1
            else:
                if layer[y][x] == isolator:
                    X[i] = [x, y]
                    i += 1

    return X
