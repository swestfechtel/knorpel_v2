import SimpleITK as sitk
import numpy as np
import pyvista as pv
import pandas as pd
import utility
import meshing
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path


def save_screenshots():
    for subdir in tqdm(Path('../Failed_region_splitting').iterdir()):
        for ss_dir in subdir.iterdir():
            if ss_dir.name == 'mask.nii.gz':
                sitk_image, np_image = utility.read_image(f'../Failed_region_splitting/{subdir.name}/{ss_dir.name}')
                femoral_cartilage = utility.build_3d_cartilage_array(np_image, 3)
                tibial_cartilage = utility.build_3d_cartilage_array(np_image, 4)
                femoral_vectors = [list(element) for element in femoral_cartilage]
                tibial_vectors = [list(element) for element in tibial_cartilage]
                cwbzl, cwbzr = utility.extract_central_weightbearing_zone(femoral_vectors, tibial_vectors)
                lpdf, rpdf, adf = utility.extract_anterior_posterior_zones(femoral_vectors, cwbzl, cwbzr)
                ladf, radf = utility.split_anterior_part(adf)

                df = pd.DataFrame(data=tibial_cartilage, columns=['x', 'y', 'z'])
                max_z = df.groupby(['x', 'y']).max()

                tmp1 = [np.array(item) for item in max_z.index]
                tmp2 = [item for item in max_z.to_numpy()]
                max_z = np.column_stack((tmp1, tmp2))

                dd = defaultdict(list)
                left_tibial_regions, right_tibial_regions, split_vector = utility.tibial_landmarks(max_z)
                for v in max_z:
                    vector = np.array(v)
                    label = utility.classify_tibial_point(vector[:2], left_tibial_regions, right_tibial_regions, split_vector)
                    dd[label].append(vector)

                p = pv.Plotter(off_screen=True, window_size=[2560, 1440])

                p.add_mesh(pv.PolyData(dd['iMT']), color='blue', point_size=20)
                p.add_mesh(pv.PolyData(dd['aMT']), color='#03b3ff', point_size=15)
                p.add_mesh(pv.PolyData(dd['eMT']), color='#0bff03', point_size=15)
                p.add_mesh(pv.PolyData(dd['pMT']), color='yellow', point_size=15)
                p.add_mesh(pv.PolyData(dd['cMT']), color='red', point_size=15)
                p.add_mesh(pv.PolyData(dd['iLT']), color='blue', point_size=15)
                p.add_mesh(pv.PolyData(dd['aLT']), color='#03b3ff', point_size=15)
                p.add_mesh(pv.PolyData(dd['eLT']), color='#0bff03', point_size=15)
                p.add_mesh(pv.PolyData(dd['pLT']), color='yellow', point_size=15)
                p.add_mesh(pv.PolyData(dd['cLT']), color='red', point_size=15)

                p.set_background('w')
                p.show_grid(color='black', use_2d=True, font_size=48, xlabel=' ', ylabel=' ', zlabel=' ', font_family='courier')
                p.camera.azimuth = 270
                p.camera.elevation = 180
                p.enable_eye_dome_lighting()

                p.screenshot(f'../temp/tibial_subregions_{subdir.name}.png')

                lower_mesh_left, upper_mesh_left = utility.build_femoral_meshes(cwbzl)
                lower_mesh_right, upper_mesh_right = utility.build_femoral_meshes(cwbzr)

                left_landmarks = utility.femoral_landmarks(upper_mesh_left.points)
                right_landmarks = utility.femoral_landmarks(upper_mesh_right.points)

                dd = defaultdict(list)
                for v in upper_mesh_left.points:
                    vector = np.array(v)
                    label = utility.classify_femoral_point(v[:2], left_landmarks, True)
                    dd[label].append(vector)
                    
                for v in upper_mesh_right.points:
                    vector = np.array(v)
                    label = utility.classify_femoral_point(v[:2], right_landmarks, False)
                    dd[label].append(vector)

                p = pv.Plotter(off_screen=True, window_size=[2560, 1440])

                p.add_mesh(pv.PolyData(ladf.to_numpy()), color='#03b3ff', point_size=10)
                p.add_mesh(pv.PolyData(radf.to_numpy()), color='#03b3ff', point_size=10)
                p.add_mesh(pv.PolyData(cwbzl.to_numpy()), color='grey', point_size=10)
                p.add_mesh(pv.PolyData(cwbzr.to_numpy()), color='grey', point_size=10)
                p.add_mesh(pv.PolyData([[x[2], x[1], x[0]] for x in lpdf.to_numpy()]), color='yellow', point_size=10)
                p.add_mesh(pv.PolyData([[x[2], x[1], x[0]] for x in rpdf.to_numpy()]), color='yellow', point_size=10)

                p.set_background('w')
                p.show_grid(color='black', use_2d=True, font_size=48, xlabel=' ', ylabel=' ', zlabel=' ', font_family='courier')
                p.camera.azimuth = 270
                p.camera.elevation = 180
                p.enable_eye_dome_lighting()

                p.screenshot(f'../temp/femoral_subregions_{subdir.name}.png')

                p = pv.Plotter(off_screen=True, window_size=[2560, 1440])

                p.add_mesh(pv.PolyData(dd['ecMF']), color='0bff03', point_size=15)
                p.add_mesh(pv.PolyData(dd['ccMF']), color='red', point_size=15)
                p.add_mesh(pv.PolyData(dd['icMF']), color='blue', point_size=15)
                p.add_mesh(pv.PolyData(dd['ecLF']), color='0bff03', point_size=15)
                p.add_mesh(pv.PolyData(dd['ccLF']), color='red', point_size=15)
                p.add_mesh(pv.PolyData(dd['icLF']), color='blue', point_size=15)

                p.set_background('w')
                p.show_grid(color='black', use_2d=True, font_size=36, xlabel=' ', ylabel=' ', zlabel=' ', font_family='courier')
                p.camera.azimuth = 315
                p.camera.elevation = 180
                p.enable_eye_dome_lighting()

                p.screenshot(f'../temp/cwbz_subregions_{subdir.name}.png')


if __name__ == '__main__':
    save_screenshots()