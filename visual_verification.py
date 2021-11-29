import utility
import os
import pyvista as pv
from multiprocessing import Pool

def function_for_pool(directory):
    segmentation_directory = f'../Manual_Segmentations/{directory}/{directory}_segm.mhd'
    sitk_image, np_image = utility.read_image(segmentation_directory)
    tibial_cartilage = utility.build_3d_cartilage_array(np_image, 4)
    femoral_cartilage = utility.build_3d_cartilage_array(np_image, 3)
    tibial_vectors = [list(element) for element in tibial_cartilage]
    femoral_vectors = [list(element) for element in femoral_cartilage]

    cwbzl, cwbzr = utility.extract_central_weightbearing_zone(femoral_vectors, tibial_vectors)
    lpdf, rpdf, adf = utility.extract_anterior_posterior_zones(femoral_vectors, cwbzl, cwbzr)
    ladf, radf = utility.split_anterior_part(adf)

    p = pv.Plotter(off_screen=True)
    p.add_mesh(pv.PolyData(ladf.to_numpy()), color='red')
    p.add_mesh(pv.PolyData(radf.to_numpy()), color='green')
    p.enable_eye_dome_lighting()
    p.camera_position = 'yz'
    p.camera.azimuth = 180
    p.camera.elevation = 90
    p.screenshot(f'../screenshots/{directory}.png')
    p.close()


if __name__ == '__main__':
    files = [f.name for f in os.scandir('../Manual_Segmentations/')]
    with Pool() as pool:
        pool.map(function_for_pool, files)
