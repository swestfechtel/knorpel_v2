import pickle
import json
import logging
import traceback

from pathlib import Path


def scan_directory(directory: Path, res_set: set):
    logging.debug(f'Scanning {directory}.')
    for subdir in directory.iterdir():
        if subdir.is_dir():
            scan_directory(subdir, res_set)
        elif subdir.name == 'metadata.json':
            try:
                f = open(subdir)
                j = json.load(f)

                assert j['BodyPartExamined'] == 'KNEE'
                assert j['MRAcquisitionType'] == '3D'
                assert j['Modality'] == 'MR'
                # assert j['ProtocolName'] == 'SAG 3D DESS WE'
                # assert j['SeriesDescription'] == 'SAG3D'
                f = open(str(p := subdir.parent) + '/image.nii.gz')
                res_set.add(p)
                logging.debug(f'Added {p} to result set.')

            except (FileNotFoundError, json.JSONDecodeError):
                logging.debug(subdir)
                logging.debug(traceback.format_exc())
                continue
            except (AssertionError, KeyError):
                continue

        else:
            continue


if __name__ == '__main__':
    logging.basicConfig(filename='/work/scratch/westfechtel/pylogs/indexing.log', encoding='utf-8',
                        level=logging.DEBUG, filemode='w')
    base_dir = '/images/Shape/Medical/Knees/OAI/FullNiftiTest'
    base_dir = Path(base_dir)
    result_set = set()
    scan_directory(base_dir, result_set)
    with open('/work/scratch/westfechtel/pickles/index/index.pickle', 'wb') as of:
        pickle.dump(result_set, of)

    for path in result_set:
        print(path)
