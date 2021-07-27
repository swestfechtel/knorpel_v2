import os
import json
import logging

from pathlib import Path


def scan_directory(directory, res_set):
    for subdir in os.scandir(directory):
        if subdir.is_dir():
            scan_directory(subdir, res_set)
        elif n := subdir.name == 'metadata.json':
            try:
                f = open(n)
                j = json.load(f)

                assert j['BodyPartExamined'] == 'KNEE'
                assert j['MRAcquisitionType'] == '3D'
                assert j['Modality'] == 'MR'
                # assert j['ProtocolName'] == 'SAG 3D DESS WE'
                # assert j['SeriesDescription'] == 'SAG3D'

                res_set.add(p := Path(os.path.relpath(subdir)).parent)
                logging.debug(f'Added {p} to result set.')

            except (FileNotFoundError, json.JSONDecodeError):
                continue

        else:
            continue


if __name__ == '__main__':
    logging.basicConfig(filename='/work/scratch/westfechtel/pylogs/indexing.log', encoding='utf-8',
                        level=logging.DEBUG, filemode='w')
    base_dir = '/images/Shape/Medical/Knees/OAI/FullNiftiTest'
    result_set = set()
    scan_directory(base_dir, result_set)
    print(result_set)
