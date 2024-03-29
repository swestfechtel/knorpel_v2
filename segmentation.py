import os
import sys
import logging
import traceback

import numpy as np

if __name__ == '__main__':
    logging.basicConfig(filename='/work/scratch/westfechtel/pylogs/segmentation/segmentation_default.log', encoding='utf-8',
                        level=logging.DEBUG, filemode='w')
    logging.debug('Entered main.')

    try:
        assert len(sys.argv) == 2
        chunk = np.load(f'/work/scratch/westfechtel/segmentation_chunks/{sys.argv[1]}.npy', allow_pickle=True)

        filehandler = logging.FileHandler(f'/work/scratch/westfechtel/pylogs/segmentation/{sys.argv[1]}.log', mode='w')
        filehandler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        root.addHandler(filehandler)
        logging.debug(f'Using chunk {sys.argv[1]} with length {len(chunk)}.')

        inputs = str()
        outputs = str()
        for i, path in enumerate(chunk):
            # inputs += '/workdir/images' + str(path) + '/image.nii.gz,'
            # outputs += f'/workdir/images/work/scratch/westfechtel/segmentations/{path.stem}.nii.gz,'
            inputs = '/workdir/images' + str(path) + '/image.nii.gz'
            outputs = f'/workdir/images/work/scratch/westfechtel/segmentations/{path.stem}.nii.gz'
            try:
                stream = os.popen(
                    f'docker run --rm -u $(id -u ${{USER}}):$(id -g ${{USER}}) -v /:/workdir/images --gpus all justusschock/shape_fitting_miccai_pred --inputs {inputs} --outputs {outputs}')
                logging.debug(stream.read())
            except Exception as e:
                logging.debug(traceback.format_exc())
                continue

        # inputs = inputs[:-1]
        # outputs = outputs[:-1]


    except Exception as e:
        logging.debug(traceback.format_exc())
        logging.debug(sys.argv)
