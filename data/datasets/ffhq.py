from ai.data.img import ImgDataset


MISSING_FFHQ_MSG = '''
~~~~~~~~~~~~~~~~~~~~
FFHQ is not yet built-in to ai.

ai.data.ffhq({imsize}) expects images here:
$AI_DATASETS_PATH/ffhq/{imsize}/data/*.png

You can download the full dataset here:
https://github.com/NVlabs/ffhq-dataset

Or if you only want the 64x64 images for testing:
https://drive.google.com/file/d/1c6Dtq2qC9rdkoWq1e5QRfXwKZGxLbsxZ/view?usp=share_link
~~~~~~~~~~~~~~~~~~~~
'''

def ffhq(imsize, **kw):
    try:
        ds = ImgDataset('ffhq', imsize, **kw)
    except ValueError as e:
        print(e)
        raise Exception(MISSING_FFHQ_MSG.format(imsize=imsize))
    return ds
