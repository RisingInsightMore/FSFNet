import os, glob, argparse, numpy as np
from PIL import Image, ImageFile

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Prepare paired AB images for Pix2Pix aligned mode.')
parser.add_argument('--dataroot', type=str, default='./datasets/AVIID',
                    help='dataset root containing trainA/trainB and testA/testB')
args = parser.parse_args()

dataroot = args.dataroot
for phase in ['train', 'test']:
    dir_out = os.path.join(dataroot, phase)
    dir_A = os.path.join(dataroot, phase + 'A')
    dir_B = os.path.join(dataroot, phase + 'B')
    if os.path.isdir(dir_out) and len(os.listdir(dir_out)) > 0:
        print(f'[prep] {dir_out} already exists ({len(os.listdir(dir_out))} files), skipping.')
        continue
    os.makedirs(dir_out, exist_ok=True)
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    paths_A = sorted([p for e in exts for p in glob.glob(os.path.join(dir_A, e))])
    count = 0
    skipped = 0
    for pA in paths_A:
        fname = os.path.basename(pA)
        pB = os.path.join(dir_B, fname)
        if not os.path.isfile(pB):
            continue
        try:
            im_A = Image.open(pA).convert('RGB')
            im_B = Image.open(pB).convert('RGB')
            # Verify images loaded correctly
            im_A.load()
            im_B.load()
            im_A = np.array(im_A)
            im_B = np.array(im_B)
            if im_A.shape != im_B.shape:
                im_B = np.array(Image.fromarray(im_B).resize((im_A.shape[1], im_A.shape[0]), Image.BICUBIC))
            concat = np.concatenate([im_A, im_B], axis=1)
            Image.fromarray(concat).save(os.path.join(dir_out, fname))
            count += 1
        except Exception as e:
            print(f'[prep] Warning: Skipping {fname} due to error: {e}')
            skipped += 1
    print(f'[prep] Created {count} paired images in {dir_out} (skipped {skipped})')
