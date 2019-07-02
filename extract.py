import zipfile

if __name__ == '__main__':
    filename = 'data/alphamatting/input_lowres.zip'
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data/alphamatting/')

    filename = 'data/alphamatting/trimap_lowres.zip'
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data/alphamatting/')
