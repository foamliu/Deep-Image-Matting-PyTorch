import zipfile


if __name__ == '__main__':
    filename = 'data/alphamatting/input_lowres.zip'
    print('Extracting {}...'.format(filename))
    with zipfile.open(filename) as tar:
        tar.extractall('data/alphamatting')

    filename = 'data/alphamatting/trimap_lowres.zip'
    print('Extracting {}...'.format(filename))
    with zipfile.open(filename) as tar:
        tar.extractall('data/alphamatting')


