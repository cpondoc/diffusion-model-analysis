'''
Puts all of the DALLE data into test and train folders
'''
import os
import shutil

def create_dalle_split():
    dalle_folders = os.listdir('data/dalle')
    counter = 0
    for folder in dalle_folders:
        if (folder[0] != "."):
            imgs = os.listdir('data/dalle/' + folder)
            for img in imgs:
                if (counter < 1098):
                    shutil.copy('data/dalle/' + folder + '/' + img, 'data/train/dalle/' + img)
                else:
                    shutil.copy('data/dalle/' + folder + '/' + img, 'data/test/dalle/' + img)
                counter += 1

def main():
    create_dalle_split()

main()