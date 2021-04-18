from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm

# webp 확장자를 가진 파일 명만 가져오기
def grabListOfFiles(startingDirectory, extention=".webp"):

    listOfFiles=[]
    print("Get Webp file")
    for file in tqdm(os.listdir(startingDirectory)) :
        # endswith = 문자열의 마지막이 지정한 문자로 끝나게함.
        if file.endswith(extention):
            listOfFiles.append(os.path.join(startingDirectory, file))
    return listOfFiles


# image를 rgb나 gray scale로 불러와서 resize 후 저장.
# webp 파일을 읽으려면 Pillow의 Image 클래스를 사용해야함.
def grabArrayOfImages(listOfFiles, resizeW=64, resizeH=64, gray=False) :
    imageArr = []
    print("Get image array")
    for f in tqdm(listOfFiles) :
        if gray :
            im = Image.open(f).convert("L")
        else :
            im = Image.open(f).convert("RGB")

        im = im.resize((resizeW, resizeH))
        imData = np.asarray(im)
        imageArr.append(imData)
    return imageArr

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='input data path')
args = parser.parse_args()

input_data = args.data

listofFiles = grabListOfFiles(input_data)
imageArrGray = grabArrayOfImages(listofFiles, resizeW=64, resizeH=64 ,gray=True)
imageArrColor = grabArrayOfImages(listofFiles, resizeW=64, resizeH=64, gray=False)

print("shape of Gray image : {}".format(np.shape(imageArrGray)))
print("shape of Color image : {}".format(np.shape(imageArrColor)))

# numpy 배열을 통째로 저장함.
np.save('./lsun_datasets/church_outdoor_train_lmdb_gray.npy', imageArrGray)
np.save('./lsun_datasets/church_outdoor_train_lmdb_color.npy', imageArrColor)



