import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def loadImgs():
    res = []
    for filename in filenames:
        img = cv.imread("./imgs/" + filename)
        cv.imwrite("./imgs/" + filename + ".png", img)
        res.append(cv.cvtColor(img, cv.COLOR_RGB2GRAY))
    return res


def getLuminance(image):
    image = np.reshape(image, (1, np.product(image.shape)))[0]
    lum = np.zeros(256)
    for l in range(256):
        lum[l] = len(image[image == l])
    return lum


def getLuminances(images):
    lums = []
    for i in range(len(images)):
        lums.append(getLuminance(imgs[i]))
    return lums

def getPBright(image):
    l = getLuminance(image)
    Left_x, Right_x = 0,255
    for is_Left in [True,False]:
        for i in range(256) if is_Left else reversed(range(256)) :
            if l[i] > 0 and is_Left :
                Left_x = i; break
            else :
                Right_x = i; break
    center = int(Left_x + round((Right_x - Left_x)/2.0))
    L_x, R_x = np.where(l[0:center]==max(l[0:center]))[0][0], np.where(l[center+1:255]==max(l[center+1:255]))[0][0]+center
    hoge = l[center:255]
    return ( (sum(hoge[np.average(hoge)<hoge])+sum(hoge))/2.0 )/sum(l)

def percentileMethod(p, img, from_bright):
    l = getLuminance(img)
    px, c = (len(img) * len(img[0])) * p, 0
    threshold = 0
    for i in reversed(range(256)) if from_bright else range(256):
        c += l[i]
        if c >= px:
            threshold = i
            break
    img[img < threshold] = 0
    img[img >= threshold] = 255
    return img


def showHistgram(l, filename):
    y = l
    plt.bar(range(256), y, align='center')
    plt.xlabel("luminance")
    plt.ylabel("number of pixel")
    plt.title("histgram of luminance about " + filename)
    plt.show()

index = 1
filenames = ['sample2.png', 'sample3.pgm', 'sample4.pgm', 'fun.png']
imgs = loadImgs()
img_gray = cv.cvtColor(imgs[index], cv.COLOR_RGB2GRAY)
lumin = getLuminance(img_gray);
p = getPBright(img_gray)
img_bit = percentileMethod(p,img_gray,True)
cv.imwrite("./imgs/bit_"+filename+".png",img_bit)
cv.imshow('hoge',img_bit);
showHistgram(lumin,"filename");
