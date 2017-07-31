import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import sys

#import countchain as cc
np.set_printoptions(threshold=np.inf)

def loadImgs(filenames):
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
    Left_x, Right_x = 0, 255
    for is_Left in [True, False]:
        for i in range(256) if is_Left else reversed(range(256)):
            if l[i] > 0 and is_Left:
                Left_x = i
                break
            else:
                Right_x = i
                break
    center = int(Left_x + round((Right_x - Left_x) / 2.0))
    L_x, R_x = np.where(l[0:center] == max(l[0:center]))[0][0], np.where(
        l[center + 1:255] == max(l[center + 1:255]))[0][0] + center
    hoge = l[center:255]
    return ((sum(hoge[np.average(hoge) < hoge]) + sum(hoge)) / 2.0) / sum(l)


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



class LookUpTable(object):

    def __init__(self):
        self.__LookUpTable = []
        self.is_same   = lambda idx: idx==-2
        self.is_not_found = lambda idx: idx==-1

    def __fetch_index(self,px1,px2):
        px1_index,px2_index = -1,-1
        for index, table in enumerate(self.__LookUpTable):
            if px1 in table and px2 in table: return [-2,-2];
            if px1 in table : px1_index = index
            if px2 in table : px2_index = index
        return [ px1_index, px2_index ]

    def __both_append(self,pxs):
        self.__LookUpTable.append(pxs)

    def __one_side_append(self,pxs,index):
        here = index[0] if index[1]==-1 else index[1]
        it = pxs[1] if index[1]==-1 else pxs[0]
        self.__LookUpTable[here].append(it)

    def __merge(self,pxs,index):
        self.__LookUpTable[index[0]].extend(self.__LookUpTable[index[1]])
        del self.__LookUpTable[index[1]]

    def set(self,px1,px2):
        pxs = [px1,px2]
        i_1,i_2 = self.__fetch_index( px1, px2 )
        if self.is_same(i_1) and self.is_same(i_2) : return
        elif self.is_not_found(i_1) and self.is_not_found(i_2) : self.__both_append(pxs)
        elif self.is_not_found(i_1) or self.is_not_found(i_2) : self.__one_side_append(pxs,[i_1,i_2])
        elif i_1 != i_2 : self.__merge(pxs,[i_1,i_2])

    def lookup(self,px):
        for index, table in enumerate(self.__LookUpTable):
            if px in table :
                return index+1;
        self.__both_append([px])
        return self.lookup(px)

    def number_of_label(self):
        return len(self.__LookUpTable)

class CountChain(object):

    def __init__(self,filename):
        self.max_label = 0
        self.lut = LookUpTable()
        self.map = (lambda a,b,c,d,e: int((a*1.0/(c-b))*(e-d*1.0)))
        self.img = cv.cvtColor( cv.imread(filename), cv.COLOR_RGB2GRAY )
        self.labels = np.zeros(self.img.shape)
        self.isset = False

    def __is_white(self,p):
        in_range = 0<=p[0] and p[0]<self.img.shape[0] and 0<=p[1] and p[1]<self.img.shape[1]
        return self.img[p[0],p[1]] == 255 if in_range else True

    def __is_label(self,p):
        in_range = 0<=p[0] and p[0]<self.img.shape[0] and 0<=p[1] and p[1]<self.img.shape[1]
        return self.labels[p[0],p[1]] > 0 if in_range else False

    def __new_label(self):
        self.max_label += 1
        return self.max_label

    def __flesh(self):
        for y,labels in enumerate(self.labels):
          for x,label in enumerate(labels):
              if self.labels[y,x] != 0 :
                  self.labels[y,x] = int(self.lut.lookup(label))

    def __up(self,point):
        return [point[0]-1,point[1]]

    def __left(self,point):
        return [point[0],point[1]-1]

    def labeling(self):
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                p = [y,x]
                if self.img[y,x] == 255:
                    if self.__is_label( self.__left(p) ) and self.__is_label( self.__up(p) ) :
                        self.labels[y,x] = self.labels[y-1,x]
                        if self.labels[y,x] != self.labels[y,x-1] : self.lut.set(self.labels[y,x],self.labels[y,x-1])
                    elif self.__is_white( self.__left(p) ) and self.__is_label( self.__up(p) ) : self.labels[y,x] = self.labels[y-1,x]
                    elif self.__is_white( self.__up(p) ) and self.__is_label( self.__left(p) ) : self.labels[y,x] = self.labels[y,x-1]
                    elif self.__is_white( self.__up(p) ) and self.__is_white( self.__left(p) ) : self.labels[y,x] = int(self.__new_label())
        self.__flesh()
        print "number of labels is ",self.lut.number_of_label()

    def get_chain_img(self):
        if self.isset:
            return self.chain
        else :
            self.chain = cv.cvtColor(self.img,cv.COLOR_GRAY2RGB)
            for y,labels in enumerate(self.labels):
              for x,label in enumerate(labels):
                  if self.labels[y,x] == 0 :
                      self.chain[y,x] = [0,0,0]
                  else :
                      h = self.map(self.labels[y,x],0,14,0,255)
                      self.chain[y,x] = [h,256-h,h**3]
            return self.chain


def main():
    sys.setrecursionlimit(20000)
    index = 0
    filenames = ['sample2.pgm', 'sample3.pgm', 'sample4.pgm', 'fun.png']
    imgs = loadImgs(filenames)
    lumin = getLuminance(imgs[index])
    p = getPBright(imgs[index])
    img_bit = percentileMethod(p, imgs[index], True)
    cv.imwrite("./imgs/bit_" + filenames[index] + ".png", img_bit)
    cv.imshow(filenames[index], img_bit)
    cc = CountChain("./imgs/bit_" + filenames[index] + ".png")
    cc.labeling()
    cv.imshow("chain",cc.get_chain_img())
    cv.imwrite("hoge.png",cc.get_chain_img())
    cv.waitKey()

if __name__ == "__main__": main()
