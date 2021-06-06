import scipy.io as sio
import cv2
from get_data import get_data_list
import numpy as np
from skimage import feature
import numpy as np
import os

class LBP:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def desc_lbp(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints,
                self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                bins=np.arange(0, 60))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist

def normalize(image):

    mean = np.mean(image)
    var = np.mean(np.square(image-mean))

    image = (image - mean)/np.sqrt(var)

    return image

def getPoints_Rect(points,rectangle={"anchor":(0,0),"width":200,"height":200}):
    
    mask=(points[:,0]<(rectangle["anchor"][1]+rectangle["width"])) & (points[:,0]> rectangle["anchor"][1])
    mask=(mask) & (points[:,1]<(rectangle["anchor"][0]+rectangle["height"])) & (points[:,1]> rectangle["anchor"][0])
    
    return np.sum(mask),points[mask,:]

def drawPoints(image,points):
    for x,y in list(points):
        cv2.circle(image, (int(x),int(y)), 1, (0, 255, 0), -1)
    return image

def extract_image(rawImage,points,lbp,numberPerRow,numberPerColumn,showImg=0):
    
    
    ###Convert to grayscale if the image is not in grayscale already
    height,width,numChannels=rawImage.shape
    if(numChannels==3):
        grayImage=cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
    else:
        grayImage=rawImage
    nbPointInImage=[]
    widthOfPatch=int(width/float(numberPerRow))
    heightOfPatch=int(height/float(numberPerColumn))
    allHist=[]
    
    for i in range(numberPerColumn):
        for j in range(numberPerRow):
            ###Get the points inside a patch
            nbPoint,headPointsInPatch=getPoints_Rect(points,rectangle={"anchor":(i*heightOfPatch,j*widthOfPatch),"width":widthOfPatch,"height":heightOfPatch})
            nbPointInImage.append(nbPoint)

            ###Draw the boundaries of the current patch
            cv2.rectangle(rawImage,(j*widthOfPatch,i*heightOfPatch),(j*widthOfPatch+widthOfPatch,i*heightOfPatch+heightOfPatch),(0,0,255),3)
            
            ###Draw the points available in the current patch
            rawImage=drawPoints(rawImage,headPointsInPatch)

            ###Show the image
            if (showImg):
                cv2.imshow("image",rawImage)
                k=cv2.waitKey(1)

            ###Calculate the LBP
            hist=lbp.desc_lbp(grayImage[i*heightOfPatch:(i+1)*heightOfPatch,j*widthOfPatch:(j+1)*widthOfPatch], eps=1e-7)

            allHist.append(hist)
            
    ###Concatenate all descriptors of all patches into one descriptor vector  
    allHistInImage=np.concatenate(allHist, axis=0)

    return nbPointInImage,allHistInImage

def process_data(path, numberPerRow, numberPerColumn, mode="train"):
    #Get the list of images.
    images_list, gts_list = get_data_list(path, mode=mode)
    allPoints=None
    allHist=None
    lbp=LBP(numPoints=8, radius=1)
    for img_idx in range(len(images_list)):
        
        # Load the image and ground truth
        rawImage = np.asarray(cv2.imread(images_list[img_idx]), dtype=np.uint8)
        matFile=sio.loadmat(gts_list[img_idx])
        if img_idx%10 == 0:
            print("Image processed : "+str(img_idx)+"/"+str(len(images_list)))
        headPoints = matFile['image_info'][0][0][0][0][0]
            
        nbPointInImage,allHistInImage=extract_image(rawImage,headPoints,lbp,numberPerRow,numberPerColumn,showImg=0)
        
        ###Concatenate all 
        if(allPoints is None):
            allPoints=np.array(nbPointInImage)
        else:
            allPoints=np.vstack((allPoints,nbPointInImage))

        if(allHist is None):
            allHist=allHistInImage

        else:
            allHist=np.vstack((allHist,allHistInImage))
        
    ###Save the extracted descriptors matrix and ground truth matrix
    sio.savemat(os.path.join(path,mode+".mat"),{"lbp_descriptors":allHist,"labels":allPoints})
      
    print(mode+ "set descriptors shape:" +str(allHist.shape))
    print(mode+ "set ground truth shape:" +str(allPoints.shape))

if __name__=='__main__':
    path = 'C:\\Users\\lizhaoyang\\Downloads\\Crowd-Count\\ShanghaiTech\\part_B'
    numberPerRow = 16
    numberPerColumn = 12
    process_data(path, numberPerRow, numberPerColumn, mode = 'train')
    process_data(path, numberPerRow, numberPerColumn, mode = 'test')
