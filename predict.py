import scipy.io as sio
import cv2
import numpy as np
from get_data import get_data_list
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from extract_feature import LBP, extract_image

def drawImageWithPredictions(image_color,nbPointGT,predictions,widthOfPatch,heightOfPatch):
    predictionImage=np.copy(image_color)
    for i in range(number_Column):
        for j in range(number_Row):
            rectangle={"anchor":(i*heightOfPatch,j*widthOfPatch),"width":widthOfPatch,"height":heightOfPatch}
            cv2.rectangle(predictionImage,(j*widthOfPatch,i*heightOfPatch),(j*widthOfPatch+widthOfPatch,i*heightOfPatch+heightOfPatch),(0,0,255),1)
    
            cv2.putText(image_color, str(max(0,nbPointGT[i,j])), (int((j+0.5)*(widthOfPatch)),int((i+0.5)*(heightOfPatch))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), thickness=3,lineType=cv2.LINE_AA)
            cv2.rectangle(image_color,(j*widthOfPatch,i*heightOfPatch),(j*widthOfPatch+widthOfPatch,i*heightOfPatch+heightOfPatch),(0,0,255),1)
            

            cv2.putText(predictionImage, str(max(0,int(predictions[i,j]))), (int((j+0.5)*(widthOfPatch)),int((i+0.5)*(heightOfPatch))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0),thickness=3, lineType=cv2.LINE_AA) 
            #cv2.imshow("prediction_image",predictionImage)
    return image_color, predictionImage


if __name__ == '__main__':
    number_Column = 12
    number_Row = 16
    path = 'C:\\Users\\lizhaoyang\\Downloads\\Crowd-Count\\ShanghaiTech\\part_B'

    images_list, gts_list = get_data_list(path, mode='test')
    try:
        pathToSaveImage=os.path.join(path,"images_Predictions")
        os.makedirs(pathToSaveImage)
        
    except:
        print("Folder already exist!")
    
    matFile=sio.loadmat(os.path.join(path,"train"))
    descriptorsTrain=matFile["lbp_descriptors"]
    labelsTrain=matFile["labels"]

    matFile=sio.loadmat(os.path.join(path,"test"))
    descriptorsTest=matFile["lbp_descriptors"]
    labelsTest=matFile["labels"]

    alpha = 0.01
    gamma = 0.59948425
    KRR = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
    KRR.fit(descriptorsTrain, labelsTrain)

    for img_idx in range(len(images_list)):
        
        matFile=sio.loadmat(gts_list[img_idx])
        train_head_points = matFile['image_info'][0][0][0][0][0]
        image_color = np.asarray(cv2.imread(images_list[img_idx]), dtype=np.uint8)

        ###Convert to grayscale if the image is not in grayscale already
        height,width,numChannels=image_color.shape
       
       
        widthOfPatch=int(width/float(number_Row))
        heightOfPatch=int(height/float(number_Column))

        lbp=LBP(numPoints=8, radius=1)
        lbpWholeImage=[]
        
                
        # Load the image and ground truth
        rawImage = np.asarray(cv2.imread(images_list[img_idx]), dtype=np.uint8)
        matFile=sio.loadmat(gts_list[img_idx])
        headPoints = matFile['image_info'][0][0][0][0][0]            
        nbPointInImage,lbpWholeImage=extract_image(rawImage,headPoints,lbp,number_Row,number_Column)
        
        for x,y in list(headPoints):
            cv2.circle(image_color, (int(x),int(y)),2,(0, 255, 0),-1)
        
        
        predictions=(KRR.predict(lbpWholeImage.reshape(1,-1))).reshape(number_Column,number_Row)
        predictionImage=np.copy(image_color)
        nbPointInImage=np.array(nbPointInImage).reshape(number_Column,number_Row)
        image_color,predictionImage=drawImageWithPredictions(image_color,nbPointInImage,predictions,widthOfPatch,heightOfPatch)

        predictionAndGroundTruthImage=255*np.ones([predictionImage.shape[0],2*predictionImage.shape[1]+30,3])
        

        predictionAndGroundTruthImage[:,0:predictionImage.shape[1],:]=image_color
        predictionAndGroundTruthImage[:,-predictionImage.shape[1]:,:]=predictionImage

        cv2.imwrite(os.path.join(pathToSaveImage,str(img_idx)+'.jpg'), predictionImage)
        cv2.imwrite(os.path.join(pathToSaveImage,str(img_idx)+'_gts.jpg'), image_color)
        cv2.imwrite(os.path.join(pathToSaveImage,str(img_idx)+'_both.jpg'), predictionAndGroundTruthImage) 