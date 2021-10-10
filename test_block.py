#%%
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math
import os
#%%
class MyDataset():
    def __init__(self,datasetDir, classDict, instanceClasses):
        self.imageDir = os.path.join(datasetDir,'image')
        imageFileName = os.listdir(self.imageDir)
        self.imagePath = [os.path.join(self.imageDir, fileName) for fileName in imageFileName]
        self.labelDir=os.path.join(datasetDir,'label')
        labelFileName = os.listdir(self.labelDir)
        self.labelPath = [os.path.join(self.labelDir, labelName) for labelName in labelFileName]
        self.classDict = classDict
        self.numOfImages = len(os.listdir(self.imageDir))
        self.image_ids = range(self.numOfImages)
        self.instanceClasses = instanceClasses

    def load_image(self,imageId):
        return plt.imread(self.imagePath[imageId])  #TODO Cropping

    def load_mask(self,imageId):
        labelImage = cv2.imread(self.labelPath[imageId])
        labelImage = cv2.cvtColor(labelImage, cv2.COLOR_BGR2RGB)
        
        mask = np.ones((labelImage.shape[0], labelImage.shape[1],1), np.bool)
        class_ids = [0]
        for class_id in range(1, len(self.instanceClasses)+1):
            hexDecimal = self.classDict[self.instanceClasses[class_id-1]].lstrip('#')
            rgb = tuple(int(hexDecimal[i:i+2], 16) for i in (0, 2, 4))
            
            classMask = (labelImage[:,:,0]==rgb[0])*(labelImage[:,:,1]==rgb[1])*(labelImage[:,:,2]==rgb[2])
            num_objects, objectIdMask = cv2.connectedComponents(classMask.astype('uint8'))
            num_objects-=1
            if num_objects:
                objectsMask = np.zeros((classMask.shape[0], classMask.shape[1], num_objects), np.bool)
                for i in range(num_objects):
                    objectsMask[:,:,i] = (objectIdMask==i+1)
                    class_ids.append(class_id)
                mask = np.concatenate((mask, objectsMask),axis = 2)
            
            
        #去重疊
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(len(class_ids)-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # return mask.astype(np.bool), class_ids.astype(np.int32)  #TODO Cropping
        return mask.astype(np.uint8), np.array(class_ids, dtype = np.int32)
    
    def load_u_net_mask(self,imageId):
        
        labelImage = cv2.imread(self.labelPath[imageId])
        labelImage = cv2.cvtColor(labelImage, cv2.COLOR_BGR2RGB)
        
        mask = np.zeros((labelImage.shape[0],labelImage.shape[1],len(self.classDict)),np.bool)
        i = 0
        for key in self.classDict:
            hexDecimal = self.classDict[key].lstrip('#')
            rgb = tuple(int(hexDecimal[i:i+2], 16) for i in (0, 2, 4))
            
            mask[:,:,i] = (labelImage[:,:,0]==rgb[0])*(labelImage[:,:,1]==rgb[1])*(labelImage[:,:,2]==rgb[2])
            i+=1
            
        return mask  #TODO Cropping



#%%
datasetDir=r"C:\Users\tnt\Desktop\dataset"
classDict={'flour':'#FF0000', 'patch':'#0000FF', 'glueAndFillement':'#FF00FF', 'outOfFocus':'#00FFFF'}
instanceClasses=['patch', 'glueAndFillement', 'outOfFocus']
#           classID    1         2            3
trainData=MyDataset(datasetDir, classDict, instanceClasses)
#%%
# unet_mask=trainData.load_u_net_mask(0)
# um=unet_mask.astype(np.uint8)
# for i in range(4):
#     test=um[:,:,i]
#     plt.imshow(test*256)
#     plt.show()
#%%
# image_ids=trainData.image_ids
# image=trainData.load_image(50)
# plt.imshow(image)
# plt.show()
#%%
mask,class_ids=trainData.load_mask(2)
for i in range(len(class_ids)):
    test=mask[:,:,i]
    plt.imshow(test*256)
    plt.show()
    print(class_ids)


#%%

