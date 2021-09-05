#%%
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math
import os
#%%
class MyDataset():
    
    def __init__(self,imageDir,annoList,classDict):
        self.imageDir=imageDir
        self.annoList=annoList
        self.classDict=classDict
        self.numOfImages = len(annoList)
        self._image_ids=self._getImageIdArray()

    def _getImageIdArray(self):
        emptyArray=[]
        for i in range(self.numOfImages):
            id=self.annoList[i]["imageInfo"]["id"]
            emptyArray.append(id)
        return np.array(emptyArray)

    @property
    def image_ids(self):
        return self._image_ids

    def load_image(self,imageId):
        path=os.path.join(self.imageDir,self.annoList[imageId]["imageInfo"]["fileName"])
        return plt.imread(path)  #TODO Cropping

    def load_mask(self,imageId):
        imageInfo=self.annoList[imageId]["imageInfo"]
        annos=self.annoList[imageId]["annos"]
        numOfAnno=len(annos)
        mask=np.zeros((imageInfo["height"],imageInfo["width"],numOfAnno),np.uint8)  
        
        tempArray=[]
        for i in range(numOfAnno):
            tempArray.append(annos[i]["classId"])
            tempMask=np.zeros((imageInfo["height"],imageInfo["width"]),np.uint8)
            cv2.fillPoly(tempMask,[annos[i]["xy"]],(1))
            mask[:,:,i]=tempMask
        class_ids=np.array(tempArray)
        #去重疊
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(numOfAnno-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        return mask.astype(np.bool), class_ids.astype(np.int32)  #TODO Cropping

    def load_u_net_mask(self,imageId):
        imageInfo=self.annoList[imageId]["imageInfo"]
        annos=self.annoList[imageId]["annos"]
        mask=np.zeros((imageInfo["height"],imageInfo["width"],len(self.classDict)),np.uint8)
        for anno in annos:
            tempMask=np.zeros((imageInfo["height"],imageInfo["width"]),np.uint8)
            cv2.fillPoly(tempMask,[anno["xy"]],(1))
            mask[:,:,anno["classId"]]+=tempMask
        return mask.astype(np.bool)  #TODO Cropping

#----------------------------------------------------------------------------------------------------------------------

def buildVIA_Anno(imageDir,CSV_path,classDict,shuffle=True,splitRate=0.9):
    import csv
    
    data=[]
    indexDict={}

    with open(CSV_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader: 
            if '#' in line[0][0]:
                continue
            fileName = line[1][2:-2]
            xy=np.array(line[4].strip('][').split(',')[1:]).reshape((-1,2)).astype(np.float).round().astype(int)
            metaData = line[5].strip('}{').split(':')
            classId=0 if not metaData[0] else int(metaData[1].strip('"'))

            if fileName not in indexDict:
                img=plt.imread(os.path.join(imageDir,fileName))
                obj={"imageInfo":{
                        "id": len(data),
                        "width": img.shape[1],
                        "height": img.shape[0],
                        "channel":img.shape[2],
                        "fileName": fileName
                    },"annos":[]}
                indexDict[fileName]=len(data)
                data.append(obj)

            anno={"classId":classId,"className":classDict[classId],"xy":xy}
            data[indexDict[fileName]]["annos"].append(anno)
            
    if shuffle:
        pass
        #TODO shuffle
    separateIndex=math.floor(len(data)*splitRate)
    return data[:separateIndex],data[separateIndex:]

#%%
imageDir=r"C:\Users\tnt\Desktop\spk_env\imageDir"
VIA_csvPath=r"C:\Users\tnt\Desktop\spk_env\mask.csv"
classDict={0:'eye',1:'spk'}
trainAnnoList,valAnnoList=buildVIA_Anno(imageDir,VIA_csvPath,classDict,shuffle=True,splitRate=0.9)

# %%
trainData=MyDataset(imageDir,trainAnnoList,classDict)
#%%
unet_mask=trainData.load_u_net_mask(50)
um=unet_mask.astype(np.uint8)
for i in range(2):
    test=um[:,:,i]
    plt.imshow(test*256)
    plt.show()
#%%
image_ids=trainData.image_ids
image=trainData.load_image(50)
plt.imshow(image)
plt.show()
#%%
mask,class_ids=trainData.load_mask(50)
m=mask.astype(np.uint8)
for i in range(len(class_ids)):
    test=m[:,:,i]
    plt.imshow(test*256)
    plt.show()


#%%

