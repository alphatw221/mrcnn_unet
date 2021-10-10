import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math
import os

#--------------custom------------------
# 這邊不一定要繼承 util Dataset  只要有三個load_image get_mask get_unet_mask方法 一個屬性image_ids
#這個類別實例帶一個字典 屬性 包含label的資訊
#由字典資訊產生對應image_id的遮罩
#後續會將實例建構 KU Sequence實例 (類似於產生器) 裡頭生成input output的資料 

WIDTH=2048
HEIGHT=2048

class MyDataset():
    
    def __init__(self,imageDir,annoList,classDict):
        self.imageDir=imageDir
        self.annoList=annoList
        self.classDict=classDict
        self.numOfImages = len(annoList)
        # self._image_ids=self._getImageIdArray()

    # def _getImageIdArray(self):
    #     emptyArray=[]
    #     for i in range(self.numOfImages):
    #         id=self.annoList[i]["imageInfo"]["id"]
    #         emptyArray.append(id)
    #     return np.array(emptyArray)

    @property
    def active_class_ids(self):
        return [1]*(1+len(self.classDict))

    @property
    def image_ids(self):
        return np.arange(self.numOfImages)

    def load_image(self,imageId):
        path=os.path.join(self.imageDir,self.annoList[imageId]["imageInfo"]["fileName"])
        return plt.imread(path)[28:28+1024,163:163+1024,:]  #TODO Cropping

    def load_mask(self,imageId):



        #這裡要想辦法把物件分離



        
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
        mask=mask.astype(np.bool)
        return mask.astype(np.float)[28:28+1024,163:163+1024,:], class_ids.astype(np.int32)  #TODO Cropping

    def load_u_net_mask(self,imageId):
        imageInfo=self.annoList[imageId]["imageInfo"]
        annos=self.annoList[imageId]["annos"]
        mask=np.zeros((imageInfo["height"],imageInfo["width"],len(self.classDict)),np.uint8)
        for anno in annos:
            tempMask=np.zeros((imageInfo["height"],imageInfo["width"]),np.uint8)
            cv2.fillPoly(tempMask,[anno["xy"]],(1))
            mask[:,:,anno["classId"]]+=tempMask
        mask=mask.astype(np.bool)
        return mask.astype(np.float)[28:28+1024,163:163+1024,:]  #TODO Cropping

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