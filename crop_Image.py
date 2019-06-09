
import cv2
import pandas as pd
from PIL import Image
import os


def cropImage(image_name,XMin,YMin,XMax,YMax,path,name):
    img = cv2.imread("./train/"+image_name)
    crop_img = img[YMin:YMax, XMin:XMax]
    cv2.imwrite(path+"/"+name+".jpg", crop_img)
    # img2 = cv2.imread(path+"/"+image_name)
    # cv2.imshow("cropped", img2)

    cv2.waitKey(0)
    #save image

    return crop_img
   
   


def edit_csv(csv_path):
    csv_file = pd.read_csv(csv_path)
    file_write= open("1"+csv_path,"w+")
    file_write.write("FileName,XMin,YMin,XMax,YMax,ClassName"+ '\n')
    for index,row in csv_file.iterrows():
        fileName = row["FileName"]
        xmin    = int(row['XMin'])
        ymin    = int(row['YMin'])
        width   = int(row['width'])
        height  = int(row['hight'])
        xmax,ymax   = int(xmin+width) , int(ymin+height)
        className = row["ClassName"]
        file_write.write(fileName + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + className + '\n')
        
    file_write.close()

def main():
    # for edit in {"adidascsv.csv","nikecsv.csv","pumacsv.csv"}:
    #     print(edit)
    #     edit_csv(edit)
    # print("update")

    for csv in {"MK_train.csv","TH_train.csv","1adidascsv.csv","1pumacsv.csv","1nikecsv.csv","27data.csv"}:
        readCSV = pd.read_csv(csv)
        print(csv)
        i=0
        for index,row in readCSV.iterrows():
            fileName = row["FileName"]
            className = row["ClassName"]
            XMin = int(row["XMin"])
            XMax = int(row["XMax"])
            YMin = int(row["YMin"])
            YMax = int(row["YMax"])
            fileName= fileName if(".jpg" in fileName) else fileName+".jpg"
            i+=1
            name = className+str(i)
            print(name)
            cropImage(fileName,XMin,YMin,XMax,YMax,"./cropped/"+className,name)

        print("DONE "+csv)


main()