
import os
import glob
import pandas as pd
import tensorflow as tf
import csv
from PIL import Image
import cv2


def combine_CSV(dirpath):
    os.chdir(dirpath)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ],sort=False)
    #export to csv
    combined_csv.to_csv( "train_com.csv", index=False, encoding='utf-8-sig')
    return combined_csv

def txt_to_csv(path,outpath):
    with open(path, 'r') as in_file:
        # print(in_file)
        # in_file.replace(" ", ",", 3)
        stripped = (line.replace(" ", ",", 6).strip() for line in in_file)
        lines = (line.split(",") for line in stripped if line)
        with open(outpath+'.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(("FileName","ClassName","id_num",'XMin', 'YMin',"XMax","YMax"))
            writer.writerows(lines)
    return out_file

def edit_csv(csv_path,outpath):
    csv_file = pd.read_csv(csv_path)
    file_write= open(outpath,"w+")
    file_write.write("FileName,XMin,XMax,YMin,YMax,ClassName"+ '\n')
    for index,row in csv_file.iterrows():
        fileName = row["FileName"]
        xmin    = int(row['XMin'])
        ymin    = int(row['YMin'])
        xmax   = int(row['XMax'])
        ymax  = int(row['YMax'])
        className = row["ClassName"]
        imgPath= './train/'+fileName if (".jpg" in fileName) else './train/'+fileName+'.jpg'
        img = cv2.imread(imgPath)
        # print(imgPath)
        # print (format(img.shape))
        height, width,channel = img.shape
        xmin,ymin,xmax,ymax=xmin/width,ymin/height,xmax/width,ymax/height
        file_write.write(fileName + ',' + str(xmin) + ',' + str(xmax) + ',' + str(ymin) + ',' + str(ymax) + ',' + className + '\n')
        
    file_write.close()
# "C:\Users\Dell\Desktop\data\logos\train\144503924.jpg"
        

def main(_):
    for f in {"adidas.csv","nike.csv","puma.csv","TH_train.csv","MK_train.csv"}:
        print(f)
        edit_csv(f,"edit/"+f)
    print("DONE")



if __name__ == '__main__':
    tf.app.run()
