import os
import sys
import csv
import datetime
import time
import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import scipy.misc
import tkinter as tk
from tkinter import *
from PIL import ImageTk,Image
import pickle
from tkinter import filedialog

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


bandwidth=0
compression_power=5

def speedtest():
   speed = os.popen("speedtest-cli --simple").read()
   lines = speed.split('\n')
   if "Cannot" in speed:
    	p = 0
    	d = 0
    	u = 0
   else:
     p = lines[0][6:11]
     d = lines[1][10:16]
     u = lines[2][8:12]
   return float(u)

def mse(image_a, image_b):
    # calculate mean square error between two images
    err = np.sum((image_a.astype(float) - image_b.astype(float)) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    return err

def get_centroids(c, p):
    # double the centroids after each iteration
    final_centroids = np.copy(c)
    for centroid in c:
        final_centroids = np.vstack((final_centroids, np.add(centroid, p)))
    return final_centroids

def load(*args):
    temp=filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    file_loc_text.set(temp)
    image=ImageTk.PhotoImage(Image.open(file_loc_text.get()).resize((300,400)))
    img=tk.Label(content1,image=image,width=300,height=400)
    img.image=image
    img.grid(column=0,row=0,rowspan=3,columnspan=4,sticky=(N,W,E,S),padx=10,pady=10)

def raise_frame(frame):
    frame.tkraise()

def next1(*args):
    bandwidth=speedtest()
    u=bandwidth*1000
    cal_band.set(str(u)+" Kbps")
    if(rad_b_text.get()=='auto'):
        if u<100:
                compression_power = 1
        elif u<200:
                compression_power = 2
        elif u<300:
                compression_power = 3
        elif u<400:
                compression_power = 4
        elif u<500:
                compression_power = 5
        elif u<600:
                compression_power = 6
        elif u<700:
                compression_power = 7
        elif u<800:
                compression_power = 8
        elif u<900:
                compression_power = 9
        elif u<1024:
                compression_power = 10
        elif u<2048:
                compression_power = 11
        else:
                compression_power = 12
        raise_frame(content2)
    else:
        raise_frame(content3)        

def compression():
    # source image
        image_location = file_loc_text.get()
        image = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)
        
        image_height = len(image)
        image_width = len(image[0])
        
        # dimension of the vector
        block_width = int(4)
        block_height = int(4)
        vector_dimension = block_width*block_height
        
        global compression_power
        
        if(rad_b_text.get()=='manual'):
            compression_power=int(range1.get())
        
        bits_per_codevector = int(compression_power)
        codebook_size = pow(2, bits_per_codevector)
        
        image_vectors = []
        for i in range(0, image_width, block_width):
            for j in range(0, image_height, block_height):
                image_vectors.append(np.reshape(image[i:i+block_width, j:j+block_height], vector_dimension))
        image_vectors = np.asarray(image_vectors).astype(float)
        number_of_image_vectors = image_vectors.shape[0]
        
        centroids = 255 * np.random.rand(codebook_size, vector_dimension)
        
        whitened = whiten(np.asarray(image_vectors))
        reconstruction_values, distortion = kmeans(image_vectors, centroids)
        
        image_vector_indices, distance = vq(image_vectors, reconstruction_values)
        
        image_after_compression = np.zeros([image_width, image_height], dtype="uint8")
        for index, image_vector in enumerate(image_vectors):
            start_row = int(index / (image_width/block_width)) * block_height
            end_row = start_row + block_height
            start_column = (index*block_width) % image_width
            end_column = start_column + block_width
            image_after_compression[start_row:end_row, start_column:end_column] = \
                np.reshape(reconstruction_values[image_vector_indices[index]],
                           (block_width, block_height))
        
        output_image_name = "compressed.png"
        scipy.misc.imsave(output_image_name, image_after_compression)
        
        if(rad_b_text.get()=='auto'):
            image2=ImageTk.PhotoImage(Image.open("compressed.png").resize((300,400)))
            img2=tk.Label(content2,image=image2,width=300,height=400)
            img2.image=image2
            img2.grid(column=0,row=0,rowspan=3,columnspan=4,sticky=(N,W,E,S),padx=10,pady=10)
        else:
            image3=ImageTk.PhotoImage(Image.open("compressed.png").resize((300,400)))
            img3=tk.Label(content3,image=image3,width=300,height=400)
            img3.image=image3
            img3.grid(column=0,row=0,rowspan=3,columnspan=4,sticky=(N,W,E,S),padx=10,pady=10)
      #  print("Mean square error = " + str(mse(image, image_after_compression)))

def encryption():
        im = Image.open('compressed.png')
        pixels = im.load()
        width,height=im.size
        
        key=[0]*(width*height)
        all_pixels=[]
        c=0
        for y in range(height):
            for x in range(width):
                cpixel=pixels[x,y]
                all_pixels.append(cpixel)
                key[c]=0
                c=c+1
            
        for y in range(height-1):
            for x in range(width-2):
                if all_pixels[x+(width*y)]==0 and all_pixels[(x+1)+(width*y)]==0 and all_pixels[x+2+(width*y)]==0:
                    if(all_pixels[(width*(y+1))+x+1])==0:
                        key[(width*(y+1))+x+1]=1
                    all_pixels[(width*(y+1))+x+1]=255
                elif all_pixels[x+(width*y)]==0 and all_pixels[x+1+(width*y)]==0 and all_pixels[x+2+(width*y)]==255:
                    if(all_pixels[(width*(y+1))+x+1])==0:
                        key[(width*(y+1))+x+1]=1
                    all_pixels[(width*(y+1))+x+1]=255
                elif all_pixels[x+(width*y)]==0 and all_pixels[x+1+(width*y)]==255 and all_pixels[x+2+(width*y)]==0:
                    if(all_pixels[(width*(y+1))+x+1])==0:
                        key[(width*(y+1))+x+1]=1
                    all_pixels[(width*(y+1))+x+1]=255
                elif all_pixels[x+(width*y)]==0 and all_pixels[x+1+(width*y)]==255 and all_pixels[x+2+(width*y)]==255:
                    if(all_pixels[(width*(y+1))+x+1])==255:
                        key[(width*(y+1))+x+1]=1
                    all_pixels[(width*(y+1))+x+1]=0
                elif all_pixels[x+(width*y)]==255 and all_pixels[x+1+(width*y)]==0 and all_pixels[x+2+(width*y)]==0:
                    if(all_pixels[(width*(y+1))+x+1])==255:
                        key[(width*(y+1))+x+1]=1
                    all_pixels[(width*(y+1))+x+1]=0
                elif all_pixels[x+(width*y)]==255 and all_pixels[x+1+(width*y)]==0 and all_pixels[x+2+(width*y)]==255:
                    if(all_pixels[(width*(y+1))+x+1])==255:
                        key[(width*(y+1))+x+1]=1
                    all_pixels[(width*(y+1))+x+1]=0
                elif all_pixels[x+(width*y)]==255 and all_pixels[x+1+(width*y)]==255 and all_pixels[x+2+(width*y)]==0:
                    if(all_pixels[(width*(y+1))+x+1])==255:
                        key[(width*(y+1))+x+1]=1
                    all_pixels[(width*(y+1))+x+1]=0
                elif all_pixels[x+(width*y)]==255 and all_pixels[x+1+(width*y)]==255 and all_pixels[x+2+(width*y)]==255:
                    if(all_pixels[(width*(y+1))+x+1])==0:
                        key[(width*(y+1))+x+1]=1
                    all_pixels[(width*(y+1))+x+1]=255
        cnt=0    
        for y in range(height):
            for x in range(width):
                pixels[x,y]=all_pixels[cnt]
                cnt=cnt+1
        #im.save('encrypt.png')
        
        pq = list()
        pq=[0]*int(width/2)
        #print(pq)
        keyrc=[4,5,7,3,6]

        lk=len(keyrc)
        
        #print("\nInitializing S")
        s=list()
        for i in range(int(width/2)):
            s.append(int(i))
        i=0
        j=0
        a=0
        k=list()
        while a<(width/2):
            i=(i+1)%int(width/2)
            j=(j+s[i])%int(width/2)
            temp=s[i]
            s[i]=s[j]
            s[j]=temp
            k.append(s[(s[i]+s[j])%(int(width/2))])
            pq[k[a]]=pq[k[a]]+1
            a=a+1
        s1=0
        s2=0
        for i in range(int(width/2)):
            if pq[i]==0:
                s1=s1+1
            if pq[i]>1:
                s2=s2+1
        #print(s1)
        #print(s2)
        i=0
        while i<(width/2):
            j=0
            l=0
            if pq[i]==1 or pq[i]==0:
                i=i+1
                continue
            if pq[i]>1:
                z=i
                while pq[j]!=0:
                    j=j+1
                while k[l]!=z:
                    l=l+1
                k[l]=j
                pq[j]=pq[j]+1
                pq[i]=pq[i]-1
            j=0
        for x in range(int(width/2)):
            for y in range(height):
                t=all_pixels[x+(y*width)]
                all_pixels[x+(y*width)]=all_pixels[k[x]+(y*width)+int(width/2)]
                all_pixels[k[x]+(y*width)+int(width/2)]=t
        cnt=0    
        for y in range(height):
            for x in range(width):
                pixels[x,y]=all_pixels[cnt]
                cnt=cnt+1
        save_obj(k,'imp')        
        
        im.save('encrypted.png')       
        
        if(rad_b_text.get()=='auto'):
            image2=ImageTk.PhotoImage(Image.open("encrypted.png").resize((300,400)))
            img2=tk.Label(content2,image=image2,width=300,height=400)
            img2.image=image2
            img2.grid(column=0,row=0,rowspan=3,columnspan=4,sticky=(N,W,E,S),padx=10,pady=10)
            encrypt=tk.Button(content2,text="DECRYPT",command=decryption)
            encrypt.grid(column=6,row=2,columnspan=2)
        else:
            image3=ImageTk.PhotoImage(Image.open("encrypted.png").resize((300,400)))
            img3=tk.Label(content3,image=image3,width=300,height=400)
            img3.image=image3
            img3.grid(column=0,row=0,rowspan=3,columnspan=4,sticky=(N,W,E,S),padx=10,pady=10)
            encrypt2=tk.Button(content3,text="DECRYPT",command=decryption)
            encrypt2.grid(column=6,row=2,columnspan=2)
            
def decryption():
    im = Image.open('encrypted.png')
    pixels = im.load()
    width,height=im.size
    
    k=load_obj('imp')
    
    key=[0]*(width*height)
    all_pixels=[]
    c=0
    for y in range(height):
        for x in range(width):
            cpixel=pixels[x,y]
            all_pixels.append(cpixel)
            key[c]=0
            c=c+1
    for x in range(int(width/2)):
      for y in range(height):
        t=all_pixels[x+(y*width)]
        all_pixels[x+(y*width)]=all_pixels[k[x]+(y*width)+int(width/2)]
        all_pixels[k[x]+(y*width)+int(width/2)]=t
    

    cnt=0    
    for y in range(height):
        for x in range(width):
            pixels[x,y]=all_pixels[cnt]
            cnt=cnt+1
    im.save('ground_test.png') 
            
    cnt=0
    for y in range(height):
        for x in range(width):
            if key[(width*(y))+x]==1:
                if all_pixels[(width*(y))+x]==0:
                    all_pixels[(width*(y))+x]=255
                else:
                    all_pixels[(width*(y))+x]=0
            pixels[x,y]=all_pixels[cnt]
            cnt=cnt+1
    im.save('decrypted.png')        
    if(rad_b_text.get()=='auto'):
            image2=ImageTk.PhotoImage(Image.open("decrypted.png").resize((300,400)))
            img2=tk.Label(content2,image=image2,width=300,height=400)
            img2.image=image2
            img2.grid(column=0,row=0,rowspan=3,columnspan=4,sticky=(N,W,E,S),padx=10,pady=10)
            encrypt=tk.Button(content2,text="ENCRYPT",command=encryption)
            encrypt.grid(column=6,row=2,columnspan=2)

    else:
            image3=ImageTk.PhotoImage(Image.open("decrypted.png").resize((300,400)))
            img3=tk.Label(content3,image=image3,width=300,height=400)
            img3.image=image3
            img3.grid(column=0,row=0,rowspan=3,columnspan=4,sticky=(N,W,E,S),padx=10,pady=10)
            encrypt2=tk.Button(content3,text="ENCRYPT",command=encryption)
            encrypt2.grid(column=6,row=2,columnspan=2)


#frame 1 
root=tk.Tk()
root.title('COMPRESSION TOOL')
content1=tk.Frame(root)
content1.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
   
img_text=StringVar()
img=tk.Label(content1,text="NO IMAGE",width=30,height=20)
img.grid(column=0,row=0,rowspan=3,columnspan=4,sticky=(N,W,E,S),padx=10,pady=10)

file_loc_text=StringVar()
file_loc=tk.Entry(content1,text='FILE LOCATION',textvariable=file_loc_text)
file_loc.grid(column=5,row=0,columnspan=2)

load_b=tk.Button(content1,text='LOAD',command=load)
load_b.grid(column=7,row=0,padx=10,pady=10)

rad_b_text=StringVar()
rad_b_text.set('auto')
rad_b1=tk.Radiobutton(content1,text='AUTOMATIC',variable=rad_b_text,value='auto')
rad_b1.grid(column=5,row=1)
rad_b2=tk.Radiobutton(content1,text='MANUAL',variable=rad_b_text,value='manual')
rad_b2.grid(column=5,row=2)

next_b=tk.Button(content1,text='NEXT',command=next1)
next_b.grid(column=7,row=2)

#frame 2
content2=tk.Frame(root)
content2.grid(column=0,row=0,sticky=(N,W,E,S))

img2=tk.Label(content2,text="NO IMAGE",width=30,height=20)
img2.grid(column=0,row=0,rowspan=3,columnspan=4,sticky=(N,W,E,S),padx=10,pady=10)
  
band=tk.Label(content2,text="BANDWIDTH")
band.grid(column=5,row=0)

cal_band=StringVar()
band_cal=tk.Entry(content2,textvariable=cal_band)
band_cal.grid(column=6,row=0,columnspan=2,padx=10,pady=10)
  
compress=tk.Button(content2,text="COMPRESS",command=compression)
compress.grid(column=6,row=1,columnspan=2)
  
encrypt=tk.Button(content2,text="ENCRYPT",command=encryption)
encrypt.grid(column=6,row=2,columnspan=2)

#frame3
content3=tk.Frame(root)
content3.grid(column=0,row=0,sticky=(N,W,E,S))

img3=tk.Label(content3,text="NO IMAGE",width=30,height=20)
img3.grid(column=0,row=0,rowspan=3,columnspan=4,sticky=(N,W,E,S),padx=10,pady=10)
  
band2=tk.Label(content3,text="BANDWIDTH")
band2.grid(column=5,row=0)

band_cal2=tk.Entry(content3,textvariable=cal_band)
band_cal2.grid(column=6,row=0,columnspan=2,padx=10,pady=10)

compress_ratio=tk.Label(content3,text="RATIO")
compress_ratio.grid(column=5,row=1)
range1=tk.Scale(content3,orient=HORIZONTAL,length=150,from_=1.0,to=12.0)
range1.grid(column=6,row=1)

  
compress2=tk.Button(content3,text="COMPRESS",command=compression)
compress2.grid(column=5,row=2)
  
encrypt2=tk.Button(content3,text="ENCRYPT",command=encryption)
encrypt2.grid(column=6,row=2)

raise_frame(content1)  
root.mainloop()

