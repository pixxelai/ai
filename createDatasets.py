import pandas as pd
import h5py as h5
import numpy as np

#from hist import histogramFunc

############################################
""" import method that returns histograms."""
############################################


def createDataset(state, crop, season, file_name = None, read_file = None ):
    
    #Get required subset of the entire csv file
    if (read_file is None):
        df = pd.read_csv('apy.csv')
    else:
        df = pd.read_csv(read_file)
    
    df_query = df.copy()
    df_query = df_query[df_query['State_Name'].str.contains(state)]
    df_query = df_query[df_query['Crop_Year']>=2001]
    df_query = df_query[df_query['Season'].str.contains(season)]
    df_query = df_query[df_query['Crop'].str.contains(crop)]
    
    
    if(file_name is None):
        file_name = (state+'_'+crop)
    df_query.to_csv(file_name+'.csv', encoding = 'utf-8', index = False)
    
    
    
    
    f = h5.File(file_name+'.hdf5', 'w')
    
    #Get data and corresponding labels
    dataset = np.asarray([])
    labels = []
    for row in df_query.itertuples(index = False):
        
        district = getattr(row, 'District_Name')
        year = getattr(row, 'Crop_Year')
        
        ##########################################################
        """ Change histogramFunc() with name of actual function"""
        ##########################################################
        
        hist = histogramFunc(district, year)
        hist = np.reshape(hist, (1, hist.shape[0], hist.shape[1], hist.shape[2]))
        prod = getattr(row,'Production')
        
        labels.append(prod)

        if(dataset.size):
            dataset = np.concatenate((dataset,hist))
        else:
            dataset = hist

    print(dataset.shape)

    #Array is of the shape [number of histogram sets,
    #                       number of histograms in each year,
    #                       number of channels,
    #                       maximum value of pixel in each image] 
    #dataset = np.swapaxes(dataset, 0, 3)

    
    # Randomize data order
    randomize = np.arange(len(labels))
    np.random.shuffle(randomize)
    dataset = dataset[randomize,:,:,:]
    labels = [labels[i] for i in randomize]
    
    #Create datasets
    f.create_dataset("train_X", (int(0.8*len(labels))+1,dataset.shape[1],dataset.shape[2],dataset.shape[3]), np.float32)
    f["train_X"][...] = dataset[:int(0.8*len(labels))+1,:,:,:]
    
    f.create_dataset("val_X", (int(0.9*len(labels))-int(0.8*len(labels)),dataset.shape[1],dataset.shape[2],dataset.shape[3]), np.float32)
    f["val_X"][...] = dataset[int(0.8*len(labels))+1:int(0.9*len(labels))+1,:,:,:]
    
    f.create_dataset("test_X", (len(labels)-int(0.9*len(labels)+1),dataset.shape[1],dataset.shape[2],dataset.shape[3]), np.float32)
    f["test_X"][...] = dataset[int(0.9*len(labels))+1:,:,:,:]
    
    f.create_dataset("train_labels", (int(0.8*len(labels))+1,), np.float32)
    f["train_labels"][...] = labels[:int(0.8*len(labels))+1]
    
    f.create_dataset("val_labels", (int(0.9*len(labels))-int(0.8*len(labels)),), np.float32)
    f["val_labels"][...] = labels[int(0.8*len(labels))+1:int(0.9*len(labels))+1]
    
    f.create_dataset("test_labels", (len(labels)-int(0.9*len(labels)+1),), np.float32)
    f["test_labels"][...] = labels[int(0.9*len(labels))+1:]
    
    
        
        
