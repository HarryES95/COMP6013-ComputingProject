import random
from scenedetect.detectors.content_detector import ContentDetector
import tensorflow as tf
from tensorflow.keras import layers,models,optimizers
import skvideo.io as svio
import numpy as np
from scenedetect import VideoManager
from scenedetect import SceneManager
import time
MODEL_HEIGHT = 160
MODEL_WIDTH = 320

def loadVideo(videoFilePath):
    numpy_video = svio.FFmpegReader(videoFilePath)
    return numpy_video

def formatData(dataset_numpy):    
    #Split data into pre and post frames / labels for those frames
    temp_dataset = dataset_numpy[0:len(dataset_numpy):2] #Use Numpy 'fancy indexing' to select every other frame
    labels = dataset_numpy[1:len(dataset_numpy):2] #Use Numpy 'fancy indexing' to select every other frame phase shifted +1
    #Stack dataset frames into shape (2,64,64)
    count = 0
    temp = np.empty((1,2,dataset_numpy.shape[-3],dataset_numpy.shape[-2]))
    while count != len(temp_dataset)-1:
        temp = np.concatenate((temp,np.reshape(np.stack((temp_dataset[count],temp_dataset[count+1])),(1,2,dataset_numpy.shape[-3],dataset_numpy.shape[-2]))))
        count += 1
    dataset = np.delete(temp,0,axis=0)
    return dataset, labels

def formatAndSaveNumpyData(numpy_video,scene_list,split):
    samplesFile = open("E:\\Dataset\\samples.npy","ba+") #Open file in binary append mode
    labelsFile = open("E:\\Dataset\\labels.npy","ba+") #Open file in binary append mode
    data = (x for x in numpy_video.nextFrame()) #Use generator so entire file is not loaded into memory
    hFactor,wFactor,hTPad,hBPad,wLPad,wRPad = split
    size = 0
    test = (numpy_video.inputframenum*hFactor*wFactor)-(len(scene_list)*7)
    for scene in scene_list:
        count = scene[0].frame_num   
        limit = scene[1].frame_num
        one = next(data, None)
        two = next(data, None)
        three = next(data, None)
        count += 3
        for i in range(0,2): #Skip first chunk of frames as transistions are not completely smooth.
            one = three
            two = next(data, None)
            three = next(data, None)
            count += 2
        while not count >= limit and three is not None: 
            #Split the numpy arrays into sections of 4 so that the model does not take up so much memory
            label = two
            sample = np.dstack((one,three))
            label,sample = np.vsplit(label,hFactor),np.vsplit(sample,hFactor)
            label,sample = [np.hsplit(l,wFactor) for l in label],[np.hsplit(s,wFactor) for s in sample]
            label,sample = np.reshape(label,(hFactor*wFactor,int(numpy_video.outputheight/hFactor),int(numpy_video.outputwidth/wFactor),3)),np.reshape(sample,(hFactor*wFactor,int(numpy_video.outputheight/hFactor),int(numpy_video.outputwidth/wFactor),6))
            label,sample = np.moveaxis(label,-1,1),np.moveaxis(sample,-1,1)
            #pad
            label,sample = np.pad(label,((0,0),(0,0),(hTPad,hBPad),(wLPad,wRPad)),mode="constant"),np.pad(sample,((0,0),(0,0),(hTPad,hBPad),(wLPad,wRPad)),mode="constant")
            #Save sections to the disk
            labelsFile.write(np.ascontiguousarray(label))
            samplesFile.write(np.ascontiguousarray(sample))
            #Move along the video
            one = three
            two = next(data, None)
            three = next(data, None)
            count += 2
            size += 1
    samplesFile.close()
    labelsFile.close()
    return size

def calculateSplitFactor(height,width):
    if(height%MODEL_HEIGHT == 0):
        hFactor = height/MODEL_HEIGHT
        hTPad = 0
        hBPad = 0
    else:
        hFactor = (height//MODEL_HEIGHT)+1
        if((MODEL_HEIGHT-(height/hFactor))%2 == 0):
            hTPad = (MODEL_HEIGHT-(height/hFactor))/2
            hBPad = (MODEL_HEIGHT-(height/hFactor))/2
        else:            
            hTPad = (MODEL_HEIGHT-(height/hFactor))/2
            hBPad = ((MODEL_HEIGHT-(height/hFactor))/2)+1
    if(width%MODEL_WIDTH == 0):
        wFactor = width/MODEL_WIDTH
        wLPad = 0
        wRPad = 0
    else:
        wFactor = (width//MODEL_WIDTH)+1
        if((MODEL_WIDTH-(width/wFactor))%2 == 0):
            wLPad = (MODEL_WIDTH-(width/wFactor))/2
            wRPad = (MODEL_WIDTH-(width/wFactor))/2
        else:            
            wLPad = (MODEL_WIDTH-(width/wFactor))/2
            wRPad = ((MODEL_WIDTH-(width/wFactor))/2)+1
    return int(hFactor),int(wFactor),int(hTPad),int(hBPad),int(wLPad),int(wRPad)

def split_scenes(numpy_video,videoFilePath):
    video_manager = VideoManager([videoFilePath])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=10.0)
    )
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    return scene_list

def generator(size):
    samGen = np.memmap("E:\\Dataset\\samples.npy", mode="r+",shape=(size,6,160,320))
    labGen = np.memmap("E:\\Dataset\\labels.npy", mode="r+",shape=(size,3,160,320))
    for i in random.sample(range(size),size):
        yield (samGen[i],labGen[i])

def buildDataset(size):
    dataset = tf.data.Dataset.from_generator(generator,output_signature=(tf.TensorSpec(shape=(6,160,320), dtype=tf.uint8, name=None),tf.TensorSpec(shape=(3,160,320), dtype=tf.uint8, name=None)),args=(size,))
    dataset = dataset.batch(32)
    dataset = dataset.map(lambda x,y: (tf.cast(x,tf.float16),tf.cast(y,tf.float16))
                         ).map(lambda x,y: (x/255.,y/255.))
    return dataset

def enhance(numpy_video, model, split):
    writer = svio.FFmpegWriter("E:\\Results\\FloorEnhanced.mp4", inputdict={
      '-r': "50",
    },
    outputdict={
      '-vcodec': 'libx264',
      '-pix_fmt': 'yuv420p',
      '-r': "50"
    })
    hFactor,wFactor,hTPad,hBPad,wLPad,wRPad = split
    data = (x for x in numpy_video.nextFrame())
    one = next(data,None)
    two = next(data,None)
    count = 0
    writer.writeFrame(one)
    while two is not None:
        stack = np.dstack((one,two))
        quadrants = [np.hsplit(q,wFactor) for q in np.vsplit(stack,hFactor)]
        quadrants = [np.moveaxis(q,-1,1) for q in [y for y in quadrants]]
        quadrants = np.reshape(quadrants,(hFactor*wFactor,6,int(numpy_video.outputheight/hFactor),int(numpy_video.outputwidth/wFactor)))
        #pad
        quadrants = np.pad(quadrants,( (0,0),(0,0),(hTPad,hBPad),(wRPad,wLPad)),mode="constant")
        predict = model.predict(tf.cast(quadrants,tf.float16)/255.)
        #crop
        predict = (predict*255.).astype(np.uint8)
        sequence = np.insert(quadrants,[3],predict,axis=1)
        sequence = sequence[:,:,hTPad:MODEL_HEIGHT-hBPad,wLPad:MODEL_WIDTH-wRPad]
        sequence = np.dstack(np.split(sequence,hFactor))
        if(wFactor == 1):
            sequence = np.dstack(np.split(sequence,wFactor))
            sequence = np.squeeze(sequence)
        else:
            sequence = np.dstack(np.squeeze(np.split(sequence,wFactor)))
        sequence = np.moveaxis(sequence,0,-1)
        [writer.writeFrame(f) for f in np.dsplit(sequence,3)[1:]]
        one = two
        two = next(data,None)
        count += 1
    writer.close()
    return

def loss(y,y_pred,e=0.001):
    return tf.math.sqrt(tf.math.square(y-y_pred+tf.math.square(e)))

def ssim(y_true,y_pred):
    return tf.reduce_mean(tf.image.ssim(tf.transpose(y_true, perm=[0,2,3,1]),tf.transpose(y_pred, perm=[0,2,3,1]),1.0))

#size = 503140
videoFilePath = "E:\\Results\\cureEnhanced.mp4"
numpy_video = loadVideo(videoFilePath)
split = calculateSplitFactor(numpy_video.outputheight,numpy_video.outputwidth)
scene_list = split_scenes(numpy_video,videoFilePath) #Split video so that 'cuts' don't interfere with formatting dataset
size = formatAndSaveNumpyData(numpy_video,scene_list,split)*split[0]*split[1]
dataset = buildDataset(size)
#split_size = int((size/32)*0.8)
#train_data = dataset.take(split_size)
#test_data = dataset.skip(split_size)

inputs = layers.Input(shape=(6,160,320))
conv2d = layers.Conv2D(32, (7,7), activation='relu', padding='same', data_format="channels_first")(inputs)
conv2d_1 = layers.Conv2D(32, (7,7), activation='relu', padding='same', data_format="channels_first")(conv2d)
max_pooling2d = layers.AveragePooling2D(pool_size=(2, 2), data_format="channels_first")(conv2d_1)
conv2d_2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same', data_format="channels_first")(max_pooling2d)
conv2d_3 = layers.Conv2D(64, (5, 5), activation='relu', padding='same', data_format="channels_first")(conv2d_2)
max_pooling2d_1 = layers.AveragePooling2D(pool_size=(2, 2), data_format="channels_first")(conv2d_3)
conv2d_4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first")(max_pooling2d_1)
conv2d_5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first")(conv2d_4)
max_pooling2d_2 = layers.AveragePooling2D(pool_size=(2, 2), data_format="channels_first")(conv2d_5)
conv2d_6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_first")(max_pooling2d_2)
conv2d_7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_first")(conv2d_6)
max_pooling2d_3 = layers.AveragePooling2D(pool_size=(2, 2), data_format="channels_first")(conv2d_7)
conv2d_8 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first")(max_pooling2d_3)
conv2d_9 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first")(conv2d_8)
up_sampling2d = layers.Concatenate(axis=1)([layers.UpSampling2D(size=(2, 2), data_format="channels_first")(conv2d_9),conv2d_7])
conv2d_10 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_first")(up_sampling2d)
conv2d_11 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_first")(conv2d_10)
up_sampling2d_1 = layers.Concatenate(axis=1)([layers.UpSampling2D(size=(2, 2), data_format="channels_first")(conv2d_11),conv2d_5])
conv2d_12 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first")(up_sampling2d_1)
conv2d_13 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first")(conv2d_12)
up_sampling2d_2 = layers.Concatenate(axis=1)([layers.UpSampling2D(size=(2,2), data_format="channels_first")(conv2d_13),conv2d_3])
conv2d_14 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first")(up_sampling2d_2)
conv2d_15 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first")(conv2d_14)
up_sampling2d_3 = layers.Concatenate(axis=1)([layers.UpSampling2D(size=(2,2), data_format="channels_first")(conv2d_15),conv2d_1])
conv2d_16 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_first")(up_sampling2d_3)
conv2d_17 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_first")(conv2d_16)
outputs = layers.Conv2D(3, (1,1), activation='sigmoid', padding='same', data_format="channels_first")(conv2d_17)

# inputs = layers.Input(shape=(6,160,320))
# conv2d = layers.Conv2D(32, (7,7), activation='relu', padding='same', data_format="channels_first")(inputs)
# max_pooling2d = layers.AveragePooling2D(pool_size=(2, 2), data_format="channels_first")(conv2d)
# conv2d_2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same', data_format="channels_first")(max_pooling2d)
# max_pooling2d_1 = layers.AveragePooling2D(pool_size=(2, 2), data_format="channels_first")(conv2d_2)
# conv2d_4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first")(max_pooling2d_1)
# max_pooling2d_2 = layers.AveragePooling2D(pool_size=(2, 2), data_format="channels_first")(conv2d_4)
# conv2d_6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_first")(max_pooling2d_2)
# max_pooling2d_3 = layers.AveragePooling2D(pool_size=(2, 2), data_format="channels_first")(conv2d_6)
# conv2d_8 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first")(max_pooling2d_3)
# up_sampling2d = layers.Concatenate(axis=1)([layers.UpSampling2D(size=(2, 2), data_format="channels_first")(conv2d_8),conv2d_6])
# conv2d_10 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_first")(up_sampling2d)
# up_sampling2d_1 = layers.Concatenate(axis=1)([layers.UpSampling2D(size=(2, 2), data_format="channels_first")(conv2d_10),conv2d_4])
# conv2d_12 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first")(up_sampling2d_1)
# up_sampling2d_2 = layers.Concatenate(axis=1)([layers.UpSampling2D(size=(2,2), data_format="channels_first")(conv2d_12),conv2d_2])
# conv2d_14 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first")(up_sampling2d_2)
# up_sampling2d_3 = layers.Concatenate(axis=1)([layers.UpSampling2D(size=(2,2), data_format="channels_first")(conv2d_14),conv2d])
# conv2d_16 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_first")(up_sampling2d_3)
# outputs = layers.Conv2D(3, (1,1), activation='sigmoid', padding='same', data_format="channels_first")(conv2d_16)

# checkpoint = tf.keras.callbacks.ModelCheckpoint(
#     filepath = "E:\\Model\\checkpoint",
#     monitor = "loss",
#     mode = "min",
#     save_weights_only = True,
#     save_freq = "epoch"
# )

class Progress(tf.keras.callbacks.Callback):

    def __init__(self,model):
        super(Progress,self).__init__()

    def on_train_batch_end(self,batch,logs=None):
        print(batch)

model = models.Model(inputs=inputs, outputs=outputs, name="CNN")

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss=loss,
              metrics=ssim)

#model.load_weights("E:\\Model\\checkpoint")
#model.save('E:\\Model\\ChaplinCNN_Test')
history = model.fit(dataset,epochs=5,callbacks=[Progress(model)])

#test = model.evaluate(test_data)
#test = model.evaluate(dataset)
#enhance(numpy_video,model,split)

#print(test)

def speedTest(dataset,split):
    data = dataset.skip(50).take(1).get_single_element()
    samples = data[0]
    labels = data[1]
    sample = tf.stack(tf.squeeze(tf.split(samples,32)[0:split[0]*split[1]]))
    start = time.time()
    predict = model.predict(sample)
    end = time.time()
    elapsed = end - start
    print(elapsed)

#speedTest(dataset,split)

print("Pause")
