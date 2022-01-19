from scenedetect.detectors.content_detector import ContentDetector
import tensorflow as tf
from tensorflow.keras import layers,models,optimizers
import skvideo.io as svio
import numpy as np
from scenedetect import VideoManager
from scenedetect import SceneManager

def loadVideo(videoFilePath):
    numpy_video = svio.FFmpegReader(videoFilePath)
    # writer = svio.FFmpegWriter("afterLoading.mp4",outputdict={'-crf':'0','-pix_fmt':'yuv420p'})
    # for frame in numpy_video:
    #     writer.writeFrame(frame*255)
    # writer.close()
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

def formatAndSaveNumpyData(numpy_video,scene_list):
    samplesFile = open("E:\\Dataset\\samples.npy","ba+") #Open file in binary append mode
    labelsFile = open("E:\\Dataset\\labels.npy","ba+") #Open file in binary append mode
    data = (x for x in numpy_video.nextFrame()) #Use generator so entire file is not loaded into memory
    for scene in scene_list:
        count = scene[0].frame_num   
        limit = scene[1].frame_num
        one = next(data, None)
        two = next(data, None)
        three = next(data, None)
        while not count >= limit and three is not None: 
            np.save(labelsFile,two)
            np.save(samplesFile,np.reshape(np.stack((one,three),axis=-1),(one.shape[0],one.shape[1],one.shape[2]*2)))
            one = three
            two = next(data, None)
            three = next(data, None)
            count += 2
    samplesFile.close()
    labelsFile.close()
    return

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

def generator():
    samGen = np.load("E:\\Dataset\\samples.npy", mmap_mode="r")
    labGen = np.load("E:\\Dataset\\labels.npy", mmap_mode="r")
    yield (tf.convert_to_tensor(samGen),tf.convert_to_tensor(labGen))

def buildDataset():
    dataset = tf.data.Dataset.from_generator(generator,output_signature=(tf.TensorSpec(shape=(360, 504, 6), dtype=tf.uint8, name=None),tf.TensorSpec(shape=(360, 504, 3), dtype=tf.uint8, name=None)))
    return dataset

videoFilePath = "Dataset\\Charlie Chaplin_ Easy Street.mp4"
numpy_video = loadVideo(videoFilePath)
scene_list = split_scenes(numpy_video,videoFilePath) #Split video so that 'cuts' don't interfere with formatting dataset
#formatAndSaveNumpyData(numpy_video, scene_list)
dataset = buildDataset()
for scene in scene_list:
    count = scene[0].frame_num   
    limit = scene[1].frame_num
    writer = svio.FFmpegWriter("E:\\Scenes\\scene{0}.mp4".format(count))
    while count != limit:        
        for frame in numpy_video.nextFrame():
            writer.writeFrame(frame)
            count += 1
            break
    writer.close()        

inputs = layers.Input(shape=(360,504,6))
conv2d = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
conv2d_1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv2d)
max_pooling2d = layers.AveragePooling2D(pool_size=(2, 2))(conv2d_1)
conv2d_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(max_pooling2d)
conv2d_3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2d_2)
max_pooling2d_1 = layers.AveragePooling2D(pool_size=(2, 2))(conv2d_3)
conv2d_4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(max_pooling2d_1)
conv2d_5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv2d_4)
max_pooling2d_2 = layers.AveragePooling2D(pool_size=(2, 2))(conv2d_5)
conv2d_6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(max_pooling2d_2)
conv2d_7 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv2d_6)
max_pooling2d_3 = layers.AveragePooling2D(pool_size=(2, 2))(conv2d_7)
conv2d_8 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(max_pooling2d_3)
conv2d_9 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv2d_8)
up_sampling2d = layers.Concatenate()([layers.UpSampling2D(size=(2, 2))(conv2d_9),conv2d_7])
conv2d_10 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(up_sampling2d)
conv2d_11 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv2d_10)
up_sampling2d_1 = layers.Concatenate()([layers.UpSampling2D(size=(2, 2))(conv2d_11),conv2d_5])
conv2d_12 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up_sampling2d_1)
conv2d_13 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv2d_12)
up_sampling2d_2 = layers.Concatenate()([layers.UpSampling2D(size=(2,2))(conv2d_13),conv2d_3])
conv2d_14 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up_sampling2d_2)
conv2d_15 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2d_14)
up_sampling2d_3 = layers.Concatenate()([layers.UpSampling2D(size=(2,2))(conv2d_15),conv2d_1])
conv2d_16 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up_sampling2d_3)
conv2d_17 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2d_16)
outputs = layers.Conv2D(1, (1,1), activation='relu', padding='same')(conv2d_17)

model = models.Model(inputs=inputs, outputs=outputs, name="CNN")

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='huber')

history = model.fit(dataset,batch_size=64,epochs=100)

print("pause")