import pickle
# import pandas
# import numpy, scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import accuracy_score

from utils import *
from gaussian import fixationGaussian
from config import *


for block in [1,2]:


    folders = filterMethodRecs(os.listdir(os.path.join(DATA_DIR, "data")), METHOD, block)

    imagesList = os.listdir(os.path.join(DATA_DIR, "images"))

    participantFeats = {
        "dursSeq": [],
        "groundTruthSeq": [],
        "sacNumSeq": [],
        "stimuliSeq": [],
        "responseSeq": [],
        "hitsSeq" : [],
        "hitsAnimal": [],
        "hitsObject": [],
        "accuracy" : [],
        "heatArea": []
    }


    imageFeats = {
        "name": [],
        "label": [],
        "hits": [0] * NUM_OF_STIMULI,
        "numOfSacs": np.zeros((NUM_OF_STIMULI, len(folders))),
        "guasMaps":np.zeros((NUM_OF_STIMULI, len(folders), STIMULI_SIZE[0], STIMULI_SIZE[1])),
        "avgHeatMaps": []
    }

    # saving the name of the stimuli and their corresponding category
    imageFeats["name"] = imagesList
    imageFeats["label"] = [1 if int(s.split('_')[1].split('.')[0]) > (NUM_OF_STIMULI/2) else 0 for s in imagesList] # images with a number less than 25 are animals


    print("Found " + str(len(folders)) + " recordings")

    for p, experiment in enumerate(folders):

        files = os.listdir(os.path.join(DATA_DIR, "data", experiment))
        files = [k for k in files if '.' not in k]
        files = [k for k in files if 'rend' not in k]
        file = open(os.path.join(DATA_DIR, "data", experiment, files[-1]) , 'rb')
        data = pickle.load(file)
        file.close()
        with open(os.path.join(DATA_DIR, "data", experiment, experiment+".asc"), 'r') as file:
            em_asc = file.readlines()

        # 
            

        #duration of each stimulus
        dursSeq = []
        hitsSeq = []
        responseSeq = []
        groundTruthSeq = []
        stimuliSeq = []
        sacNumSeq = []
        hitsAnimal = 0
        hitsObject = 0
        heatArea = []

        for i, trial in enumerate(data['eye_data_gaze']):
            if SKIP_FIRST_TRIALS != None:
                if i < 5: continue
            dur = trial[-1][0] - trial[0][0]
            dursSeq.append(dur)


        # accuracy of responses
        for i, trial_end in enumerate(data['end_trial']):
            
            if SKIP_FIRST_TRIALS != None:
                if i < 5: continue
                
            print("computing experiment " + experiment + " trial " + str(i))

            img = re.sub(r'^.*(?=img)', '', data['eye_data_img'][i][0][1])     # get the image name and directory
            keyResponse = trial_end[1]              # get the pressed key

            stimuliSeq.append(img)

            # correcting misunderstanding with participant number 6
            #  who pressed up button instead of left during the experiment
            if experiment == '1_1_06_2024_06_18_15_50' and keyResponse == 'key_pressed_up':
                keyResponse = 'key_pressed_left'

            lbl, response, hitOrMiss = responseCheck(img, keyResponse)   # check the type of stimulus and the accuracy of response 
            groundTruthSeq.append(lbl)                # add the type of stimulus to the sequence of appearances
            hitsSeq.append(hitOrMiss)               # add the hit or miss to the sequence of hit or miss
            responseSeq.append(response)
            
            
            if lbl == 0: # if image was for an animal
                hitsAnimal += hitOrMiss             # add to the counter of hits for animals
            else: hitsObject += hitOrMiss           # add to the counter of hits for objects


            #image-based info
            stim_et = [row for row in data['eye_data_gaze'][i] if len(row) == 4]
            stim_et = np.array(stim_et)

            sacs = countSacs_el(em_asc, stim_et[0,0], stim_et[-1,0])
            # sacs = 0
            sacNumSeq.append(sacs)
            
            stim_et = stim_et[:, 1:3]
            

            gaussian_map = fixationGaussian(stim_et)
            heatAr = np.sum(gaussian_map)
            gaussian_map[gaussian_map < HEATMAP_THRESH] = 0
            heatArea.append(heatAr)

            
            imageFeats = stimulusAppend(imageFeats, img, p, sacs, gaussian_map, hitOrMiss)
            
        # if p == 2: break # early finishing for debugging
        
        # print(np.mean(stds))
        score = accuracy_score(responseSeq, groundTruthSeq)

        print(score)
        # storing the extracted information from this block 
        participantFeats = participantAppend(participantFeats, dursSeq, hitsSeq,\
                                            groundTruthSeq, hitsAnimal, hitsObject,\
                                                responseSeq, score, stimuliSeq, sacNumSeq, heatArea)

        
    with open('imageFeats-'+METHOD+'-'+str(block)+'.pickle', 'wb') as handle:
        pickle.dump(imageFeats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('participantFeats-'+METHOD+'-'+str(block)+'.pickle', 'wb') as handle:
        pickle.dump(participantFeats, handle, protocol=pickle.HIGHEST_PROTOCOL)

