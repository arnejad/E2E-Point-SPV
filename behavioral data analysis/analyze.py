import pickle
from utils import *
DATA_DIR = "DIR TO THE LOCATION OF PROCESSED DATA FROM data_process.py"

with open(os.path.join(DATA_DIR, "processed_data","participantFeats-GSPV-1.pickle" ), "rb") as input_file:
    gspv1_p = pickle.load(input_file)

with open(os.path.join(DATA_DIR, "processed_data","participantFeats-GSPV-2.pickle" ), "rb") as input_file:
    gspv2_p = pickle.load(input_file)


with open(os.path.join(DATA_DIR, "processed_data","participantFeats-ED-1.pickle" ), "rb") as input_file:
    ed1_p = pickle.load(input_file)

with open(os.path.join(DATA_DIR, "processed_data","participantFeats-ED-2.pickle"), "rb") as input_file:
    ed2_p = pickle.load(input_file)



plot_images_in_2d_space(np.array(gspv1_p["stimuliSeq"]), np.array(gspv1_p["hitsSeq"]), np.array(gspv1_p["dursSeq"]), image_folder=DATA_DIR+"/images")

# COMPARISON BETWEEN FIRST BLOCKS OF TWO METHODS
ledgendTitles = ["GSPV", "ED"]

visualizeShadedLinePlot2(gspv1_p["sacNumSeq"], ed1_p["sacNumSeq"], "Number of saccades per trial", "Number of occurance", labels=ledgendTitles)

visualizeShadedLinePlot2(gspv1_p["dursSeq"], ed1_p["dursSeq"], "Response Duration per trial", "Number of occurance", labels=ledgendTitles)




ledgendTitles = ["Block 1 - Point-SPV", "Block 2 - Canny", "Block 1 - Canny", "Block 2 - Point-SPV"]
visParallelViolinPlot2(np.mean(gspv1_p["dursSeq"], axis=1), np.mean(gspv2_p["dursSeq"], axis=1), np.mean(ed1_p["dursSeq"], axis=1), np.mean(ed2_p["dursSeq"], axis=1), "Response time", "Time (ms)", ledgendTitles)
visParallelViolinPlot2(np.array(gspv1_p["accuracy"]), np.array(gspv2_p["accuracy"]), np.array(ed1_p["accuracy"]), np.array(ed2_p["accuracy"]), "Accuracy", "Accuracy [0-1]", ledgendTitles)
visParallelViolinPlot2(np.mean(gspv1_p["sacNumSeq"], axis=1), np.mean(gspv2_p["sacNumSeq"], axis=1), np.mean(ed1_p["sacNumSeq"], axis=1), np.mean(ed2_p["sacNumSeq"], axis=1), "Saccade frequency", "Number of saccades", ledgendTitles)



# heatmap visualization

with open(os.path.join(DATA_DIR, "processed_data","imageFeats-GSPV-1.pickle" ), "rb") as input_file:
    gspv1_i = pickle.load(input_file)

gspv1_i["avgHeatMaps"] = averageHeatMaps(gspv1_i)
# visualizeHeatMaps(gspv1_i, DATA_DIR, "gspv-1", DATE)
gspv1_i["avgFixAreaImage"] = np.sum(gspv1_i["avgHeatMaps"], axis=(1,2))/2073600
gspv1_i["avgFixAreaPar"] = (np.sum(gspv1_i["guasMaps"], axis=(0,2,3))/50)/2073600
gspv1_i_avgFixAreaPar = gspv1_i["avgFixAreaPar"]
del gspv1_i


with open(os.path.join(DATA_DIR, "processed_data","imageFeats-ED-1.pickle" ), "rb") as input_file:
    ed1_i = pickle.load(input_file)

ed1_i["avgHeatMaps"] = averageHeatMaps(ed1_i)
# visualizeHeatMaps(ed1_i, DATA_DIR, "ed-1", DATE)
ed1_i["avgFixAreaImage"] = np.sum(ed1_i["avgHeatMaps"], axis=(1,2))/2073600
ed1_i["avgFixAreaPar"] = (np.sum(ed1_i["guasMaps"], axis=(0,2,3))/50)/2073600
ed1_i_avgFixAreaPar = ed1_i["avgFixAreaPar"]
del ed1_i

with open(os.path.join(DATA_DIR, "processed_data","imageFeats-GSPV-2.pickle" ), "rb") as input_file:
    gspv2_i = pickle.load(input_file)

gspv2_i["avgHeatMaps"] = averageHeatMaps(gspv2_i)
# visualizeHeatMaps(gspv2_i, DATA_DIR, "gspv-2", DATE)
gspv2_i["avgHeatMaps"] = averageHeatMaps(gspv2_i)
gspv2_i["avgFixAreaImage"] = np.sum(gspv2_i["avgHeatMaps"], axis=(1,2))/2073600
gspv2_i["avgFixAreaPar"] = (np.sum(gspv2_i["guasMaps"], axis=(0,2,3))/50)/2073600
gspv2_i_avgFixAreaPar = gspv2_i["avgFixAreaPar"]
del gspv2_i


with open(os.path.join(DATA_DIR, "processed_data","imageFeats-ED-2.pickle" ), "rb") as input_file:
    ed2_i = pickle.load(input_file)

ed2_i["avgHeatMaps"] = averageHeatMaps(ed2_i)
# visualizeHeatMaps(ed2_i, DATA_DIR, "ed-2", DATE)
ed2_i["avgHeatMaps"] = averageHeatMaps(ed2_i)
ed2_i["avgFixAreaImage"] = np.sum(ed2_i["avgHeatMaps"], axis=(1,2))/2073600
ed2_i["avgFixAreaPar"] = (np.sum(ed2_i["guasMaps"], axis=(0,2,3))/50)/2073600
ed2_i_avgFixAreaPar = ed2_i["avgFixAreaPar"]
del ed2_i



visParallelViolinPlot2(gspv1_i_avgFixAreaPar, gspv2_i_avgFixAreaPar, ed1_i_avgFixAreaPar, ed2_i_avgFixAreaPar, "Stimulus coverage", "Coverage proportion [0-1]", ledgendTitles)





print("done")