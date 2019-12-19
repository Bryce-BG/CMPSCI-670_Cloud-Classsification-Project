import pandas as pd
import json

# read our predicted encoded pixels in (from the validation part of the training data)
# what our model predicted
df_predicted_labels = pd.read_csv("our-encoded-results/Kaivan_Train_results.csv")

# this is given with the dataset as the correct values
df_gold_labels = pd.read_csv("our-encoded-results/train(GOLD)_shortened.csv")

# images where we predicted no boxes for <type> and there were none in the gold labels
labels = ["Sugar", "Gravel", "Fish", "Flower"]

# initilize counts
# labels where we correctly didn't find any regions with our cloud types (indicates our system isn't over classifying)
true_neg = {}
# labels where we got at least ONE of the bounding boxes of <type> for an image (need more analysis to determine if we got them all)
true_pos = {}
#images where there should have been a mask of type x and our model didn't find a mask
false_neg = {}
# Could indicate we are predicting 1 class when there should be others predicted
false_pos = {}
for x in labels:
    true_neg[x] = 0  # initialize true negative counts for each type
    true_pos[x] = 0  # initialize true positives counts for each type
    false_neg[x] = 0  # initialize false negative counts for each type
    false_pos[x] = 0  # initialize false positives counts for each type

im_rows__predicted = {}
im_rows__gold = {}
curName = ""
for index, row in df_predicted_labels.iterrows():  # itterate over each image and see where things went wrong
    im_name = row['Image_Label'].split("_")[0]  # file name
    row_class = row['Image_Label'].split("_")[1]  # what class the runs in this row represent

    if curName == im_name:  # same image continue working
        im_rows__predicted[row_class] = row['EncodedPixels']  # store
        im_rows__gold[row_class] = df_gold_labels.iloc[index]['EncodedPixels']
    elif curName == "":  # starting a new image
        curName = im_name
        im_rows__predicted[row_class] = row['EncodedPixels']  # store
        im_rows__gold[row_class] = df_gold_labels.iloc[index]['EncodedPixels']
    else:  # switching to new image so need to analyze and updage counts of LAST image
        labelsLeft = ["Sugar", "Gravel", "Fish", "Flower"]

        # update counts based on image matchings
        for x in im_rows__predicted:
            # print(im_rows__predicted[x],  im_rows__gold[x])
            if pd.isna(im_rows__predicted[x]) and ~pd.isna(im_rows__gold[x]): #False Negitives
                false_neg[x] += 1  # no mask for an image of type x was found (should have been 1)
            elif ~pd.isna(im_rows__predicted[x]) and pd.isna(im_rows__gold[x]):  # False positive
                false_pos[x] += 1  # false positive for class x was generated on image.
            elif pd.isna(im_rows__predicted[x]) and pd.isna(im_rows__gold[x]):  # == "nan" and im_rows__gold[x] == 'nan':  # (shouldn't ever occur)
                print("impossible occured")
                true_neg[x] += 1
            else:
                true_pos[x] += 1
            labelsLeft.remove(
                x)  # add to list to remove remove from LabelLeft type observed (so we know which entries are missing for an image
        # for each labels Left add +1 to count TrueNegative
        for x in labelsLeft:  # update true negitives (where both were zero)
            true_neg[x] += 1

        # reset data structures for next image proccessing
        im_rows__gold = {}
        im_rows__predicted = {}
        curName = ""

print("True Neg count: ", true_neg)
print("True Pos count: ", true_pos)
print("False Pos count: ", false_pos)
print("False Neg count: ", false_neg)
