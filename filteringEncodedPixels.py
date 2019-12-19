
import pandas as pd
#read our predicted encoded pixels in (from the validation part of the training data)
df_PredictedData = pd.read_csv("our-encoded-results/resultValidationReal.csv")


#read the gold_labels for the entire training data in
df_OriginalData = pd.read_csv("our-encoded-results/train.csv")
print(df_OriginalData.head())



#filter the train.csv so only entries that are in our validation_set exist
# m = df_OriginalData.Image_Label.isin(df_PredictedData.Image_Label)
# df_validation_gold_labels = df_OriginalData[m] #apply the filter
#
# #write to file
# df_validation_gold_labels.to_csv(r'our-encoded-results/TrueValidationLabels.csv', index=None, header=True)

df_new = df_OriginalData.copy()
# df_new = df_new[df_new.Image_Label.isin(df_PredictedData['Image_Label'])] #filter
a = df_PredictedData['Image_Label'].tolist()


df_new = df_new[~df_new['Image_Label'].isin(a)]
print(df_new.head())
df_new.to_csv(r'our-encoded-results/TrueValidationLabels2.csv', index=None, header=True)





