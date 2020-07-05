import pandas as pd
import numpy as np
import matchzoo as mz
import sys


if len(sys.argv)<2:
    print("Give fold number")
    sys.exit(0)

elif len(sys.argv)<3:
    print("Give path folder")
    sys.exit(0)


path = sys.argv[2]


queries_csv = pd.read_csv(path + "my_queries.csv", index_col=0)

docs_csv = pd.read_csv(path + "my_docs.csv")

#Ranked relevance
ranked_rels = pd.read_csv(path + "ranked_rels.csv",index_col=0)
ranked_rels.columns = ['id_left', 'id_right', 'label']


#Huston and Croft folds
folds = dict()
folds[1] = [302,303,309,316,317,319,323,331,336,341,356,357,370,373,378,381,383,392,394,406,410,411,414,426,428,433,447,448,601,607,608,612,617,619,635,641,642,646,647,654,656,662,665,669,670,679,684,690,692,700]
folds[2] = [301,308,312,322,327,328,338,343,348,349,352,360,364,365,369,371,374,386,390,397,403,419,422,423,424,432,434,440,446,602,604,611,623,624,627,632,638,643,651,652,663,674,675,678,680,683,688,689,695,698]
folds[3] = [306,307,313,321,324,326,334,347,351,354,358,361,362,363,376,380,382,396,404,413,415,417,427,436,437,439,444,445,449,450,603,605,606,614,620,622,626,628,631,637,644,648,661,664,666,671,677,685,687,693]
folds[4] = [320,325,330,332,335,337,342,344,350,355,368,377,379,387,393,398,402,405,407,408,412,420,421,425,430,431,435,438,616,618,625,630,633,636,639,649,650,653,655,657,659,667,668,672,673,676,682,686,691,697]
folds[5] = [304,305,310,311,314,315,318,329,333,339,340,345,346,353,359,366,367,372,375,384,385,388,389,391,395,399,400,401,409,416,418,429,441,442,443,609,610,613,615,621,629,634,640,645,658,660,681,694,696,699]

fold = int(sys.argv[1])


print("Starting fold: ", fold)
test_rels = ranked_rels[ranked_rels['id_left'].isin(folds[fold])]
test_queries = queries_csv[queries_csv['id_left'].isin(folds[fold])]
test_doc_id = np.unique(ranked_rels[ranked_rels['id_left'].isin(folds[fold])]['id_right'])
test_docs = docs_csv[docs_csv['id_right'].isin(test_doc_id)]

if fold!=5:
    dev_rels = ranked_rels[ranked_rels['id_left'].isin(folds[fold+1])]
    dev_queries = queries_csv[queries_csv['id_left'].isin(folds[fold+1])]
    dev_doc_id = np.unique(ranked_rels[ranked_rels['id_left'].isin(folds[fold+1])]['id_right'])
    train_rels = ranked_rels[-(ranked_rels['id_left'].isin(folds[fold]) | ranked_rels['id_left'].isin(folds[fold+1]))]
    train_queries = queries_csv[-(queries_csv['id_left'].isin(folds[fold]) | queries_csv['id_left'].isin(folds[fold+1]))]

else:
    dev_rels = ranked_rels[ranked_rels['id_left'].isin(folds[1])]
    dev_queries = queries_csv[queries_csv['id_left'].isin(folds[1])]
    dev_doc_id = np.unique(ranked_rels[ranked_rels['id_left'].isin(folds[1])]['id_right'])
    train_rels = ranked_rels[-(ranked_rels['id_left'].isin(folds[fold]) | ranked_rels['id_left'].isin(folds[1]))]
    train_queries = queries_csv[-(queries_csv['id_left'].isin(folds[fold]) | queries_csv['id_left'].isin(folds[1]))]

dev_docs = docs_csv[docs_csv['id_right'].isin(dev_doc_id)]


print("Data folded...")


test_queries.set_index('id_left',inplace=True)
test_queries = test_queries.dropna()
test_docs.set_index('id_right',inplace=True)
test_docs = test_docs.dropna()
test_rels = test_rels.reset_index()
test_rels = test_rels.drop(columns='index')
test_rels = test_rels.dropna()

dev_queries.set_index('id_left',inplace=True)
dev_queries = dev_queries.dropna()
dev_docs.set_index('id_right',inplace=True)
dev_docs = dev_docs.dropna()
dev_rels = dev_rels.reset_index()
dev_rels = dev_rels.drop(columns='index')
dev_rels = dev_rels.dropna()

train_queries.set_index('id_left',inplace=True)
train_queries = train_queries.dropna()

train_rels = train_rels.reset_index()
train_rels = train_rels.drop(columns='index')
train_rels = train_rels.dropna()

test_pack = mz.data_pack.DataPack(relation=test_rels, left=test_queries, right=test_docs)
dev_pack = mz.data_pack.DataPack(relation=dev_rels, left=dev_queries, right=dev_docs)
train_pack = mz.data_pack.DataPack(relation=train_rels, left=train_queries, right=docs_csv)

train_pack.right.set_index('id_right',inplace=True)

preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=10, fixed_length_right=500, remove_stop_words=False)
print("train preprocessing...")
train_pack_processed = preprocessor.fit_transform(train_pack)
print("dev preprocessing...")
dev_pack_processed = preprocessor.transform(dev_pack)
print("test preprocessing...")
test_pack_processed = preprocessor.transform(test_pack)


print("Saving datapacks")
train_pack_processed.save(path + "robust_train_fold_"+str(fold))
dev_pack_processed.save(path + "robust_dev_fold_"+str(fold))
test_pack_processed.save(path + "robust_test_fold_"+str(fold))
preprocessor.save(path + "robust_preprocessor_fold_"+str(fold))
