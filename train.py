
import pandas as pd
import keras
import numpy as np
import matchzoo as mz
import json
import sys

if len(sys.argv)<2:
   print("Give fold number")
   sys.exit(0)
elif len(sys.argv)<3:
    print("Give path folder")
    sys.exit(0)


path = sys.argv[2]



print("loading embedding ...")
glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
print("embedding loaded as `glove_embedding`")

fold = sys.argv[1]

print("Loading fold:  ",fold)
preprocessor = mz.load_preprocessor(path + "robust_preprocessor_fold_"+fold)

print("preprocessor context:   ", preprocessor.context)

ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss(num_neg=1))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=20),
    mz.metrics.MeanAveragePrecision(),
    mz.metrics.Precision(k=20)
]

print("ranking task ok")

bin_size = 30
model = mz.models.DRMM()
model.params.update(preprocessor.context)
model.params['input_shapes'] = [[10,], [10, bin_size,]]
model.params['task'] = ranking_task
model.params['mask_value'] = -1
model.params['embedding_output_dim'] = glove_embedding.output_dim
model.params['mlp_num_layers'] = 2
model.params['mlp_num_units'] = 5
model.params['mlp_num_fan_out'] = 1
model.params['mlp_activation_func'] = 'tanh'
model.params['optimizer'] = 'adadelta'
model.build()
model.compile()
model.backend.summary()

print("model params set")

train_pack_processed = mz.load_data_pack(path + "robust_train_fold_"+fold)

dev_pack_processed = mz.load_data_pack(path + "robust_dev_fold_"+fold)

print("datapacks OK")



embedding_matrix = glove_embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
#normalize the word embedding for fast histogram generating.
l2_norm = np.sqrt((embedding_matrix*embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]


model.load_embedding_matrix(embedding_matrix)

print("embedding matrix loaded")

hist_callback = mz.data_generator.callbacks.Histogram(embedding_matrix, bin_size=30, hist_mode='LCH')

pred_generator = mz.DataGenerator(dev_pack_processed, mode='point', callbacks=[hist_callback])

print("pred generator")

pred_x, pred_y = pred_generator[:]
evaluate = mz.callbacks.EvaluateAllMetrics(model,
                                           x=pred_x,
                                           y=pred_y,
                                           once_every=2,
                                           batch_size=len(pred_y),
                                           model_save_path='./pretrained_models/drmm_pretrained_model_fold'+fold+'/'
                                          )

train_generator = mz.DataGenerator(train_pack_processed, mode='pair', num_dup=2, num_neg=1, batch_size=20,
                                   callbacks=[hist_callback])
print('num batches:', len(train_generator))

history = model.fit_generator(train_generator, epochs=50, callbacks=[evaluate], workers=8, use_multiprocessing=True)
