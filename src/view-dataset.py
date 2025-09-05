import tensorflow_datasets as tfds

# carrega dados do dataset 'mnist' e 'decompõe' nas várias para fácil acesso
ds_train, ds_info = tfds.load('mnist', split='train', with_info=True)

tfds.visualization.show_examples(ds=ds_train, ds_info=ds_info)



