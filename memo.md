# Changelog

[Trainer](./recbole_debias/trainer/trainer.py)
1. Added `decay` config

[Dataset](./recbole_debias/data/dataset.py)
1. `H2NET`: new `data_augmentation` function;
2. `H2NET`: no change;
3. `DICE`: `DebiasDataset`, no change;

[Dataloader](./recbole_debias/data/dataloader.py)
1. `neg_sampling` function add `mask` parameter;

[Sampler](./recbole_debias/sampler/sampler.py)
1. Added `item_ids_repeat` parameter;

# TODO

For the new model, we implement 
1. a variant of the Dataset of `DIEN`;
2. the sampler of `DICE`
3. the default `Trainer`;
4. the default `Dataloader`;

[x] To implement **masked negative sampling** in dataset and **sampler**;

[x] To implement the new model based on `DIEN`;

[ ] Save seq benchmark datasets;

[ ] Implement intervention dataset generation method;

[ ] Implement diversity evaluation metrics;
