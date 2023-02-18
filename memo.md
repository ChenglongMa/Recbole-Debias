# Changelog

[Trainer](./recbole_debias/trainer/trainer.py)
1. Added `decay` config

[Dataset](./recbole_debias/data/dataset.py)
1. `TICEN`: new `data_augmentation` function;
2. `TICEN`: no change;
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

# Implement `DICE` dataloaders

## Implement `MaskedNegSampleDataLoader`
* Parent classes: `NegSampleDataLoader`
  * methods:
    1. `__init__`: just added `get_logger`;
    2. `_set_neg_sample_args`: don't need to override
    3. `_neg_sampling`
    4. `_neg_sample_by_pair_wise_sampling`
    5. `_neg_sample_by_point_wise_sampling`
    6. `get_model`: don't need to override
*  
