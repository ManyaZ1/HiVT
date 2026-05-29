import h5py
with h5py.File("teacher_outputs/train.h5", "r") as f:
    keys = list(f.keys())
    print(f"Entries: {len(keys)}")
    sample_key = next((key for key in keys if key != "_meta"), None)
    if sample_key is None:
        raise KeyError("No scene groups found in teacher_outputs/train.h5")
    print(f"Sample key: {sample_key}")
    print(f"Sample loc shape: {f[sample_key]['loc'].shape}")