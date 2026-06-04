torch cat vs torch stack

original shape was (6, 30, 2)

stack makes it (4, 6, 30, 2) and cat  (24, 30, 2)

PyG's new collator treats dim-0 as the "node" dimension for any tensor attribute it doesn't know about. So:

## why tis happns
PyG is optimized for graphs where the number of nodes and edges varies from graph to graph. To avoid wasting memory with padding, PyG doesn't add a classic batch dimension like [Batch, Nodes, Features]. Instead, it combines graphs into one giant "disjoint union" graph.

[6, 30, 2] → dim-0 has size 6 → collator thinks there are 6 nodes → concatenates along dim-0 → [12, 30, 2] for batch=2. Wrong.

The proper fix: register __inc__ and __cat_dim__ on your Data object
PyG exposes two hooks on Data subclasses for exactly this:

__cat_dim__(key, value) — which dimension to concatenate along (return None to stack instead of cat)
__inc__(key, value) — how much to increment indices when batching (0 for non-index tensors)

Create a KDData subclass:
python# in kd_dataset.py
from torch_geometric.data import Data

class KDData(Data):
    """Data subclass that tells PyG how to batch teacher soft targets."""

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("teacher_loc", "teacher_scale", "teacher_pi"):
            # Stack along a NEW dim-0 → [B, F, H, 2] and [B, F]
            # Returning None means PyG will torch.stack instead of torch.cat
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("teacher_loc", "teacher_scale", "teacher_pi", "has_teacher"):
            return 0
        return super().__inc__(key, value, *args, **kwargs)
Then in KDDataset.__getitem__, copy the base Data fields into a KDData:
pythondef __getitem__(self, idx):
    base_data = self.base[idx]
    data = KDData(**{k: v for k, v in base_data})   # copy all existing attrs

    teacher = self._load_teacher(data.seq_id)
    if teacher is None:
        # ... warning logic unchanged ...
        data.has_teacher = torch.tensor(False)
        return data

    data.teacher_loc   = teacher["loc"]    # [6, 30, 2]  — no unsqueeze needed
    data.teacher_scale = teacher["scale"]  # [6, 30, 2]
    data.teacher_pi    = teacher["pi"]     # [6]
    data.has_teacher   = torch.tensor(True)
    return data