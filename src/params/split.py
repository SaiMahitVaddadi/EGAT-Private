
class SplitParams:
    split_type: str
    train_size: float
    test_size: float
    n_splits: int
    fold_shuffle: bool
    random_state: int
    smiles: str
    target: str
    astartes: Optional[dict] = None

@dataclass
class AstartesParams:
    hopts: dict
    fingerprint: str
    fingerprint_args: dict