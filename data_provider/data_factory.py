"""DataLoader factory."""

from __future__ import annotations

from torch.utils.data import DataLoader

from data_provider.data_loader import TimeSeriesDataset


def data_provider(args, flag: str):
    """Create dataset and PyTorch DataLoader for one split."""
    scaler = getattr(args, "scaler", None) if flag != "train" else None
    dataset = TimeSeriesDataset(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=(args.seq_len, args.label_len, args.pred_len),
        features=args.features,
        target=args.target,
        data=args.data,
        use_zscore=getattr(args, "use_zscore", True),
        timeenc=getattr(args, "timeenc", 0),
        freq=getattr(args, "freq", "h"),
        use_llm_features=getattr(args, "use_llm_features", False),
        llm_rule_path=getattr(args, "llm_rule_path", None),
        scaler=scaler,
    )

    if flag == "train":
        args.scaler = dataset.scaler
        args.raw_input_dim = dataset.feature_dim
        args.llm_feature_dim = dataset.llm_feature_dim
        args.target_dim = dataset.target_dim
        args.target_indices = dataset.target_indices
        args.mask_names = dataset.mask_names
        args.llm_feature_names = dataset.llm_feature_names
        args.enc_in = dataset.feature_dim + (dataset.llm_feature_dim if _flag(getattr(args, "use_llm_features", False)) else 0)
        args.c_out = dataset.target_dim

    batch_size = int(getattr(args, "batch_size", 32))
    shuffle = flag == "train"
    drop_last = flag == "train" and len(dataset) >= batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(getattr(args, "num_workers", 0)),
        drop_last=drop_last,
    )
    return dataset, loader


def _flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)
