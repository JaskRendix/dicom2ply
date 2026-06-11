import logging
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset

logger = logging.getLogger(__name__)


class CTSliceCache:
    """
    Cache for CT slices indexed by SOPInstanceUID.
    Provides lazy loading, caching, and validation.
    """

    def __init__(self, index: dict[str, str]):
        self.index = {uid: str(path) for uid, path in index.items()}
        self.cache: dict[str, Dataset | None] = {}

    def __contains__(self, sop_uid: str) -> bool:
        return sop_uid in self.index

    def __len__(self) -> int:
        return len(self.index)

    def clear(self) -> None:
        self.cache.clear()

    def load(self, sop_uid: str) -> Dataset:
        # Fast path: cached
        if sop_uid in self.cache:
            ds = self.cache[sop_uid]
            if ds is None:
                raise FileNotFoundError(f"No CT slice for UID {sop_uid}")
            logger.debug(f"CTSliceCache hit: {sop_uid}")
            return ds

        # Missing slice
        path = self.index.get(sop_uid)
        if path is None:
            logger.error(f"No CT slice for UID {sop_uid}")
            self.cache[sop_uid] = None  # negative cache
            raise FileNotFoundError(f"No CT slice for UID {sop_uid}")

        # Read from disk
        try:
            ds = pydicom.dcmread(path)
        except Exception as e:
            logger.error(f"Failed to read CT slice {path}: {e}")
            raise

        # Validate modality
        if getattr(ds, "Modality", None) != "CT":
            logger.error(f"Referenced slice {path} is not CT (Modality={ds.Modality})")
            raise ValueError("Referenced slice is not CT")

        # Cache and return
        self.cache[sop_uid] = ds
        return ds
