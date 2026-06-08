import logging

import pydicom
from pydicom.dataset import Dataset

logger = logging.getLogger(__name__)


class CTSliceCache:
    def __init__(self, index: dict[str, str]):
        self.index = index
        self.cache: dict[str, Dataset] = {}

    def load(self, sop_uid: str) -> Dataset:
        # Fast path: cached
        if sop_uid in self.cache:
            return self.cache[sop_uid]

        # Missing slice
        path = self.index.get(sop_uid)
        if path is None:
            logger.error(f"No CT slice for UID {sop_uid}")
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

        self.cache[sop_uid] = ds
        return ds
