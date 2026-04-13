import pydicom
from pydicom.dataset import Dataset


class CTSliceCache:
    def __init__(self, index: dict[str, str]):
        self.index = index
        self.cache: dict[str, Dataset] = {}

    def load(self, sop_uid: str) -> Dataset:
        if sop_uid in self.cache:
            return self.cache[sop_uid]

        path = self.index.get(sop_uid)
        if path is None:
            raise FileNotFoundError(f"No CT slice for UID {sop_uid}")

        ds = pydicom.dcmread(path)
        if getattr(ds, "Modality", None) != "CT":
            raise ValueError("Referenced slice is not CT")

        self.cache[sop_uid] = ds
        return ds
