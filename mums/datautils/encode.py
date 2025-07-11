from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import json

class ConditionalLabelEncoder:
    def __init__(self, schema, exclude=[]):
        self.schema = schema
        self.exclude = exclude
        self.encoders = defaultdict(dict)  # e.g. encoders['shared']['instrument_name'] = LabelEncoder()

    def fit(self, records):
        for field in self.schema["shared"]:
            if field in self.exclude:
                continue
            values = [r[field] for r in records.values() if field in r]
            le = LabelEncoder()
            le.fit(values)
            self.encoders["shared"][field] = le

        for rtype, fields in self.schema["by_type"].items():
            subset = [r for r in records.values() if r.get("type_str") == rtype]
            for field in fields:
                if field in self.exclude:
                    continue
                values = [r[field] for r in subset if field in r]
                le = LabelEncoder()
                le.fit(values)
                self.encoders[rtype][field] = le

    def transform(self, records):
        for r in records.values():  # in-place transformation
            for field in self.schema["shared"]:
                if field in self.exclude:
                    continue
                if field in r:
                    r[f"{field.replace('_str', '_id')}"] = int(self.encoders["shared"][field].transform([r[field]])[0])
            rtype = r.get("type_str")
            if rtype in self.schema["by_type"]:
                for field in self.schema["by_type"][rtype]:
                    if field in self.exclude:
                        continue
                    if field in r:
                        r[f"{field.replace('_str', '_id')}"] = int(self.encoders[rtype][field].transform([r[field]])[0])

    def save(self, path):
        info = {
            "schema": self.schema,
            "encoders": {
                group: {
                    field: le.classes_.tolist()
                    for field, le in fields.items()
                }
                for group, fields in self.encoders.items()
            },
        }
        with open(path, "w") as f:
            json.dump(info, f, indent=4)

    def load(self, path):
        with open(path) as f:
            state = json.load(f)
        self.schema = state["schema"]
        self.encoders = defaultdict(dict)
        for group, fields in state["encoders"].items():
            for field, classes in fields.items():
                le = LabelEncoder()
                le.classes_ = classes
                self.encoders[group][field] = le