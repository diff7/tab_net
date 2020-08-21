import torch
from torch.utils.data import Dataset


to_int = lambda x: torch.tensor(x, dtype=torch.int64)
to_float = lambda x: torch.tensor(x, dtype=torch.float32)
to_long = lambda x: torch.tensor(x, dtype=torch.long)


"""
Several functions to prepare dataset and its description.

"""

class LabelEnc:
    def __init__(self):
        self.mappers = dict()

    def fit(self, frame, cat_fe):
        for cat_fe in cat_fe:
            labels = frame[cat_fe].unique()
            mapper = {labels[i]:i for i in range(len(labels))}
            self.mappers[cat_fe] = mapper
            
    def transform(self, frame_orig):
        frame = frame_orig.copy()
        if len(self.mappers) == 0:
            print('Use fit first')
            return None
        
        for col in self.mappers:
            frame[col] = frame[col].map(self.mappers[col])
            frame[col].fillna(max(self.mappers[col].values())+1, inplace=True)
        return frame

    def fit_transform(self, frame, cat_fes):
        self.fit(frame, cat_fes)
        return self.transform(frame)

def get_low_variance_objects(frame, th=None):
    if th is None:
        th = frame.shape[0] * 0.01

    low_var = set(frame.T[frame.nunique() < th].index.to_list())
    objects = set(frame.select_dtypes(include="object").columns.to_list())
    print(f"Low variance items all: {len(low_var)}, Objects all: {len(objects)}")
    low_var_objects = low_var.intersection(objects)
    high_var_objects = objects.difference(low_var)
    print(
        f"Low variance objects: {len(low_var_objects)}, High var objects: {len(high_var_objects)}"
    )
    return low_var_objects, high_var_objects, low_var.difference(objects)


def get_feature_sizes(frame, params, th=None):
    low_var_objects, high_var_objects, low_var_real = get_low_variance_objects(
        frame, th=th
    )
    categoical = low_var_objects.union(low_var_real)
    low_var_n_objects = categoical.union(high_var_objects)
    real_features = set(frame.columns).difference(low_var_n_objects)
    real_features_size = len(real_features)
    cat_embed_sizes = dict()
    input_concat_vector_size = real_features_size

    for cat_name in categoical:
        n = frame[cat_name].nunique()
        cat_embed_sizes[cat_name] = {
            "embedding_dim": int(
                max(n**params['embed_exponent'] / params["embed_scaler"], params["min_embed_size"])
            ),
            "num_categories": n + 1,
        }
        input_concat_vector_size += cat_embed_sizes[cat_name]["embedding_dim"]
    params.update(
        {
            "real_features_size": real_features_size,
            "num_categoical": len(categoical),
            "cat_embed_sizes": cat_embed_sizes,
            "real_features": real_features,
            "input_concat_vector_size": input_concat_vector_size,
            "high_var_objects": high_var_objects,
        }
    )
    return params


class PandasCatLoader(Dataset):
    def __init__(self, data, params):
        self.data = data
        self.params = params

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        real_vector = to_float(self.data.loc[idx, self.params["real_features"]].values)
        cat_vector = dict()
        for name in self.params["cat_embed_sizes"]:
            if name in self.data.columns:
                cat_vector[name] = to_int(self.data.loc[idx, name])
        target = to_long(self.data.loc[idx, self.params["TARGET"]])

        return {"real_vector": real_vector, "cat_vector": cat_vector, "target": target}
