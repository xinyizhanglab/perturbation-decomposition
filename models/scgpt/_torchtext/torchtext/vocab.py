class Vocab:
    def __init__(self, token_to_id=None, default_idx=0):
        if isinstance(token_to_id, Vocab):
            self._t2i = dict(token_to_id._t2i)
        elif isinstance(token_to_id, dict):
            self._t2i = dict(token_to_id)
        elif token_to_id is None or (isinstance(token_to_id, list) and len(token_to_id) == 0):
            self._t2i = {}
        else:
            self._t2i = dict(token_to_id) if token_to_id else {}
        self._i2t = {v: k for k, v in self._t2i.items()}
        self._default = default_idx

    @property
    def vocab(self):
        return self

    def __getitem__(self, token):
        return self._t2i.get(token, self._default)
    def __contains__(self, token):
        return token in self._t2i
    def __len__(self):
        return len(self._t2i)
    def __call__(self, tokens):
        if isinstance(tokens, str):
            return self[tokens]
        return [self[t] for t in tokens]
    def set_default_index(self, idx):
        self._default = idx
    def get_itos(self):
        return [self._i2t.get(i, "") for i in range(max(self._i2t.keys()) + 1 if self._i2t else 0)]
    def get_stoi(self):
        return dict(self._t2i)
    def insert_token(self, token, idx):
        self._t2i[token] = idx
        self._i2t[idx] = token
    def append_token(self, token):
        if token not in self._t2i:
            idx = max(self._i2t.keys()) + 1 if self._i2t else 0
            self._t2i[token] = idx
            self._i2t[idx] = token

def vocab(ordered_dict, **kwargs):
    return Vocab({token: i for i, token in enumerate(ordered_dict)})
