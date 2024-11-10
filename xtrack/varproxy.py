from collections import UserDict

# Behaves like defaultdict, but with a custom __setitem__ method
class VarProxy(UserDict):

    def __init__(self, *args, **kwargs):
        self.default_factory = kwargs.pop('default_factory', None)
        super().__init__(*args, **kwargs)
        self.mng = None

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.mng is not None:
            setattr(self.mng.MADX, key.replace('.', '_'), value)

    def __getitem__(self, key):
        if key not in self and self.default_factory is not None:
            self[key] = self.default_factory()
        return super().__getitem__(key)
