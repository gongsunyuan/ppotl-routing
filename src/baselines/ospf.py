class OSPFPolicy:
    needs_pretrain = False

    def __init__(self, cfg=None):
        pass

    def select_action(self, state, greedy=True):
        return 0, 0.0

    def save(self, path, norms=None):
        pass

    def load(self, path):
        return None
