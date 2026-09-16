"""Inference failures shared by reusable runtimes and domain pipelines."""


class ClassifierUnavailable(RuntimeError):
    pass


class ClassifierBusy(RuntimeError):
    pass


class ClassifierOutputError(RuntimeError):
    pass
