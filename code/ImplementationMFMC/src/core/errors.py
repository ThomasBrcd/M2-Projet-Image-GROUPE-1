class FaceAnonError(RuntimeError):
    """Erreur exploitable par l'interface et la CLI."""


class ModelError(FaceAnonError):
    pass


class SelectionError(FaceAnonError):
    pass


class BankError(FaceAnonError):
    pass
