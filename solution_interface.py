import abc

class FormalParserInterface(metaclass=abc.ABCMeta):
    """
    Interface, który każde rozwiązanie ma implementować,
    ma mieć metody:
    -wczytanie danych z jsona
    -rozwiązanie problemu (zwrócenie instancji klasy Result)
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load_data_from_json') and
                callable(subclass.load_data_from_json) and
                hasattr(subclass, 'solve') and
                callable(subclass.solve))