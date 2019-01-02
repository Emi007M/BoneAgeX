from enum import Enum
from data.IData import Data
from data.data_line import DataLine
from data.data_sin import DataSin
from data.data_triple import DataTriple
from data.data_points import DataPoints
from data.data_bottleneck import DataBottleneck
from data.data_image import DataImage

# type Bottleneck compatible with graph type BAA
class DataType(Enum):
    Line = 1
    Sin = 2
    Triple = 3
    Points = 4
    Bottleneck = 5
    Jpeg = 6


class DataService:
    __data: Data

    def __init__(self, type:DataType):
        if type is DataType.Line:
            self.__data = DataLine()
        elif type is DataType.Sin:
            self.__data = DataSin()
        elif type is DataType.Triple:
            self.__data = DataTriple()
        elif type is DataType.Points:
            self.__data = DataPoints()
        elif type is DataType.Bottleneck:
            self.__data = DataBottleneck()
        elif type is DataType.Jpeg:
            self.__data = DataImage()
        else:
            self.__data = DataLine()

    def get_data_struct(self) -> Data:
        return self.__data
