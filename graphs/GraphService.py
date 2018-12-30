from Lib.enum import Enum
from graphs.IGraph import Graph
from graphs.graph_linear import GraphLinear
from graphs.graph_dense import GraphDense
from graphs.graph_baa import GraphBAA
from graphs.graph_baa_keras import GraphBAAkeras
from graphs.graph_baa_keras_jpeg import GraphBAAkerasJpeg

# type BAA compatible with data type Bottleneck
class GraphType(Enum):
    Linear = 1
    Dense = 2
    BAA = 3
    BAAkeras = 4
    BAAkerasJpeg = 5


class GraphService:
    __graph: Graph

    def __init__(self, type:GraphType):
        if type is GraphType.Linear:
            self.__graph = GraphLinear()
        elif type is GraphType.Dense:
            self.__graph = GraphDense()
        elif type is GraphType.BAA:
            self.__graph = GraphBAA()
        elif type is GraphType.BAAkeras:
            self.__graph = GraphBAAkeras()
        elif type is GraphType.BAAkerasJpeg:
            self.__graph = GraphBAAkerasJpeg()
        else:
            self.__graph = GraphDense()

    def get_graph_struct(self, input_tensor_size, output_tensor_size) -> Graph:
        self.__graph.set_tensor_lengths(input_tensor_size, output_tensor_size)
        return self.__graph
