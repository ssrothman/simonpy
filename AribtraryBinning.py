import numpy as np
#import torch
import itertools
import json
import os
import hist
from typing import List, Union, Tuple

class _BinningBlock:
    '''
    Not part of the public interface

    _BinningBlocks are the basic building blocks of arbitrary binnings
    A _BinningBlock represents a contiguous block of rectangular binning
    Any arbitrary (nonrectangular) binning can be represented as a collection of _BinningBlocks

    _BinningBlocks are part of the private interface and should not be used directly by users

    Attributes:
        strides : List[int] - list of strides for each axis
        extents : List[int] - list of axis extents (including flow bins)
        axis_names : List[str] - list of axis names
        Nax : int - number of axes
        total_size : int - total number of bins in the block
        offset : int - offset of the block in the overall binning
        ax_details : dict - dictionary of axis details. Has the following format:
            {
                axis_name : {
                    'edges' : List[float] - list of bin edges (including flow bin edges)
                    'extent' : int - number of bins along this axis
                    'minedge' : float - minimum edge of the axis (including flow bin)
                    'maxedge' : float - maximum edge of the axis (including flow bin)
                }
            }

    '''
    def __init__(self):
        '''
        Initialize empty _BinningBlock
        '''
        self.strides : List[int] = []
        self.extents : List[int]= []
        self.axis_names : List[str] = []
        self.ax_details : dict= {}
        self.Nax : int = 0
        self.total_size :int = 1
        self.offset : int = 0

    def __eq__(self, other) -> bool:
        '''
        Equality operator
                
        :param self: This object
        :param other: Other object
        :return: True iff other is a _BinningBlock with the same properties
        :rtype: bool
        '''
        if not isinstance(other, _BinningBlock):
            return False

        if self.Nax != other.Nax:
            return False

        if self.total_size != other.total_size:
            return False

        if self.axis_names != other.axis_names:
            return False

        for i, ax_name in enumerate(self.axis_names):
            if ax_name not in other.axis_names:
                return False

            if self.extents[i] != other.extents[other.axis_names.index(ax_name)]:
                return False

            if self.ax_details[ax_name]['edges'] != other.ax_details[ax_name]['edges']:
                return False

        return True

    def copy(self) -> '_BinningBlock':
        '''
        Make a deep copy of this _BinningBlock
                
        :param self: This object
        :return: A deep copy of this _BinningBlock
        :rtype: _BinningBlock
        '''
        result = _BinningBlock()
        result.from_dict(self.to_dict())
        return result

    def to_dict(self) -> dict:
        '''
        Dump to python dictionary. Inverse of _BinningBlock.from_dict()
                
        :param self: This object
        :return: A dictionary representation of this _BinningBlock
        :rtype: dict[Any, Any]
        '''
        result = {
            'axis_names': self.axis_names,
            'Nax': self.Nax,
            'extents': self.extents,
            'ax_details': self.ax_details,
            'strides': self.strides,
            'total_size': self.total_size,
            'offset': self.offset
        }
        return result

    def from_dict(self, thedict : dict):
        '''
        Initialize from python dictionary. Inverse of _BinningBlock.to_dict()
                
        :param self: This object
        :param thedict: A dictionary representation of a _BinningBlock
        :type thedict: dict
        '''
        self.axis_names = thedict['axis_names']
        self.Nax = thedict['Nax']
        self.extents = thedict['extents']
        self.ax_details = thedict['ax_details']
        self.strides = thedict['strides']
        self.total_size = thedict['total_size']
        self.offset = thedict['offset']

    def from_hist(self, H : hist.Hist):
        '''
        Initialize from a hist Histogram object
        
        :param self: This object
        :param H: A hist Histogram object
        :type H: hist.Hist
        '''
        self.Nax = len(H.axes)

        for ax in H.axes:
            self.axis_names.append(ax.name)
            self.extents.append(ax.extent)
            edges = ax.edges.tolist()
            if ax.traits.underflow:
                edges = [-np.inf] + edges
            if ax.traits.overflow:
                edges = edges + [np.inf]
            self.ax_details[ax.name] = {
                'edges': edges,
                'extent': ax.extent,
                'minedge': edges[0],
                'maxedge': edges[-1]
            }
            self.total_size *= ax.extent

        self.offset = 0

        self.calculate_strides()

    def calculate_strides(self):
        '''
        Calculate the strides for each axis in the binning block.
        
        :param self: This object
        '''
        self.strides = [0] * self.Nax
        self.strides[self.Nax - 1] = 1
        for i in range(self.Nax - 1, 0, -1):
            self.strides[i-1] = self.strides[i] * self.extents[i]

    def block_edges(self, lower: bool =True) -> dict:
        '''
        Get the bin edges for all axes in this _BinningBlock
                
        :param self: This object
        :param lower: If True, get lower edges; if False, get upper edges
        :type lower: bool
        :return: Dictionary of bin edges for all axes, in the format
            { axis_name : np.ndarray in the same shape as the data }
        :rtype: dict[Any, Any]
        '''
        fullshape = [self.ax_details[name]['extent'] for name in self.axis_names]

        result = {}
        for i in range(self.Nax):
            name = self.axis_names[i]
            edges = np.asarray(self.ax_details[name]['edges'])
            if lower:
                edges = edges[:-1]
            else:
                edges = edges[1:]

            #need to expand to match data shape
            theshape = [1] * self.Nax
            theshape[i] = self.ax_details[name]['extent']
            edges = edges.reshape(theshape)
            edges = np.broadcast_to(edges, fullshape)
            result[name] = edges
        
        return result

    def rebin(self, rebinning_spec : Union[dict, str]) -> List['_BinningBlock']:
        '''
        Rebin this _BinningBlock according to the provided rebinning_spec.
        Result is a list of new _BinningBlocks representing the rebinned structure.
                
        :param self: This object
        :param rebinning_spec: Either a dictionary representing the rebinning spec, or a path to a JSON file containing the spec
        :type rebinning_spec: Union[dict, str]
        :return: A list of new _BinningBlocks representing the rebinned structure
        :rtype: List[_BinningBlock]
        '''
        if type(rebinning_spec) is str:
            with open(rebinning_spec, 'r') as f:
                rebinning_spec = json.load(f)

        if type(rebinning_spec) is not dict:
            raise ValueError("Rebinning specification must be a dictionary or a path to a JSON file.")

        newblocks = []
        running_offset = 0
        for specblock in rebinning_spec['spec']:
            nextblock = _BinningBlock()
            nextblock.axis_names = self.axis_names.copy()
            nextblock.Nax = self.Nax
            for name in self.axis_names:
                extent = len(specblock[name]) - 1
                nextblock.extents.append(extent)
                edges = [self.ax_details[name]['edges'][i] for i in specblock[name]]
                nextblock.ax_details[name] = {
                    'edges' : edges,
                    'extent' : extent,
                    'minedge' : edges[0],
                    'maxedge' : edges[-1]
                }
                nextblock.total_size *= extent

            nextblock.calculate_strides()
            nextblock.offset = running_offset
            running_offset += nextblock.total_size

            newblocks.append(nextblock)

        return newblocks

    def edge_to_index(self, name : str, edge : Union[None, int, float]) -> Union[None, int]:
        '''
        Convert a bin edge to its corresponding index along the specified axis.
        NB None's are passed through unchanged to facilitate parsing of slices.
        
        :param self: This object
        :param name: The name of the axis
        :type name: str
        :param edge: The bin edge value
        :type edge: Union[None, int, float]
        :return: The index corresponding to the bin edge
        :rtype: int | None
        '''
        if edge is None: #special case, needed for slices
            return None

        edges = self.ax_details[name]['edges']
        for i, e in enumerate(edges):
            if np.abs(e-edge) < 1e-5 or e == edge: #== for inf values
                return i

        print()
        print(edge)
        print(self.ax_details[name]['edges'])
        print()
        raise ValueError(f"Edge {edge} not found in axis {name} edges.")

    def edges_to_indices(self, name : str, edges : Union[None, int, float, List, Tuple, slice, dict]) -> Union[None, int, List[Union[None,int]], slice, dict]:
        '''
        Vectorized version of edge_to_index(). Accepts many input types:
            None, int, float are passed as-is to edge_to_index()
            List or Tuple are passed to edge_to_index() element-wise and a List is returned
            slice start and stop are passed to edge_to_index() and a slice is returned, with unchanged step
            dict values are passed to edge_to_index() element-wise and a dict is returned
                NB for dicts the name parameter is ignored in favor of the dict keys
        
        :param self: This object
        :param name: The name of the axis
        :type name: str
        :param edges: The bin edge values
        :type edges: Union[None, int, float, List, Tuple, slice, dict]
        :return: The corresponding indices
        :rtype: int | List[int | None] | slice[Any, Any, Any] | dict[Any, Any] | None
        '''
        if type(edges) is int or type(edges) is float or edges is None:
            return self.edge_to_index(name, edges)
        elif type(edges) is list or type(edges) is tuple:
            return [self.edge_to_index(name, edge) for edge in edges]
        elif type(edges) is slice:
            start = self.edge_to_index(name, edges.start)
            stop = self.edge_to_index(name, edges.stop)
            return slice(start, stop, edges.step)
        elif type(edges) is dict:
            return {dictname: self.edges_to_indices(dictname, edges[dictname]) for dictname in edges}
        else:
            raise ValueError(f"Invalid type for edges: {type(edges)}. Expected int, float, list, slice, or dict.")

    def index_to_edge(self, name : str, index : Union[None, int]) -> Union[None, int, float]:
        '''
        Lookup bin edge corresponding to a given index along the specified axis.
        
        :param self: This object
        :param name: The name of the axis
        :type name: str
        :param index: The index along the axis
        :type index: Union[None, int]
        :return: The bin edge corresponding to the index
        :rtype: int | float | None
        '''
        if index is None: #special case, needed for slices
            return None

        if index < 0 or index >= self.ax_details[name]['extent']:
            raise IndexError(f"Index {index} out of bounds for axis {name}.")

        return self.ax_details[name]['edges'][index]

    def indices_to_edges(self, name : str, indices : Union[int, List[int], Tuple[int], dict, slice]) -> Union[None, int, float, List[Union[int, float, None]], dict, slice]:
        '''
        Vectorized version of index_to_edge(). Accepts many input types:
            int is passed as-is to index_to_edge()
            List or Tuple are passed to index_to_edge() element-wise and a List is returned
            slice start and stop are passed to index_to_edge() and a slice is returned, with unchanged step
            dict values are passed to index_to_edge() element-wise and a dict is returned
        
        :param self: This object
        :param name: The name of the axis
        :type name: str
        :param indices: The indices along the axis
        :type indices: Union[int, List[int], Tuple[int], dict, slice]
        :return: The bin edges corresponding to the indices
        :rtype: int | float | List[int | float | None] | dict[Any, Any] | slice[Any, Any, Any] | None
        '''
        if type(indices) is int:
            return self.index_to_edge(name, indices)
        elif type(indices) is list or type(indices) is tuple:
            return [self.index_to_edge(name, idx) for idx in indices]
        elif type(indices) is dict:
            return {name: self.indices_to_edges(name, indices[name]) for name in indices}
        elif type(indices) is slice:
            return slice(self.index_to_edge(name, indices.start),
                         self.index_to_edge(name, indices.stop),
                         indices.step)
        else:
            raise ValueError(f"Invalid type for indices: {type(indices)}. Expected int, list, tuple, dict, or slice.")

    def flatten_index(self, theindices : dict[str, int]) -> int:
        '''
        Get flat index corresponding to a fully-specified set of axis indices such that data[flat_index] gives the value at those indices.
        Inverse of unflatten_index().
        
        :param self: This object
        :param theindices: A dictionary of indices in all axes represented in this block, in the format
            { axis_name : index }
            The keys in this dictionary must be exactly the list of axis names in this block.
        :type theindices: dict[str, int]
        :return: The flat index corresponding to the specified axis indices such that data[flat_index] gives the value at those indices
        :rtype: int
        '''
        for name in theindices:
            if name not in self.axis_names:
                raise ValueError(f"Invalid axis name: {name}")

        for name in self.axis_names:
            if name not in theindices:
                raise ValueError(f"Missing value for axis: {name}")

        indices = []
        for name in self.axis_names:
            indices.append(theindices[name])

        idx = 0
        for i, index in enumerate(indices):
            if index < 0 or index >= self.extents[i]:
                raise IndexError(f"Index {index} out of bounds for axis {self.axis_names[i]}")
            idx += index * self.strides[i]

        return idx

    def unflatten_index(self, index : int) -> dict:
        '''
        Unpack a flat index into a dictionary of axis indices. Inverse of flatten_index().
        
        :param self: This object
        :param index: The flat index to unpack
        :type index: int
        :return: A dictionary of axis indices corresponding to the flat index. The format is
            { axis_name : index }
        :rtype: dict[Any, Any]
        '''
        result = {}
        for i, name in enumerate(self.axis_names):
            result[name] = (index // self.strides[i]) % self.extents[i]
        return result

    def get_slice_indices(self, sliceindices : dict) -> np.ndarray: 
        '''
        Get all the flat indices corresponding to a (not necessarily contiguous in memory) slice speficied by the input. 
        
        :param self: This object
        :param sliceindices: A dictionary specifying the slice along each axis in the format
            { axis_name : (start_index, stop_index) }
            This dictionary need not specify all axes in the block; any axes not specified will be allowed to run over their full range
        :type sliceindices: dict
        :return: A numpy array of flat indices corresponding to the specified slice
        :rtype: ndarray[Any, Any]
        '''

        #check input format
        for name in sliceindices:
            if name not in self.axis_names:
                raise ValueError(f"Invalid axis name: {name}")
            if type(sliceindices[name]) not in [list, tuple]:
                raise ValueError(f"Slice indices for axis {name} must be a list or tuple.")
            if len(sliceindices[name]) != 2:
                raise ValueError(f"Slice indices for axis {name} must contain exactly two elements: (start, stop).")

        indices = []
        for i in range(self.total_size): #for each possible flat index
            index = self.unflatten_index(i)
            accepted = True
            for axname in sliceindices:
                if index[axname] < sliceindices[axname][0] or index[axname] >= sliceindices[axname][1]: # index not in slice
                    accepted = False
                    break
            if accepted:
                indices.append(i)

        return np.asarray(indices, dtype=np.int64)

    def get_slice_from_indices(self, data : np.ndarray, sliceindices : dict) -> np.ndarray:
        '''
        Use get_slice_indices() to extract the specified slice from the data. 
        
        :param self: This object
        :param data: A numpy.ndarray containing the data to slice
        :type data: np.ndarray
        :param sliceindices: A dictionary specifying the slice along each axis in the format
            { axis_name : (start_index, stop_index) }
            This dictionary need not specify all axes in the block; any axes not specified will be allowed to run over their full range
        :type sliceindices: dict
        :return: An np.ndarray containing the extracted slice
        :rtype: ndarray[Any, Any]
        '''
        indices = self.get_slice_indices(sliceindices)
        #if type(data) is torch.Tensor:
        #    return torch.take(data, torch.tensor(self.offset + indices, device=data.device))
        #else:
        return np.take(data, self.offset+indices, axis=0)


    def get_slice_from_edges(self, data : np.ndarray, edges : dict) -> np.ndarray:
        '''
        Wrap get_slice_from_indices() to extract a slice specified by bin edges rather than indices. 
        
        :param self: This object
        :param data: A numpy.ndarray containing the data to slice
        :type data: np.ndarray
        :param edges: A dictionary specifying the slice along each axis in the format
            { axis_name : (start_edge, stop_edge) }
            This dictionary need not specify all axes in the block; any axes not specified will be allowed to run over their full range
        :type edges: dict
        :return: An np.ndarray containing the extracted slice
        :rtype: ndarray[Any, Any]
        '''
        indicesdict : dict = {name: self.edges_to_indices(name, edges[name]) for name in edges}
        return self.get_slice_from_indices(data, indicesdict)


    def assign_to_indices(self, data : np.ndarray, values : np.ndarray, indices : dict):
        '''
        Assign values to the specified indices in the data array
        
        :param self: This object
        :param data: A numpy.ndarray containing the data to modify
        :type data: np.ndarray
        :param values: A numpy.ndarray containing the values to assign
        :type values: np.ndarray
        :param indices: A dictionary specifying the indices along each axis in the format
            { axis_name : (start_index, stop_index) }
            This dictionary need not specify all axes in the block; any axes not specified will be allowed to run over their full range
        :type indices: dict
        '''
        indices_ = self.get_slice_indices(indices)
        data[indices_ + self.offset] = values
        
    def assign_to_indices_2d(self, data : np.ndarray, values : np.ndarray, sliceindices : dict):
        '''
        Utility for assigning to indices in 2D (usually covariance) matrices.
        Has the major limitation that it only works for contiguous-in-memory blocks of indices
        
        :param self: This object
        :param data: The 2D numpy.ndarray containing the data to modify
        :type data: np.ndarray
        :param values: The 2D numpy.ndarray containing the values to assign
        :type values: np.ndarray
        :param indices: A dictionary specifying the indices along each axis in the format
            { axis_name : (start_index, stop_index) }
            This dictionary need not specify all axes in the block; any axes not specified will be allowed to run over their full range
        :type indices: dict
        '''
        indices = self.get_slice_indices(sliceindices)

        #check if indices are contiguous
        indexmin = np.min(indices)
        indexmax = np.max(indices)
        indexrange = indexmax - indexmin +1
        if indexrange != values.shape[0]:
            raise ValueError("Can only assign to a continuous block of indices")
        
        #index with a slice object
        indexing = slice(self.offset + indexmin, 
                         self.offset + indexmax + 1)
        data[indexing, indexing] = values

    def project_out(self, data : np.ndarray, axis_name : str) -> Tuple[np.ndarray, '_BinningBlock']:
        '''
        Integrate over the specified axis, returning the projected data and a new _BinningBlock representing the new binning structure.
        
        :param self: This object
        :param data: The data array to project
        :type data: np.ndarray
        :param axis_name: The name of the axis to project out
        :type axis_name: str
        :return: A tuple containing the projected data and the new _BinningBlock
        :rtype: Tuple[ndarray[Any, Any], _BinningBlock]
        '''
        #the part of the data that is contained in this BinningBlock
        blockdata = data[self.offset: self.offset + self.total_size]
        shape = []

        whichax = 0
        for i, name in enumerate(self.axis_names):
            shape.append(self.ax_details[name]['extent'])
            if name == axis_name:
                whichax = i

        extradims = list(data.shape[1:])
        shape = shape + extradims

        blockdata = blockdata.reshape(shape)
        result = np.sum(blockdata, axis=whichax, keepdims=False)
        result = result.reshape((-1, *extradims))

        newblock = _BinningBlock()
        newblock.axis_names = self.axis_names.copy()
        newblock.axis_names.pop(whichax)
        newblock.Nax = self.Nax - 1
        newblock.extents = self.extents.copy()
        newblock.extents.pop(whichax)
        newblock.ax_details = self.ax_details.copy()
        newblock.ax_details.pop(axis_name)
        newblock.total_size = int(np.prod(newblock.extents))
        newblock.calculate_strides()
        newblock.offset = 0

        return result, newblock

    def value_at(self, data : np.ndarray, indices : dict):
        '''
        Lookup value in data at the specified indices
        
        :param self: This object
        :param data: The data array to lookup
        :type data: np.ndarray
        :param indices: The indices at which to lookup the value
        :type indices: dict
        :return: The value at the specified indices
        :rtype: Any
        '''
        return data[self.offset+self.flatten_index(indices)]

    def edges_in_block(self, sliceedges: dict) -> bool:
        '''
        Check whether the specified edges are fully-contained within this block.
        
        :param self: This object
        :param sliceedges: The edges to check. The format is
            { axis_name : (min_edge, max_edge) }
        :type sliceedges: dict
        :return: Whether the edges are fully-contained within this block
        :rtype: bool
        '''
        for name in sliceedges:
            if name not in self.axis_names:
                raise ValueError(f"Invalid axis name: {name}")

        for name in sliceedges:
            edges = sliceedges[name]
            allowedmin = self.ax_details[name]['minedge']
            allowedmax = self.ax_details[name]['maxedge']

            #check format
            if type(edges) not in [list,tuple]:
                raise ValueError(f"Edges for axis {name} must be a list or tuple.")
            elif len(edges) != 2:
                raise ValueError(f"Edges for axis {name} must contain exactly two elements: (min, max).")

            if edges[0] < allowedmin or edges[1] > allowedmax:
                return False

        return True

    def clip_edges_to_block(self, sliceedges : dict) -> Union[None, dict]:
        '''
        Clip the specified edges to be within this block
        
        :param self: This object
        :param sliceedges: The edges to clip. The format is
            { axis_name : (min_edge, max_edge) }
        :type sliceedges: dict
        :return: The clipped edges, or None if the edges do not overlap this block
        :rtype: dict[Any, Any] | None
        '''
        for name in sliceedges:
            if name not in self.axis_names:
                raise ValueError(f"Invalid axis name: {name}")

        clippededges = {}
        for name in sliceedges:
            edges = sliceedges[name]
            allowedmin = self.ax_details[name]['minedge']
            allowedmax = self.ax_details[name]['maxedge']

            #check format
            if type(edges) not in [list, tuple]:
                raise ValueError(f"Edges for axis {name} must be a list or tuple.")
            elif len(edges) != 2:
                raise ValueError(f"Edges for axis {name} must contain exactly two elements: (min, max).")

            clippedmin = max(edges[0], allowedmin)
            clippedmax = min(edges[1], allowedmax)

            if clippedmax <= clippedmin:
                return None

            clippededges[name] = (clippedmin, clippedmax)

        return clippededges

class ArbitraryBinning:
    '''
    This class represents an arbitrary multi-dimensional binning structure.
    '''
    def __init__(self):
        '''
        Initialize empty Binning
        
        :param self: This object
        '''
        self._blocks : List[_BinningBlock]= []
        self._axis_names : List[str] = []
        self._Nax : int = 0

    @property
    def single_block(self) -> bool:
        '''
        Check if this ArbitraryBinning consists of a single _BinningBlock
        
        :param self: This object
        :return: True iff this ArbitraryBinning consists of a single _BinningBlock
        :rtype: bool
        '''
        return len(self._blocks) == 1

    @property
    def Nax(self) -> int:
        '''
        Get the number of axes in this ArbitraryBinning
        
        :param self: This object
        :return: The number of axes
        :rtype: int
        '''
        return self._Nax

    @property
    def edges(self) -> dict:
        '''
        Get the bin edges for all axes in this ArbitraryBinning
        
        :param self: This object
        :return: A dictionary of bin edges for all axes in this ArbitraryBinning. The format is
            { axis_name : List[float] }
        :rtype: dict[Any, Any]
        '''
        result = {}
        if len(self._blocks) == 0:
            return result
        elif len(self._blocks) > 1:
            raise ValueError("Cannot get edges for ArbitraryBinning with multiple blocks.")


        block = self._blocks[0]
        for name in block.axis_names:
            result[name] = np.asarray(block.ax_details[name]['edges'])
            
        return result

    @property
    def axis_names(self) -> List[str]:
        '''
        Get the list of axis names in this ArbitraryBinning
        
        :param self: This object
        :return: The list of axis names
        :rtype: List[str]
        '''
        return self._axis_names

    @property
    def total_size(self) -> int:
        '''
        Get the total number of bins in this ArbitraryBinning
        
        :param self: This object
        :return: The total number of bins
        :rtype: int
        '''
        size = 0
        for block in self._blocks:
            size += block.total_size
        return size

    def copy(self):
        '''
        Make a deep copy of this ArbitraryBinning
        
        :param self: This object
        '''
        result = ArbitraryBinning()
        result.from_dict(self.to_dict())
        return result

    def setup_from_histogram(self, H : hist.Hist):
        '''
        Initialize from a hist.Hist object
        
        :param self: This object
        :param H: The histogram to initialize from
        :type H: hist.Hist
        '''
        self._Nax = len(H.axes)
        block = _BinningBlock()
        block.from_hist(H)
        self._blocks = [block]
        self._axis_names = block.axis_names

    def dump_to_file(self, file : str):
        '''
        Dump to json file
        
        :param self: This object
        :param file: The file path to write to
        :type file: str
        '''
        print("Writing binning spec to file: %s" % file)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        resultdict = self.to_dict()
        with open(file, 'w') as f:
            json.dump(resultdict, f, indent=4)

    def load_from_file(self, file : str):
        '''
        Load from json file
        
        :param self: This object
        :param file: The file path to read from
        :type file: str
        '''
        print("Reading binning spec from file: %s" % file)

        with open(file, 'r') as f:
            resultdict = json.load(f)

        self.from_dict(resultdict)

    def to_dict(self) -> dict:
        '''
        Dump to python dictionary
        
        :param self: This object
        :return: The dictionary representation of this object
        :rtype: dict[Any, Any]
        '''
        resultdict = {}
        resultdict['axis_names'] = self._axis_names
        resultdict['Nax'] = self._Nax
        resultdict['blocks'] = []
        for block in self._blocks:
            resultdict['blocks'].append(block.to_dict())
        return resultdict

    def from_dict(self, resultdict : dict):
        '''
        Initialize from python dictionary
        
        :param self: This object
        :param resultdict: The dictionary representation of this object
        :type resultdict: dict
        '''
        self._axis_names = resultdict['axis_names']
        self._Nax = resultdict['Nax']
        self._blocks = []
        for blockdata in resultdict['blocks']:
            block = _BinningBlock()
            block.from_dict(blockdata)
            self._blocks.append(block)
        
    def _get_edges(self, lower: bool =True) -> dict:
        '''
        Get the bin edges for all axes in this ArbitraryBinning
        
        :param self: This object
        :param lower: If True, get lower edges; if False, get upper edges
        :type lower: bool
        :return: Dictionary of bin edges for all axes, in the format
            { axis_name : np.ndarray in the same shape as the data }
        :rtype: dict[Any, Any]
        '''
        result = {}

        for block in self._blocks:
            block_edges = block.block_edges(lower=lower)
            for name in block_edges:
                if name not in result:
                    result[name] = block_edges[name]
                else:
                    result[name] = np.concatenate((result[name], block_edges[name]), axis=0)

        return result

    def lower_edges(self) -> dict:
        '''
        Get the lower bin edges for all axes in this ArbitraryBinning
        
        :param self: This object
        :return: Dictionary of lower bin edges for all axes, in the format
            { axis_name : np.ndarray in the same shape as the data }
        :rtype: dict[Any, Any]
        '''
        return self._get_edges(lower=True)
    
    def upper_edges(self) -> dict:
        '''
        Get the upper bin edges for all axes in this ArbitraryBinning
        
        :param self: This object
        :return: Dictionary of upper bin edges for all axes, in the format
            { axis_name : np.ndarray in the same shape as the data }
        :rtype: dict[Any, Any]
        '''
        return self._get_edges(lower=False)

    def value_at(self, data : np.ndarray, theedges : dict):
        '''
        Lookup value at the specified bin, identified by the lower bin edges.
        
        :param self: This object
        :param data: The data array to lookup
        :type data: np.ndarray
        :param theedges: The edges specifying the location. The format is
            { axis_name : lower_bin_edge }
            This dictionary must fully-specify an edge for every axis in the binning
        :type theedges: dict

        :return: The value at the specified bin
        :rtype: Any
        '''
        in_block = np.zeros(len(self._blocks), dtype=bool)
        for i, block in enumerate(self._blocks):
            if block.edges_in_block(theedges):
                in_block[i] = True

        if np.sum(in_block) == 0:
            raise ValueError("No block contains the specified bin!")
        elif np.sum(in_block) > 1:
            print("Multiple blocks contain the specified bin! This is a sign of a bug.\nSome troubleshooting information:")
            print("\tEdges:")
            for name in theedges:
                print("\t\t",name, ':', theedges[name])
            print("\tBlocks:", np.where(in_block)[0])
            raise ValueError("Multiple blocks contain the specified bin.")
        else:
            whichblock = np.argmax(in_block)

            #pass empty name parameter for dict input
            #I guarentee that dict inputs give dict outputs, byt pyright doesn't see that, so ignore the type error
            theindices : dict = self._blocks[whichblock].edges_to_indices('', theedges) # pyright: ignore[reportAssignmentType]
            return self._blocks[whichblock].value_at(data, theindices)
        
    def get_slice(self, data : np.ndarray, theedges : dict) -> np.ndarray:
        '''
        Get a slice of the data specified by the provided edges.
        
        :param self: This object
        :param data: The data to slice
        :type data: np.ndarray
        :param theedges: The edges specifying the slice. The format is
            { axis_name : (min_edge, max_edge) }
            This dictionary need not specify all axes in the binning
        :type theedges: dict
        :return: The sliced data
        :rtype: ndarray[Any, Any]
        '''

        #check which blocks overlap the requested edges
        #and get the clipped edges for each overlapping block
        overlap_block = np.ones(len(self._blocks), dtype=bool)
        clipped_edges = []
        for i, block in enumerate(self._blocks):
            clipped = block.clip_edges_to_block(theedges)
            if clipped is None:
                overlap_block[i] = False
                clipped_edges.append(None)
            else:
                clipped_edges.append(clipped)

        if np.sum(overlap_block) == 0:
            print("No blocks overlap with the specified edges.")

        #if type(data) is torch.Tensor:
        #    result = torch.empty((0, *data.shape[1:]), dtype=data.dtype, device=data.device)
        #else:
        result = np.empty((0, *data.shape[1:]), dtype=data.dtype)

        for i, block in enumerate(self._blocks):
            if overlap_block[i]:
                nextslice = block.get_slice_from_edges(data, clipped_edges[i])
                #if type(data) is torch.Tensor:
                #    result = torch.cat((result, theslice), dim=0)
                #else:
                result = np.append(result, nextslice, axis=0)
        return result

    def get_slice_cov2d(self, data : np.ndarray, theedges : dict) -> np.ndarray:
        '''
        Wrapper for get_slice() to handle 2D (usually covariance) matrices. 
        Just calls get_slice() twice, transposing the data inbetween to index the other axis.
    
        :param self: This object
        :param data: The data to slice
        :type data: np.ndarray
        :param theedges: The edges specifying the slice. The format is
            { axis_name : (min_edge, max_edge) }
            This dictionary need not specify all axes in the binning
        :type theedges: dict
        :return: The sliced data
        :rtype: ndarray[Any, Any]
        '''
        result = self.get_slice(data.T, theedges)
        return self.get_slice(result.T, theedges)

    def get_sliced_binning(self, theedges : dict) -> 'ArbitraryBinning':
        '''
        Get a new ArbitraryBinning representing the binning structure of the slice specified by the provided edges.
        
        :param self: This object
        :param theedges: The edges specifying the slice. The format is
            { axis_name : (min_edge, max_edge) }
            This dictionary need not specify all axes in the binning
        :type theedges: dict
        :return: A new ArbitraryBinning representing the binning structure of the specified slice
        :rtype: ArbitraryBinning
        '''

        #check which blocks overlap the requested edges
        #and get the clipped edges for each overlapping block
        overlap_block = np.ones(len(self._blocks), dtype=bool)
        clipped_edges = []
        for i, block in enumerate(self._blocks):
            clipped = block.clip_edges_to_block(theedges)
            if clipped is None:
                overlap_block[i] = False
                clipped_edges.append(None)
            else:
                clipped_edges.append(clipped)

        if np.sum(overlap_block) == 0:
            raise ValueError("No blocks overlap with the specified edges.")

        #build new binning
        newbinning = ArbitraryBinning()
        newbinning._axis_names = self._axis_names.copy()
        newbinning._Nax = self._Nax
        newbinning._blocks = []

        running_offset = 0
        for i, block in enumerate(self._blocks):
            if overlap_block[i]:
                newblock = _BinningBlock()
                newblock.axis_names = block.axis_names.copy()
                newblock.Nax = block.Nax
                for name in block.axis_names:
                    extent = 0
                    edges = []
                    if name in clipped_edges[i]:
                        edgepair = clipped_edges[i][name]
                        edges = [e for e in block.ax_details[name]['edges'] if e >= edgepair[0] and e <= edgepair[1]]
                        extent = len(edges) - 1
                    else:
                        edges = block.ax_details[name]['edges'].copy()
                        extent = block.ax_details[name]['extent']

                    newblock.extents.append(extent)
                    newblock.ax_details[name] = {
                        'edges' : edges,
                        'extent' : extent,
                        'minedge' : edges[0],
                        'maxedge' : edges[-1]
                    }
                    newblock.total_size *= extent
                newblock.calculate_strides()
                newblock.offset = running_offset
                running_offset += newblock.total_size
                newbinning._blocks.append(newblock)


        unique_edges = {}
        for name in newbinning._axis_names:
            unique_edges[name] = set()
        for block in newbinning._blocks:
            for name in block.axis_names:
                for edge in block.ax_details[name]['edges']:
                    unique_edges[name].add(edge)
        for name in unique_edges:
            if len(unique_edges[name]) <= 2: #then this axis is degenerate and can be REMOVED
                newbinning._axis_names.remove(name)
                newbinning._Nax -= 1
                for block in newbinning._blocks:
                    offending_index = block.axis_names.index(name)
                    block.extents.pop(offending_index)
                    block.strides.pop(offending_index)
                    block.axis_names.pop(offending_index)
                    block.Nax -= 1
                    del block.ax_details[name]

        return newbinning
    
    def project_out(self, data : np.ndarray, axis_name: str) -> Tuple[np.ndarray, 'ArbitraryBinning']:
        '''
        Project out an axis from the data, returning the projected data and a new ArbitraryBinning representing the new binning structure.
        
        Some smart unit tests probably need to be devised for this. I've tried to make it robust, but I can imagine that there are edge cases that break this, especially if you order your axes in a way I didn't expect.
        As a simple check, the method checks that the total sum of the data is conserved, and errors otherwise. With arbitrary data, it's hard to imagine failures that don't change the total sum, so at least this can catch most errors.

        :param self: This object
        :param data: The data array to project
        :type data: np.ndarray
        :param axis_name: The name of the axis to project out
        :type axis_name: str
        :return: A tuple containing the projected data and a new ArbitraryBinning instance representing the new binning structure
        :rtype: Tuple[ndarray[Any, Any], ArbitraryBinning]
        '''
        #for consistency check conservation of total sum
        startsum = data.sum(axis=None)

        #result array
        result = np.empty((0, *data.shape[1:]), dtype=data.dtype)

        #new binning object
        newbinning = ArbitraryBinning()
        newbinning._axis_names = self._axis_names.copy()
        newbinning._axis_names.remove(axis_name)
        newbinning._Nax = self._Nax - 1

        #get new data and new blocks
        newdata = []
        newblocks = []
        for block in self._blocks:
            blockdata, newblock = block.project_out(data, axis_name)
            newdata.append(blockdata)
            newblocks.append(newblock)

        #build the new binning and new data objects, merging blocks as needed
        running_offset = 0

        used = np.zeros(len(newblocks), dtype=bool)
        for i in range(len(newblocks)):
            if used[i]:
                continue

            used[i] = True

            result = np.append(result, newdata[i], axis=0)
            newblocks[i].offset = running_offset
            running_offset += newblocks[i].total_size
            newbinning._blocks.append(newblocks[i])

            #check for equivalent blocks and merge them into this one
            for j in range(i+1, len(newblocks)):
                if used[j]:
                    continue

                if newblocks[j] == newblocks[i]:
                    result[-newblocks[i].total_size:] += newdata[j]
                    used[j] = True

        endsum = result.sum(axis=None)
        if not np.isclose(startsum, endsum):
            raise ValueError(f"Projection changed the sum of the data: {startsum} -> {endsum}")

        return result, newbinning

    def project_out_cov2d(self, data : np.ndarray, axis_name: str) -> Tuple[np.ndarray, 'ArbitraryBinning']:
        '''
        Wrapper for project_out() to handle 2D (usually covariance) matrices. 
        Just calls project_out() twice, transposing the data inbetween to index the other axis

        :param self: This object
        :param data: The data array to project
        :type data: np.ndarray
        :param axis_name: The name of the axis to project out
        :type axis_name: str
        :return: A tuple containing the projected data and a new ArbitraryBinning instance representing the new binning structure
        :rtype: Tuple[ndarray[Any, Any], ArbitraryBinning]
        '''
        result, newbinning = self.project_out(data.T, axis_name)
        result, _ = self.project_out(result.T, axis_name)
        return result, newbinning

    def rebin(self, data : np.ndarray, rebinning_spec : Union[str, dict]) -> Tuple[np.ndarray, 'ArbitraryBinning']:
        '''
        Rebin data according to the supplied specification, which can either be a dict or a path to a json file.
        NB only rectangular binnings (represented by a single _BinningBlock) can be rebinned.

        I don't check that the rebinning spec is valid (e.g. that the specified indices cover all original bins exactly once),
        but I do check that the total sum of the data is conserved, which should catch most mistakes.
        
        :param self: This object
        :param data: The data to rebin
        :type data: np.ndarray
        :param rebinning_spec: The rebinning specification, either as a dictionary or a path to a JSON file
        :type rebinning_spec: Union[str, dict]
        :return: A tuple containing the rebinned data and a new ArbitraryBinning instance representing the new binning structure
        :rtype: Tuple[ndarray[Any, Any], ArbitraryBinning]
        '''
        if len(self._blocks) != 1:
            raise ValueError("Can only rebin single-block binnings")

        if type(rebinning_spec) is str:
            with open(rebinning_spec, 'r') as f:
                rebinning_spec = json.load(f)

        if type(rebinning_spec) is not dict:
            raise ValueError("Rebinning specification must be a dictionary or a path to a JSON file.")

        #for consistency check conservation of total sum
        startsum = data.sum(axis=None)

        #check consistency of inputs
        for i, specblock in enumerate(rebinning_spec['spec']):
            for name in self._axis_names:
                if name not in specblock:
                    raise ValueError(f"Axis {name} not found in rebinning specification.")
            for name in specblock:
                if name not in self._axis_names:
                    raise ValueError(f"Axis {name} not found in histogram axes.")
                if type(specblock[name]) is not list:
                    raise ValueError(f"Axis {name} in rebinning specification must be a list of indices.")
                if np.min(specblock[name]) < 0:
                    raise ValueError(f"Axis {name} in rebinning specification contains negative indices.")

        #get the rebinned data
        extradims = list(data.shape[1:])
        result = np.empty((0, *extradims))
        for specblock in rebinning_spec['spec']:
            nextvals = self._get_specblock_binning(data, specblock)
            nextvals = nextvals.reshape((-1, *extradims))
            result = np.append(result, nextvals, axis=0)

        #get the new binning
        newbinning = ArbitraryBinning()
        newbinning._blocks = self._blocks[0].rebin(rebinning_spec)
        newbinning._axis_names = self._axis_names
        newbinning._Nax = self._Nax
            
        endsum = result.sum(axis=None)

        if not np.isclose(startsum, endsum):
            raise ValueError(f"Rebinning changed the sum of the data: {startsum} -> {endsum}")

        return result, newbinning

    def rebin_cov2d(self, data : np.ndarray, rebinning_spec : Union[str, dict]) -> Tuple[np.ndarray, 'ArbitraryBinning']:
        '''
        Wrapper for rebin() to handle 2D (usually covariance) matrices. 
        Just calls rebin() twice, transposing the data inbetween to index the other axis
        
        :param self: This object
        :param data: The data to rebin
        :type data: np.ndarray
        :param rebinning_spec: The rebinning specification, either as a dictionary or a path to a JSON file
        :type rebinning_spec: Union[str, dict]
        :return: A tuple containing the rebinned data and a new ArbitraryBinning instance representing the new binning structure
        :rtype: Tuple[ndarray[Any, Any], ArbitraryBinning]
        '''
        result, newbinning = self.rebin(data.T, rebinning_spec)
        result, _ = self.rebin(result.T, rebinning_spec)
        return result, newbinning

    def get_fluxes_shapes_cov2d(self,
                                fluxes : np.ndarray, 
                                shapes : np.ndarray,
                                cov : np.ndarray, 
                                axes : List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Get covariance of fluxes and shapes as returned by get_fluxes_shapes()
        
        :param self: This object
        :param data: 2d Covariance matrix
        :type data: np.ndarray
        :param axes: List of axes along which to take fluxes and shapes
        :type axes: List[str]
        :return: (covflux, covshape, covfluxshape)
        :rtype: Tuple[ndarray[_AnyShape, dtype[Any]], ndarray[_AnyShape, dtype[Any]], ndarray[_AnyShape, dtype[Any]]]
        '''
        blocks = self.get_blocks(axes)
        Nflux = len(fluxes)
        Nshape = len(shapes)

        covflux = np.zeros(shape=(Nflux, Nflux), dtype=fluxes.dtype)
        covfluxshape = np.zeros((Nflux, Nshape), dtype=cov.dtype)
        covshapes = np.zeros_like(cov) 

        #mapping from shape to flux
        fluxindex = np.zeros(Nshape, dtype=np.int32)
        for i, block in enumerate(blocks):
            fluxindex[block['slice']] = i

        for a, blockA in enumerate(blocks):
            sliceA = blockA['slice']
            for b, blockB in enumerate(blocks):
                sliceB = blockB['slice']

                covflux[a, b] = np.sum(cov[sliceA, :][:, sliceB])

        #covfluxshape
        for a, blockA in enumerate(blocks):
            sliceA = blockA['slice']

            for b, blockB in enumerate(blocks):
                sliceB = blockB['slice']
                
                covfluxshape[a, sliceB] += np.sum(cov[sliceA, sliceB], axis=0) / fluxes[b]
                covfluxshape[a, sliceB] -= (shapes[sliceB]/fluxes[b])  * np.sum(cov[sliceA, :][:, sliceB], axis=None)

        #covshape
        for a, blockA in enumerate(blocks):
            sliceA = blockA['slice']

            for b, blockB in enumerate(blocks):
                sliceB = blockB['slice']

                covshapes[sliceA, :][:, sliceB] += cov[sliceA, :][:, sliceB] / (fluxes[a] * fluxes[b])

                covshapes[sliceA, :][:, sliceB] += (np.outer(shapes[sliceA], shapes[sliceB]) / (fluxes[a] * fluxes[b])) * np.sum(cov[sliceA, :][:, sliceB], axis=None)

                covshapes[sliceA, :][:, sliceB] -= np.outer(shapes[sliceA]/(fluxes[a] * fluxes[b]), np.sum(cov[sliceA, :][:, sliceB], axis=0))
                covshapes[sliceA, :][:, sliceB] -= np.outer(np.sum(cov[sliceA, :][:, sliceB], axis=1), shapes[sliceB]/(fluxes[a] * fluxes[b]))

        return covflux, covshapes, covfluxshape


    def get_fluxes_shapes(self, data : np.ndarray, axes : List[str]) -> Tuple[np.ndarray, np.ndarray, 'ArbitraryBinning']:
        '''
        Get fluxes and shapes along specified axes. 
        The "fluxes" are the integrated distributions per block along the specified axes.
        The "shapes" are the normalized distributions per block along the specified axes. 
        For example, say you have a 3D distribution in (x,y,z) and you call get_fluxes_shapes(data, ['x','y']).
        The result will be fluxes[i] = np.sum(data[xi,yi,:]) for each (xi,yi) in the binning
                       and shapes[xi,yi,:] = data[xi,yi,:]/fluxes[i].      
        
        The binning of the shapes array is exactly the original binning
        The binning of the fluxes binning is a new ArbitraryBinning with only the specified axes.
                       
        :param self: This object
        :param data: The data array to compute fluxes and shapes from
        :type data: np.ndarray
        :param axes: The axes along which to compute fluxes and shapes
        :type axes: List[str]
        :return: A tuple containing the fluxes, shapes, and a new ArbitraryBinning instance representing the flux binning
        :rtype: Tuple[ndarray[Any, Any], ndarray[Any, Any], ArbitraryBinning]
        '''
        axisblocks = self.get_blocks(axes)
        #if type(data) is np.ndarray:
        shapes = np.ones(data.shape, dtype=data.dtype)
        fluxes = np.empty(len(axisblocks), dtype=data.dtype)
        #else:
        #    shapes = torch.ones(data.shape, dtype=data.dtype, device=data.device)
        #    fluxes = torch.ones(len(axisblocks), dtype=data.dtype, device=data.device)

        fluxbinning = ArbitraryBinning()
        fluxbinning._Nax = len(axes)
        fluxbinning._axis_names = axes

        for i, block in enumerate(axisblocks):
            indexing = block['slice']
            #if type(data) is torch.Tensor:
            #    fluxes[i] *= torch.sum(data[indexing])
            #else:
            fluxes[i] = np.sum(data[indexing])

            shapes[indexing] = data[indexing]/fluxes[i]

            binningblock = _BinningBlock()
            binningblock.Nax = len(axes)
            binningblock.axis_names = axes
            for ax in axes:
                binningblock.extents.append(1)
                binningblock.ax_details[ax] = {
                    'edges' : block['edges'][ax],
                    'extent' : 1,
                    'minedge' : block['edges'][ax][0],
                    'maxedge' : block['edges'][ax][1],
                }
            binningblock.total_size = 1
            binningblock.offset = i
            binningblock.calculate_strides()

            fluxbinning._blocks.append(binningblock)

        return fluxes, shapes, fluxbinning

    def merge_fluxes_shapes(self, fluxes : np.ndarray, shapes : np.ndarray, fluxbinning : 'ArbitraryBinning') -> np.ndarray:
        '''
        Merge fluxes and shapes arrays back into full distribution.
        This is the inverse operation to get_fluxes_shapes().
        
        :param self: This object
        :param fluxes: Fluxes array
        :type fluxes: np.ndarray
        :param shapes: Shapes array
        :type shapes: np.ndarray
        :param fluxbinning: Binning for fluxes array
        :type fluxbinning: 'ArbitraryBinning'
        :return: The merged full distribution
        :rtype: ndarray[Any, Any]
        '''
        axes = fluxbinning.axis_names
        axisblocks = self.get_blocks(axes)

        #if type(fluxes) is torch.Tensor:
        #    result = torch.empty(shapes.shape, dtype=shapes.dtype, device=shapes.device)
        #else:
        result = np.empty(shapes.shape, dtype=shapes.dtype)

        for i, block in enumerate(axisblocks):
            indexing = block['slice']
            shape = shapes[indexing]
            flux = fluxbinning.get_slice(fluxes, block['edges'])
            result[indexing] = shape * flux

        return result

    def _get_specblock_binning(self, data : np.ndarray, specblock : dict) -> np.ndarray:
        '''
        Handle the rebinning of the data in a single block of the rebinning spec.

        NB this is not part of the public interface
        
        :param self: This object
        :param data: The data to rebin
        :type data: np.ndarray
        :param specblock: A single block in the rebinning specification, in the format
            { axis_name : [list of edge indices to keep] }
        :type specblock: dict
        :return: The data, rebinned according to the specblock
        :rtype: ndarray[Any, Any]
        '''
        #edge index ranges and sizes per axis
        ranges : dict = {name : (np.min(specblock[name]), np.max(specblock[name])) for name in self._axis_names}
        sizes = [np.max(specblock[name]) - np.min(specblock[name]) for name in self._axis_names]

        #get the slice of data corresponding to the full range of the specblock
        theslice = self._blocks[0].get_slice_from_indices(data, ranges)

        #preserve trailing data dimensions
        extradims = list(data.shape[1:])
        sizes = sizes + extradims

        #reshape to actually realize all of the binning axes
        theslice = theslice.reshape(sizes)

        #use np.add.reduceat to rebin according to the spec
        for i, name in enumerate(self._axis_names):
            theslice = np.add.reduceat(theslice, 
                                       np.asarray(specblock[name][:-1]) - np.min(specblock[name]),
                                       axis=i)
        return theslice


    def get_blocks(self, axes : List[str]) -> List[dict]:
        '''
        Get the the indices corresonding to all slices along axes specified in `axes`.
        For example, say you have a 3D binning in (x,y,z) and you call _get_blocks(['x', 'y']).
        The result will be the specs for all slices along x and y. 
        ie to make data[0,0,:], data[0,1,:], data[0,2,:], ..., data[1,0,:], data[1,1,:], etc.

        The format of the result is a list of dicts, each with format:
        {
            'slice' : slice object or np.ndarray of indices,
            'edges' : dict of { axis_name : (min_edge, max_edge) } specifying the slice
        }
        
        NB this is not part of the public interface.
        
        :param self: This object
        :param axes: List of axis names to get blocks for
        :type axes: List[str]
        :return: List of dicts specifying slices and edges
        :rtype: List[dict[Any, Any]]
        '''

        #get full edge list across all blocks:
        #1. get everything with repeats
        global_edges = {name : [] for name in self._axis_names}
        for name in self._axis_names:
            for block in self._blocks:
                global_edges[name] += block.ax_details[name]['edges']
        #2. uniquify and sort
        for name in self._axis_names:
            global_edges[name] = sorted(list(set(global_edges[name])))

        
        result = []

        #the implementation here is a bit brute-force, but I think it works

        #list of all bin indices in every requested axis
        ranges = [range(len(global_edges[name]) - 1) for name in axes]
        #iterate over all slices along requested axes = cross-product of all bins per axis
        for indices in itertools.product(*ranges):
            #build a "slice" dict according to axis edges
            theslice : dict = {}
            for i, name in enumerate(axes):
                theslice[name] = (global_edges[name][indices[i]], 
                                  global_edges[name][indices[i] + 1])

            #find all the flat indices corrdsponding to this slice
            globalindices = np.empty((0,), dtype=np.int64)
            for block in self._blocks:
                #clip the slice to this block
                blockslice : Union[dict, None] = block.clip_edges_to_block(theslice)
                if blockslice is not None: #if clipped slice is non-empty
                    #lookup indices from edges
                    blockslice = {name: block.edges_to_indices(name, blockslice[name]) for name in blockslice}
                    #get the flat indices from the block
                    nextindices = block.offset+block.get_slice_indices(blockslice)
                    #append to global list
                    globalindices = np.append(globalindices, nextindices)

            #if indices are contiguous, return a slice object, otherwise return a sorted list of indices
            minindex = np.min(globalindices)
            maxindex = np.max(globalindices)
            size = maxindex - minindex + 1
            if size == 0:
                raise ValueError("Could not find ANY indices for slice: %s" % theslice)
            elif size == len(globalindices):
                result.append({
                    'slice' : slice(minindex, maxindex + 1, 1),
                    'edges' : theslice,
                })
            else:
                result.append({
                    'slice' : np.sort(globalindices),
                    'edges' : theslice,
                })
        return result

class ArbitraryGenRecoBinning:
    '''
    Docstring for AribtraryGenRecoBinning

    This class represents a pair of ArbitraryBinning objects, one for generator-level and one for reconstruction-level.
    It's mostly just a utility wrapper for handling both binnings together.
    '''
    def __init__(self):
        '''
        Initialize empty AribtraryGenRecoBinning
        
        :param self: This object
        '''
        self._genbinning : ArbitraryBinning = ArbitraryBinning()
        self._recobinning : ArbitraryBinning = ArbitraryBinning()

    @property
    def genbinning(self):
        '''
        Get the generator-level binning 
        
        :param self: This object
        '''
        return self._genbinning
    
    @property
    def recobinning(self):
        '''
        Get the reconstruction-level binning
        
        :param self: This object
        '''
        return self._recobinning

    def copy(self) -> 'ArbitraryGenRecoBinning':
        '''
        Make a deep copy of this AribtraryGenRecoBinning
        
        :param self: This object

        :return: A deep copy of this object
        :rtype: AribtraryGenRecoBinning
        '''
        result = ArbitraryGenRecoBinning()
        result.from_dict(self.to_dict())
        return result

    def setup_from_histograms(self, Hreco : hist.Hist, Hgen : hist.Hist):
        '''
        Initialize from a pair of hist.Hist objects
        
        :param self: This object
        :param Hreco: Reconstruction-level histogram
        :type Hreco: hist.Hist
        :param Hgen: Generator-level histogram
        :type Hgen: hist.Hist
        '''
        self._genbinning = ArbitraryBinning()
        self._genbinning.setup_from_histogram(Hgen)

        self._recobinning = ArbitraryBinning()
        self._recobinning.setup_from_histogram(Hreco)

    def dump_to_file(self, file : str):
        '''
        Dump to json file
        
        :param self: This object
        :param file: Path to the output file
        :type file: str
        '''
        print("Writing binning spec to file: %s" % file)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        resultdict = self.to_dict()
        with open(file, 'w') as f:
            json.dump(resultdict, f, indent=4)

    def load_from_file(self, file : str):
        '''
        Load from json file
        
        :param self: This object
        :param file: Path to the input file
        :type file: str
        '''
        print("Reading binning spec from file: %s" % file)

        with open(file, 'r') as f:
            resultdict = json.load(f)

        self.from_dict(resultdict)

    def to_dict(self) -> dict:
        '''
        Dump to python dictionary
        
        :param self: This object
        :return: The dictionary representation of this object
        :rtype: dict[Any, Any]
        '''
        gendict = self.genbinning.to_dict() 
        recodict = self.recobinning.to_dict()
        resultdict = {
            'gen': gendict,
            'reco': recodict
        }
        return resultdict

    def from_dict(self, resultdict : dict):
        '''
        Initialize from python dictionary
        
        :param self: This object
        :param resultdict: The dictionary representation of this object
        :type resultdict: dict
        '''
        self._genbinning = ArbitraryBinning()
        self._genbinning.from_dict(resultdict['gen'])

        self._recobinning = ArbitraryBinning()
        self._recobinning.from_dict(resultdict['reco'])

    def get_slice(self, data : np.ndarray, genreco : str, theedges : dict) -> np.ndarray:
        '''
        Get a slice of the data specified by the provided edges, for either generator-level or reconstruction-level binning.
        
        :param self: This object
        :param data: The data array to slice
        :type data: np.ndarray
        :param genreco: Specify 'gen' for generator-level or 'reco' for reconstruction-level binning
        :type genreco: str
        :param theedges: The edges specifying the slice. The format is
            { axis_name : (min_edge, max_edge) }
            This dictionary need not specify all axes in the binning
        :type theedges: dict
        :return: The sliced data
        :rtype: np.ndarray
        '''
        if genreco.lower().strip() == 'gen':
            thebinning = self.genbinning
        elif genreco.lower().strip() == 'reco':
            thebinning = self.recobinning
        else:
            raise ValueError("genreco must be 'gen' or 'reco'")

        return thebinning.get_slice(data, theedges)

    def get_slice_cov2d(self, data : np.ndarray, genreco : str, theedges : dict) -> np.ndarray:
        '''
        Get a slice of the 2D data (usually covariance matrix) specified by the provided edges, for either generator-level or reconstruction-level binning.
        This calls get_slice() twice, transposing the data inbetween to index the other axis.

        :param self: This object
        :param data: The 2D data array to slice
        :type data: np.ndarray
        :param genreco: Specify 'gen' for generator-level or 'reco' for reconstruction-level binning
        :type genreco: str
        :param theedges: The edges specifying the slice. The format is
            { axis_name : (min_edge, max_edge) }
            This dictionary need not specify all axes in the binning
        :type theedges: dict
        :return: The sliced data
        :rtype: np.ndarray
        '''
        if genreco.lower().strip() == 'gen':
            thebinning = self.genbinning
        elif genreco.lower().strip() == 'reco':
            thebinning = self.recobinning
        else:
            raise ValueError("genreco must be 'gen' or 'reco'")

        return thebinning.get_slice_cov2d(data, theedges)

    def get_slice_transfer2d(self, data : np.ndarray, theedges : dict) -> np.ndarray:
        '''
        Get a slice of the 2D transfer matrix data specified by the provided edges.
        This calls get_slice() twice, once for generator-level and once for reconstruction-level binning
        
        :param self: This object
        :param data: The 2D transfer matrix data array to slice
        :type data: np.ndarray
        :param theedges: The edges specifying the slice. The format is
            { axis_name : (min_edge, max_edge) }
            This dictionary need not specify all axes in the binning
        :type theedges: dict
        :return: The sliced data
        :rtype: ndarray[Any, Any]
        '''
        result = self.get_slice(data.T, genreco='gen', theedges=theedges)
        result = self.get_slice(result.T, genreco='reco', theedges=theedges)
        return result

    def project_out(self, data : np.ndarray, genreco : str, axis_name : str) -> Tuple[np.ndarray, 'ArbitraryBinning']:
        '''
        Project out an axis from the data, for either generator-level or reconstruction-level binning.
        
        :param self: This object
        :param data: The data array from which to project out an axis
        :type data: np.ndarray
        :param genreco: Specify 'gen' for generator-level or 'reco' for reconstruction-level binning
        :type genreco: str
        :param axis_name: The name of the axis to project out
        :type axis_name: str

        :return: A tuple containing the projected data and a new ArbitraryBinning instance representing the new binning structure
        :rtype: Tuple[ndarray[Any, Any], ArbitraryBinning]
        '''
        if genreco.lower().strip() == 'gen':
            thebinning = self.genbinning
        elif genreco.lower().strip() == 'reco':
            thebinning = self.recobinning
        else:
            raise ValueError("genreco must be 'gen' or 'reco'")

        return thebinning.project_out(data, axis_name)

    def project_out_cov2d(self, data : np.ndarray, genreco : str, axis_name : str) -> Tuple[np.ndarray, 'ArbitraryBinning']:
        '''
        Project out an axis from the 2D data (usually covariance matrix), for either generator-level or reconstruction-level binning.
        This calls project_out() twice, transposing the data inbetween to index the other axis
        
        :param self: This object
        :param data: The 2D data array from which to project out an axis
        :type data: np.ndarray
        :param genreco: Specify 'gen' for generator-level or 'reco' for reconstruction-level binning
        :type genreco: str
        :param axis_name: The name of the axis to project out
        :type axis_name: str
        :return: A tuple containing the projected data and a new ArbitraryBinning instance representing the new binning structure
        :rtype: Tuple[ndarray[Any, Any], ArbitraryBinning]
        '''
        if genreco.lower().strip() == 'gen':
            thebinning = self.genbinning
        elif genreco.lower().strip() == 'reco':
            thebinning = self.recobinning
        else:
            raise ValueError("genreco must be 'gen' or 'reco'")

        return thebinning.project_out_cov2d(data, axis_name)

    def project_out_transfer2d(self, data : np.ndarray, axis_name : str) -> Tuple[np.ndarray, 'ArbitraryGenRecoBinning']:
        '''
        Project out an axis from the 2D transfer matrix data.
        This calls project_out() twice, once for generator-level and once for reconstruction-level binning
        
        :param self: Description
        :param data: Description
        :type data: np.ndarray
        :param axis_name: Description
        :type axis_name: str
        :return: Description
        :rtype: Tuple[list[Any] | ndarray[Any, Any], AribtraryGenRecoBinning]
        '''
        result, newbinning_gen = self.genbinning.project_out(data.T, axis_name)
        result, newbinning_reco = self.recobinning.project_out(result.T, axis_name)

        newbinning = ArbitraryGenRecoBinning()
        newbinning._genbinning = newbinning_gen
        newbinning._recobinning = newbinning_reco

        return result, newbinning

    def rebin(self, data : np.ndarray, genreco : str, rebinning_spec : Union[str, dict]) -> Tuple[np.ndarray, 'ArbitraryBinning']:
        '''
        Rebin data according to the supplied specification, for either generator-level or reconstruction-level binning.
        NB only rectangular binnings (represented by a single _BinningBlock) can be rebinned.
        
        :param self: This object
        :param data: The data to rebin
        :type data: np.ndarray
        :param genreco: Specify 'gen' for generator-level or 'reco' for reconstruction-level binning
        :type genreco: str
        :param rebinning_spec: The rebinning specification, either as a dictionary or a path to a JSON file
        :type rebinning_spec: Union[str, dict]
        :return: A tuple containing the rebinned data and a new ArbitraryBinning instance representing the new binning structure
        :rtype: Tuple[ndarray[Any, Any], ArbitraryBinning]
        '''
        if genreco.lower().strip() == 'gen':
            thebinning = self.genbinning
        elif genreco.lower().strip() == 'reco':
            thebinning = self.recobinning
        else:
            raise ValueError("genreco must be 'gen' or 'reco'")
        
        if rebinning_spec is None:
            return data, thebinning.copy()

        return thebinning.rebin(data, rebinning_spec)

    def rebin_cov2d(self, data : np.ndarray, genreco : str, rebinning_spec : Union[str, dict]) -> Tuple[np.ndarray, 'ArbitraryBinning']:
        '''
        Rebin 2D data (usually covariance matrix) according to the supplied specification, for either generator-level or reconstruction-level binning.
        This calls rebin() twice, transposing the data inbetween to index the other axis
        
        :param self: This object
        :param data: The data to rebin
        :type data: np.ndarray
        :param genreco: Specify 'gen' for generator-level or 'reco' for reconstruction-level binning
        :type genreco: str
        :param rebinning_spec: The rebinning specification, either as a dictionary or a path to a JSON file
        :type rebinning_spec: Union[str, dict]
        :return: A tuple containing the rebinned data and a new ArbitraryBinning instance representing the new binning structure
        :rtype: Tuple[ndarray[Any, Any], ArbitraryBinning]
        '''
        if genreco.lower().strip() == 'gen':
            thebinning = self.genbinning
        elif genreco.lower().strip() == 'reco':
            thebinning = self.recobinning
        else:
            raise ValueError("genreco must be 'gen' or 'reco'")

        if rebinning_spec is None:
            return data, thebinning

        return thebinning.rebin_cov2d(data, rebinning_spec)

    def rebin_transfer2d(self, data : np.ndarray, rebinning_reco : Union[str, dict], rebinning_gen : Union[str, dict]) -> Tuple[np.ndarray, 'ArbitraryGenRecoBinning']:
        '''
        Rebin 2D transfer matrix data according to the supplied specifications, once for generator-level and once for reconstruction-level binning.
        This calls rebin() twice, once for generator-level and once for reconstruction-level binning
        
        :param self: This object
        :param data: The data to rebin
        :type data: np.ndarray
        :param rebinning_reco: The rebinning specification for reconstruction-level binning, either as a dictionary or a path to a JSON file
        :type rebinning_reco: Union[str, dict]
        :param rebinning_gen: The rebinning specification for generator-level binning, either as a dictionary or a path to a JSON file
        :type rebinning_gen: Union[str, dict]
        :return: A tuple containing the rebinned data and a new AribtraryGenRecoBinning instance representing the new binning structure
        :rtype: Tuple[ndarray[Any, Any], AribtraryGenRecoBinning]
        '''
        result, newbinning_gen = self.rebin(data.T, 'gen', rebinning_gen)
        result, newbinning_reco = self.rebin(result.T, 'reco', rebinning_reco) 

        newbinning = ArbitraryGenRecoBinning()
        newbinning._genbinning = newbinning_gen
        newbinning._recobinning = newbinning_reco

        return result, newbinning
