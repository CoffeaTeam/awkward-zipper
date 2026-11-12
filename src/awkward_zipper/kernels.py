import functools

import awkward
import numba
import numpy as np


def ensure_array(arraylike):
    """
    Converts arraylike to a Numpy array
    """
    if isinstance(arraylike, (awkward.contents.Content | awkward.Array)):
        return awkward.to_numpy(arraylike)
    if isinstance(arraylike, awkward.index.Index):
        return arraylike.data
    return np.asarray(arraylike)


def _materialize_input(array):
    if isinstance(array, int):
        return array
    # buffers such as the offsets from counts2offsets are not awkward arrays
    if isinstance(array, awkward._nplikes.virtual.VirtualNDArray):
        return array.materialize()
    return awkward.materialize(array)


# function: tp.Callable[[],]
def dispatch_wrap(function):
    @functools.wraps(function)
    def _wrapper(*input_arrays, data=None, dtype=np.int64):
        """
        Calls a function and passes it input_arrays as parameters. Returns a function result in eager case.
         In virtual case returns a result from a function wrapped in a Virtual Array.
        Args:
            input_arrays: function parameters. Awkward arrays and integers are accepted.
             This is a VirtualNDArray generator limitation.
            data: additional parameter for VirtualNDArray creation
            function: function to return
            dtype: additional parameter for VirtualNDArray creation

        Returns: function(input_arrays) or
         awkward.VirtualNDArray(generator=lambda: function(input_arrays))

        """
        # Virtual array
        if data is not None:
            return awkward._nplikes.virtual.VirtualNDArray(
                nplike=data._nplike,
                shape=(awkward._nplikes.shape.unknown_length,),
                dtype=dtype,
                generator=lambda: function(
                    *(_materialize_input(array) for array in input_arrays)
                ),
                shape_generator=None,
            )
        # concrete array
        return function(*input_arrays)

    return _wrapper


def local2globalindex(index, counts):
    """
    Convert a jagged local index to a global index

    This is the same as local2global(index, counts2offsets(counts))
    where local2global and counts2offsets are as in coffea.nanoevents.transforms

    Example usage:
    Index array
    [[], [1], [], [2, 3]]
    Counts array
    [8, 7, 4, 7]
    Output array will be:
    [[], [9], [], [21, 22]]
    (here 21=8+7+4+2)
    """

    @dispatch_wrap
    def _local2globalindex(index, counts):
        offsets = counts2offsets(counts)
        # make sure that offsets is always a Numpy array
        offsets = ensure_array(offsets)
        index = index.mask[index >= 0] + offsets[:-1]
        index = index.mask[index < offsets[1:]]  # guard against out of bounds
        # workaround ValueError: can not (unsafe) zip ListOffsetArrays with non-NumpyArray contents
        # index.type is N * var * int32?
        index = awkward.fill_none(index, -1)
        output = awkward.flatten(index)
        # make sure that output is always Numpy array
        return ensure_array(output)

    # Check if VirtualNDArray
    index_data = None
    if not all(awkward.to_layout(_).is_all_materialized for _ in (index, counts)):
        index_data = index.layout.content.data

    # resulting global index will have the same offsets as local index
    index_offsets = index.layout.offsets

    index_content = _local2globalindex(index, counts, data=index_data)
    # index_content shape would be index_data.shape
    index_content = awkward.contents.numpyarray.NumpyArray(index_content)
    # create new parameters for the final array
    parameters = awkward.parameters(index)
    parameters["__doc__"] = "global " + parameters.get("__doc__", "")
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            offsets=index_offsets,
            content=index_content,
            parameters=parameters,
        )
    )


def nestedindex(indices):
    """
    Concatenate a list of indices along a new axis
    Outputs a jagged array with same outer shape as index arrays

    Example usage:
    First index array
    [[0, 2, 4],
     [8, 6]]
    Second index array
    [[1, 3, 5],
     [-1, 7]]
    Output
    [[[0, 1], [2, 3], [4, 5]],
     [[8, -1], [6, 7]]]
    """

    @dispatch_wrap
    def _nestedindex_content(*indices):
        # return awkward.concatenate([idx[:, None] for idx in indexers], axis=1)
        flat_indices = []
        for idx in indices:
            # flatten the index
            flat_indices.append(awkward.Array(idx.layout.content))

        n = len(flat_indices)
        out = np.empty(n * len(flat_indices[0]), dtype="int64")
        for i, idx in enumerate(flat_indices):
            #  index arrays should all be same shape flat arrays
            out[i::n] = idx

        return out

    @dispatch_wrap
    def _get_nested_index_offsets(nested_index_content, n_indices):
        return np.arange(0, len(nested_index_content) + 1, n_indices, dtype=np.int64)

    def _combine_parameters(indices):
        parameters = {}
        for idx in indices:
            if "__doc__" in parameters:
                parameters["__doc__"] += " and "
            else:
                parameters["__doc__"] = "nested from "

            parameters["__doc__"] += awkward.parameters(idx)["__doc__"]
        return parameters

    if not all(
        isinstance(
            awkward.to_layout(index), awkward.contents.listoffsetarray.ListOffsetArray
        )
        for index in indices
    ):
        raise RuntimeError

    # Check if VirtualNDArray
    index_data = None
    if not all(awkward.to_layout(_).is_all_materialized for _ in indices):
        index_data = indices[0].layout.content.data

    # store offsets to later reapply them to the arrays
    offsets_stored = indices[0].layout.offsets
    nested_index_content = _nestedindex_content(
        *indices, data=index_data, dtype=np.int64
    )
    nested_index_content = awkward.contents.NumpyArray(nested_index_content)
    nested_index_offsets = _get_nested_index_offsets(
        nested_index_content, len(indices), data=index_data, dtype=np.int64
    )

    # combine offsets and content
    nested_index = awkward.contents.ListOffsetArray(
        awkward.index.Index64(nested_index_offsets),
        nested_index_content,
    )
    # combine the parameters
    parameters = _combine_parameters(indices)
    # reapply the offsets
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            offsets_stored,
            nested_index,
            parameters=parameters,
        )
    )


def counts2nestedindex(local_counts, target_offsets):
    """Turn jagged local counts into doubly-jagged global index into a target
    Outputs a jagged array with same axis-0 shape as counts axis-1

    Example usage:
    Local counts
    [[4, 3, 2],
     [4, 2]]
    Target offsets
    [9, 6]
    Target output
    [[[0, 1, 2, 3], [4, 5, 6], [7, 8]],
     [[9, 10, 11, 12], [13, 14]]]
    """

    @dispatch_wrap
    def _arange(array):
        return np.arange(array[-1], dtype=np.int64)

    @dispatch_wrap
    def _flatten(array):
        return awkward.flatten(array)

    if not isinstance(
        local_counts.layout, awkward.contents.listoffsetarray.ListOffsetArray
    ):
        raise RuntimeError
    if not isinstance(target_offsets.layout, awkward.contents.numpyarray.NumpyArray):
        raise RuntimeError

    # count offsets the same way as with counts2offsets in coffea.nanoevents.transforms
    offsets = counts2offsets(target_offsets)

    # store offsets to later reapply them to the arrays
    offsets_stored = local_counts.layout.offsets

    # Check if VirtualNDArray
    local_counts_data = local_counts_data_dtype = None
    if not all(
        awkward.to_layout(_).is_all_materialized for _ in (local_counts, target_offsets)
    ):
        local_counts_data = local_counts.layout.content.data
        local_counts_data_dtype = local_counts_data.dtype

    nested_index_content = _arange(offsets, data=local_counts_data)
    flat_counts = _flatten(
        local_counts, data=local_counts_data, dtype=local_counts_data_dtype
    )

    nested_index_offsets = counts2offsets(flat_counts)
    # combine offsets and content
    out = awkward.contents.ListOffsetArray(
        awkward.index.Index64(nested_index_offsets),
        awkward.contents.NumpyArray(nested_index_content),
    )

    # reapply the offsets
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            offsets_stored,
            out,
        )
    )


# TODO: add dispatch_wrapper decorator for this function too
def counts2offsets(counts):
    # Cumulative sum of counts
    def _counts2offsets(counts):
        # make sure that input is always Numpy array
        counts = ensure_array(counts)
        # awkward index default type is int64, so we use the same type for new arrays
        offsets = np.empty(len(counts) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        return offsets

    # VirtualNDArray
    # if isinstance(counts.layout.data, awkward._nplikes.virtual.VirtualNDArray):
    if (
        isinstance(counts, awkward._nplikes.virtual.VirtualNDArray)
        and not counts.is_materialized
    ):
        virtual_array = counts
    elif isinstance(counts, awkward.Array) and not counts.layout.is_all_materialized:
        virtual_array = counts.layout.data
    else:
        virtual_array = None

    if virtual_array is not None:
        return awkward._nplikes.virtual.VirtualNDArray(
            nplike=virtual_array._nplike,
            shape=(awkward._nplikes.shape.unknown_length,),
            dtype=np.int64,
            generator=lambda: _counts2offsets(virtual_array.materialize()),
            shape_generator=None,
        )
    # concrete array
    return _counts2offsets(counts.layout.data)


def full_like_from_counts(counts, fill_value):
    """Create a jagged array shaped like a collection with ``counts`` elements per
    event, with every element set to ``fill_value`` (as float32).

    This mirrors coffea's ``full_like_from_offsets`` transform, but works from the
    ``n{collection}`` counts branch instead of the ``o{collection}`` offsets branch.
    It is used to synthesize branches (e.g. ``Photon_mass``, ``Jet_charge``) that
    must be present for the 4-vector behaviors to work.

    Only the flat content of the returned array is ultimately consumed by the
    schema (the collection re-wraps it with its own offsets), but a valid
    ``ListOffsetArray`` is returned so that ``awkward.to_buffers`` yields the
    expected ``{"node0-offsets", "node1-data"}`` buffers.
    """

    def _offsets_from_counts(counts_arr):
        counts_arr = ensure_array(counts_arr)
        offsets = np.empty(len(counts_arr) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts_arr, out=offsets[1:])
        return offsets

    def _content_from_counts(counts_arr):
        counts_arr = ensure_array(counts_arr)
        n_elements = int(counts_arr.sum())
        return np.full(n_elements, fill_value, dtype=np.float32)

    if not counts.layout.is_all_materialized:
        virtual_array = counts.layout.data
        # the outer (per-event) length is known even when the data is virtual, so
        # give the offsets a concrete shape: this keeps awkward.to_buffers from
        # having to materialize the counts just to learn the list length
        n_events = counts.layout.length
        offsets = awkward._nplikes.virtual.VirtualNDArray(
            nplike=virtual_array._nplike,
            shape=(n_events + 1,),
            dtype=np.int64,
            generator=lambda: _offsets_from_counts(virtual_array.materialize()),
            shape_generator=None,
        )
        content = awkward._nplikes.virtual.VirtualNDArray(
            nplike=virtual_array._nplike,
            shape=(awkward._nplikes.shape.unknown_length,),
            dtype=np.float32,
            generator=lambda: _content_from_counts(virtual_array.materialize()),
            shape_generator=None,
        )
    else:
        offsets = _offsets_from_counts(counts)
        content = _content_from_counts(counts)

    return awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index(offsets), awkward.contents.NumpyArray(content)
        )
    )


def met_to_rho(met_content, eta_content):
    """Compute ``rho = met * cosh(eta)`` on flat contents, lazily.

    Mirrors coffea's ``met_to_rho`` transform. ``met_content`` and
    ``eta_content`` are flat ``NumpyArray`` contents; the result is returned as a
    flat ``NumpyArray`` content and is only evaluated when the buffer is read.
    """

    def _compute(met, eta):
        return ensure_array(met) * np.cosh(ensure_array(eta))

    met_data = met_content.data
    eta_data = eta_content.data
    is_virtual = awkward._nplikes.virtual.VirtualNDArray
    met_virtual = isinstance(met_data, is_virtual) and not met_data.is_materialized
    eta_virtual = isinstance(eta_data, is_virtual) and not eta_data.is_materialized

    if met_virtual or eta_virtual:
        base = met_data if met_virtual else eta_data
        result_dtype = np.result_type(met_content.dtype, eta_content.dtype)

        def _materialize(x):
            return x.materialize() if isinstance(x, is_virtual) else x

        result = awkward._nplikes.virtual.VirtualNDArray(
            nplike=base._nplike,
            shape=(awkward._nplikes.shape.unknown_length,),
            dtype=result_dtype,
            generator=lambda: _compute(_materialize(met_data), _materialize(eta_data)),
            shape_generator=None,
        )
    else:
        result = _compute(met_data, eta_data)

    return awkward.contents.NumpyArray(result)


@dispatch_wrap
@numba.njit
def _distinct_parent_kernel(allpart_parent, allpart_pdg):
    out = np.empty(len(allpart_pdg), dtype=np.int64)
    for i in range(len(allpart_pdg)):
        parent = allpart_parent[i]
        if parent < 0:
            out[i] = -1
            continue
        thispdg = allpart_pdg[i]
        while parent >= 0 and allpart_pdg[parent] == thispdg:
            if parent >= len(allpart_pdg):
                msg = "parent index beyond length of array!"
                raise RuntimeError(msg)
            parent = allpart_parent[parent]
        out[i] = parent
    return out


def distinct_parent(parents, pdg):
    """Compute first parent with distinct PDG id

    Signature: globalparents,globalpdgs,!distinctParent
    Expects global indexes, flat arrays, which should be same length
    """
    if not isinstance(pdg.layout, awkward.contents.listoffsetarray.ListOffsetArray):
        raise RuntimeError
    if not isinstance(parents.layout, awkward.contents.listoffsetarray.ListOffsetArray):
        raise RuntimeError

    # Check if VirtualNDArray
    parents_data = None
    if not all(awkward.to_layout(_).is_all_materialized for _ in (parents, pdg)):
        parents_data = parents.layout.content.data

    # store offsets to later reapply them
    result_offsets = parents.layout.offsets
    # calculate the contents
    result_content = _distinct_parent_kernel(
        awkward.Array(parents.layout.content),
        awkward.Array(pdg.layout.content),
        data=parents_data,
    )

    return awkward.Array(
        awkward.contents.ListOffsetArray(
            result_offsets,
            awkward.contents.NumpyArray(result_content),
        )
    )


@dispatch_wrap
@numba.njit
def _children_kernel_content(offsets_in, parentidx):
    content1_out = np.empty(len(parentidx), dtype=np.int64)

    offset1 = 0
    for record_index in range(len(offsets_in) - 1):
        start_src, stop_src = offsets_in[record_index], offsets_in[record_index + 1]

        for index in range(start_src, stop_src):
            for possible_child in range(index, stop_src):
                if parentidx[possible_child] == index:
                    if offset1 >= len(content1_out):
                        msg = "offset1 went out of bounds!"
                        raise RuntimeError(msg)
                    content1_out[offset1] = possible_child
                    offset1 = offset1 + 1

    return content1_out[:offset1]


@dispatch_wrap
@numba.njit
def _children_kernel_offsets(offsets_in, parentidx, content1_out):
    offsets1_out = np.empty(len(parentidx) + 1, dtype=np.int64)
    # content1_out = np.empty(len(parentidx), dtype=np.int64)
    offsets1_out[0] = 0

    offset0 = 1
    offset1 = 0
    for record_index in range(len(offsets_in) - 1):
        start_src, stop_src = offsets_in[record_index], offsets_in[record_index + 1]

        for index in range(start_src, stop_src):
            for possible_child in range(index, stop_src):
                if parentidx[possible_child] == index:
                    if offset1 >= len(content1_out):
                        msg = "offset1 went out of bounds!"
                        raise RuntimeError(msg)
                    # content1_out[offset1] = possible_child
                    offset1 = offset1 + 1
            if offset0 >= len(offsets1_out):
                msg = "offset0 went out of bounds!"
                raise RuntimeError(msg)
            offsets1_out[offset0] = offset1
            offset0 = offset0 + 1

    return offsets1_out


def children(counts, globalparents):
    """Compute children

    Signature: offsets,globalparents,!children
    Output will be a jagged array with same outer shape as globalparents content
    """
    if not isinstance(
        globalparents.layout, awkward.contents.listoffsetarray.ListOffsetArray
    ):
        raise RuntimeError
    offsets = counts2offsets(counts)

    # Check if VirtualNDArray
    globalparents_data = None
    if not all(
        awkward.to_layout(_).is_all_materialized for _ in (counts, globalparents)
    ):
        globalparents_data = globalparents.layout.content.data
    # store offsets to later reapply them
    result_offsets = globalparents.layout.offsets
    # Numba can't accept Virtual arrays directly, so wrap them with awkward
    ccontent = _children_kernel_content(
        awkward.Array(awkward.contents.NumpyArray(offsets)),
        awkward.Array(globalparents.layout.content),
        data=globalparents_data,
    )
    ccontent = awkward.contents.NumpyArray(ccontent)
    coffsets = _children_kernel_offsets(
        awkward.Array(awkward.contents.NumpyArray(offsets)),
        awkward.Array(globalparents.layout.content),
        awkward.Array(ccontent),
        data=globalparents_data,
    )

    out = awkward.contents.ListOffsetArray(
        awkward.index.Index64(coffsets),
        ccontent,
    )

    # reapply the offsets
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            result_offsets,
            out,
        )
    )


@dispatch_wrap
@numba.njit
def _distinct_children_deep_kernel_content(offsets_in, global_parents, global_pdgs):
    # offsets_out = np.empty(len(global_parents) + 1, dtype=np.int64)
    content_out = np.empty(len(global_parents), dtype=np.int64)
    # offsets_out[0] = 0

    # offset0 = 1
    offset1 = 0
    for record_index in range(len(offsets_in) - 1):
        start_src, stop_src = offsets_in[record_index], offsets_in[record_index + 1]

        for index in range(start_src, stop_src):
            this_pdg = global_pdgs[index]

            # only perform the deep lookup when this particle is not already part of a decay chain
            # otherwise, the same child indices would be repeated for every parent in the chain
            # which would require content_out to have a length that isa-priori unknown
            if (
                global_parents[index] >= 0
                and this_pdg != global_pdgs[global_parents[index]]
            ):
                # keep an index of parents with same pdg id
                parents = np.empty(stop_src - index, dtype=np.int64)
                parents[0] = index
                offset2 = 1

                # keep an additional index with parents that have at least one child
                parents_with_children = np.empty(stop_src - index, dtype=np.int64)
                offset3 = 0

                for possible_child in range(index, stop_src):
                    possible_parent = global_parents[possible_child]
                    possibe_child_pdg = global_pdgs[possible_child]

                    # compare with seen parents
                    for parent_index in range(offset2):
                        # check if we found a new child
                        if parents[parent_index] == possible_parent:
                            # first, remember that the parent has at least one child
                            if offset3 >= len(parents_with_children):
                                msg = "offset3 went out of bounds!"
                                raise RuntimeError(msg)
                            parents_with_children[offset3] = possible_parent
                            offset3 = offset3 + 1

                            # then, depending on the pdg id, add to parents or content
                            if possibe_child_pdg == this_pdg:
                                # has the same pdg id, add to parents
                                if offset2 >= len(parents):
                                    msg = "offset2 went out of bounds!"
                                    raise RuntimeError(msg)
                                parents[offset2] = possible_child
                                offset2 = offset2 + 1
                            else:
                                # has a different pdg id, add to content
                                if offset1 >= len(content_out):
                                    msg = "offset1 went out of bounds!"
                                    raise RuntimeError(msg)
                                content_out[offset1] = possible_child
                                offset1 = offset1 + 1
                            break

                # add parents with same pdg id that have no children
                for parent_index in range(1, offset2):
                    possible_child = parents[parent_index]
                    if possible_child not in parents_with_children[:offset3]:
                        if offset1 >= len(content_out):
                            msg = "offset1 went out of bounds! pt2"
                            raise RuntimeError(msg)
                        content_out[offset1] = possible_child

                        offset1 = offset1 + 1

    return content_out[:offset1]


@dispatch_wrap
@numba.njit
def _distinct_children_deep_kernel_offsets(
    offsets_in, global_parents, global_pdgs, content_out
):
    offsets_out = np.empty(len(global_parents) + 1, dtype=np.int64)
    offsets_out[0] = 0

    offset0 = 1
    offset1 = 0
    for record_index in range(len(offsets_in) - 1):
        start_src, stop_src = offsets_in[record_index], offsets_in[record_index + 1]

        for index in range(start_src, stop_src):
            this_pdg = global_pdgs[index]

            # only perform the deep lookup when this particle is not already part of a decay chain
            # otherwise, the same child indices would be repeated for every parent in the chain
            # which would require content_out to have a length that isa-priori unknown
            if (
                global_parents[index] >= 0
                and this_pdg != global_pdgs[global_parents[index]]
            ):
                # keep an index of parents with same pdg id
                parents = np.empty(stop_src - index, dtype=np.int64)
                parents[0] = index
                offset2 = 1

                # keep an additional index with parents that have at least one child
                parents_with_children = np.empty(stop_src - index, dtype=np.int64)
                offset3 = 0

                for possible_child in range(index, stop_src):
                    possible_parent = global_parents[possible_child]
                    possibe_child_pdg = global_pdgs[possible_child]

                    # compare with seen parents
                    for parent_index in range(offset2):
                        # check if we found a new child
                        if parents[parent_index] == possible_parent:
                            # first, remember that the parent has at least one child
                            if offset3 >= len(parents_with_children):
                                msg = "offset3 went out of bounds!"
                                raise RuntimeError(msg)
                            parents_with_children[offset3] = possible_parent
                            offset3 = offset3 + 1

                            # then, depending on the pdg id, add to parents or content
                            if possibe_child_pdg == this_pdg:
                                # has the same pdg id, add to parents
                                if offset2 >= len(parents):
                                    msg = "offset2 went out of bounds!"
                                    raise RuntimeError(msg)
                                parents[offset2] = possible_child
                                offset2 = offset2 + 1
                            else:
                                # has a different pdg id, add to content
                                if offset1 >= len(content_out):
                                    msg = "offset1 went out of bounds!"
                                    raise RuntimeError(msg)
                                offset1 = offset1 + 1
                            break

                # add parents with same pdg id that have no children
                for parent_index in range(1, offset2):
                    possible_child = parents[parent_index]
                    if possible_child not in parents_with_children[:offset3]:
                        if offset1 >= len(content_out):
                            msg = "offset1 went out of bounds! pt2"
                            raise RuntimeError(msg)

                        offset1 = offset1 + 1

            # finish this item by adding an offset
            if offset0 >= len(offsets_out):
                msg = "offset0 went out of bounds!"
                raise RuntimeError(msg)
            offsets_out[offset0] = offset1
            offset0 = offset0 + 1

    return offsets_out


def distinct_children_deep(counts, global_parents, global_pdgs):
    """Compute all distinct children, skipping children with same pdg id in between.

    Signature: offsets,global_parents,global_pdgs,!distinctChildrenDeep
    Expects global indexes, flat arrays, which should be same length
    """
    if not isinstance(
        global_parents.layout, awkward.contents.listoffsetarray.ListOffsetArray
    ):
        raise RuntimeError
    if not isinstance(
        global_pdgs.layout, awkward.contents.listoffsetarray.ListOffsetArray
    ):
        raise RuntimeError
    offsets = counts2offsets(counts)

    # Check if VirtualNDArray
    global_parents_data = None
    if not all(
        awkward.to_layout(_).is_all_materialized
        for _ in (counts, global_parents, global_pdgs)
    ):
        global_parents_data = global_parents.layout.content.data

    # store offsets to later reapply them
    result_offsets = global_parents.layout.offsets
    ccontent = _distinct_children_deep_kernel_content(
        awkward.Array(awkward.contents.NumpyArray(offsets)),
        awkward.Array(global_parents.layout.content),
        awkward.Array(global_pdgs.layout.content),
        data=global_parents_data,
    )
    ccontent = awkward.contents.NumpyArray(ccontent)
    coffsets = _distinct_children_deep_kernel_offsets(
        awkward.Array(awkward.contents.NumpyArray(offsets)),
        awkward.Array(global_parents.layout.content),
        awkward.Array(global_pdgs.layout.content),
        awkward.Array(ccontent),
        data=global_parents_data,
    )

    out = awkward.contents.ListOffsetArray(
        awkward.index.Index64(coffsets),
        ccontent,
    )

    # reapply the offsets
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            result_offsets,
            out,
        )
    )


def _lazy_flat_content(datas, fn, dtype):
    """Apply ``fn`` to flat buffers lazily, returning a ``NumpyArray`` content.

    Each element of ``datas`` is a raw buffer (numpy array or ``VirtualNDArray``).
    If any is an unmaterialized virtual buffer the result is a ``VirtualNDArray``
    whose generator runs ``fn`` on the materialized inputs; otherwise ``fn`` runs
    eagerly. Used for the PHYSLITE derived columns.
    """
    is_virtual = awkward._nplikes.virtual.VirtualNDArray

    def _materialize(x):
        return x.materialize() if isinstance(x, is_virtual) else x

    virtuals = [d for d in datas if isinstance(d, is_virtual) and not d.is_materialized]
    if virtuals:
        base = virtuals[0]
        result = awkward._nplikes.virtual.VirtualNDArray(
            nplike=base._nplike,
            shape=(awkward._nplikes.shape.unknown_length,),
            dtype=dtype,
            generator=lambda: fn(*(ensure_array(_materialize(d)) for d in datas)),
            shape_generator=None,
        )
    else:
        result = fn(*(ensure_array(d) for d in datas))
    return awkward.contents.NumpyArray(result)


def qoverp_to_p(qoverp_content):
    """p = 1 / |qOverP| (cast to the source dtype, matching coffea)."""
    dtype = qoverp_content.dtype
    return _lazy_flat_content(
        [qoverp_content.data],
        lambda q: (1.0 / np.abs(q)).astype(dtype),
        dtype,
    )


def qoverp_theta_to_pt(qoverp_content, theta_content):
    """pt = sin(theta) / |qOverP|."""
    dtype = qoverp_content.dtype
    return _lazy_flat_content(
        [qoverp_content.data, theta_content.data],
        lambda q, t: (np.sin(t) / np.abs(q)).astype(dtype),
        dtype,
    )


def full_like_from_content(source_content, fill_value):
    """A flat content shaped like ``source_content``, filled with ``fill_value``."""
    dtype = source_content.dtype
    return _lazy_flat_content(
        [source_content.data],
        lambda s: np.full(len(s), fill_value, dtype=dtype),
        dtype,
    )


def eventindex_content(offsets_index):
    """Flat int64 content giving each element's event index (from offsets)."""

    def _compute(offsets):
        offsets = ensure_array(offsets)
        counts = np.diff(offsets)
        return np.repeat(np.arange(len(counts), dtype=np.int64), counts)

    return _lazy_flat_content([offsets_index.data], _compute, np.int64)


def full_like_from_array(content, fill_value, dtype=np.float32):
    """Create a flat content shaped like ``content`` with every element set to
    ``fill_value``.

    Mirrors coffea's ``full_like_from_content`` transform (the ATLAS ntuple
    variant: takes a possibly jagged array, fills float32). Unlike
    :func:`full_like_from_counts` this works from an existing flat content, which is
    what the ATLAS ntuple schema needs: its collections carry their own offsets and
    there is no ``n{collection}`` counts branch to derive a shape from.

    A jagged source keeps its offsets, so the result lines up element-for-element
    with the collection it was derived from.

    Stays lazy: when the source buffer is virtual the result is a virtual buffer that
    only materializes the source to learn how many elements to fill.
    """
    layout = awkward.to_layout(content)

    if layout.is_list:
        return type(layout)(
            layout.offsets,
            full_like_from_array(layout.content, fill_value, dtype=dtype),
        )

    buffer = layout.data

    if isinstance(buffer, awkward._nplikes.virtual.VirtualNDArray):
        return awkward.contents.NumpyArray(
            awkward._nplikes.virtual.VirtualNDArray(
                nplike=buffer._nplike,
                # reuse the source shape so that consumers never have to materialize
                # the source just to learn the length
                shape=buffer.shape,
                dtype=np.dtype(dtype),
                generator=lambda: np.full(
                    len(buffer.materialize()), fill_value, dtype=dtype
                ),
                shape_generator=None,
            )
        )

    return awkward.contents.NumpyArray(np.full(len(buffer), fill_value, dtype=dtype))


def begin_end_counts(begin_content, end_content):
    """Flat ``end - begin`` counts for EDM4HEP begin/end index ranges (lazy)."""
    return _lazy_flat_content(
        [begin_content.data, end_content.data],
        lambda b, e: e.astype(np.int64) - b.astype(np.int64),
        np.int64,
    )


def begin_end_mapping(begin, end, target_content):
    """Group ``target_content`` into per-item lists using begin/end index ranges.

    EDM4HEP stores OneToMany relations and VectorMembers as flat per-event arrays
    plus per-item ``{member}_begin`` / ``{member}_end`` ranges. Those ranges are
    contiguous and exactly cover the target, so the mapping reduces to regrouping
    the target by ``counts = end - begin``.

    ``begin``/``end`` are the per-event jagged range arrays (``ListOffsetArray``);
    ``target_content`` is the flat content of the per-event target array. Returns
    a doubly-jagged ``ListOffsetArray`` (event -> item -> sub-items).
    """
    counts = begin_end_counts(begin.content, end.content)
    inner_offsets = counts2offsets(awkward.Array(counts))
    # coffea builds this via awkward.ArrayBuilder, which yields float64 regardless
    # of the target dtype; match that so the layouts compare equal
    if isinstance(target_content, awkward.contents.NumpyArray):
        target_content = _lazy_flat_content(
            [target_content.data],
            lambda x: np.asarray(x).astype(np.float64),
            np.float64,
        )
    inner = awkward.contents.ListOffsetArray(
        awkward.index.Index(inner_offsets), target_content
    )
    return awkward.contents.ListOffsetArray(begin.offsets, inner)


def _local2global_compute(index, index_offsets, target_offsets):
    """``index + target_offsets[:-1]`` per event, -1 where out of range."""
    index = np.asarray(index).astype(np.int64)
    index_offsets = np.asarray(index_offsets).astype(np.int64)
    target_offsets = np.asarray(target_offsets).astype(np.int64)
    counts = np.diff(index_offsets)
    starts = np.repeat(target_offsets[:-1], counts)
    stops = np.repeat(target_offsets[1:], counts)
    out = np.where(index >= 0, index + starts, -1)
    return np.where(out < stops, out, -1)


def regular_to_jagged(regular, dtype=np.float64):
    """Convert a ``RegularArray(size, NumpyArray)`` to a jagged list of ``dtype``.

    EDM4HEP stores fixed-size members (e.g. ``covMatrix.values[21]``) as a regular
    array; coffea's ArrayBuilder-based transform turns them into variable-length
    lists of float64. Evaluated lazily.
    """
    inner = regular.content
    size = regular.size
    offsets = _lazy_flat_content(
        [inner.data],
        lambda x: np.arange(0, len(x) + 1, size, dtype=np.int64),
        np.int64,
    )
    content = _lazy_flat_content(
        [inner.data], lambda x: np.asarray(x).astype(dtype), dtype
    )
    return awkward.contents.ListOffsetArray(awkward.index.Index(offsets.data), content)


def local2global(index, target_offsets):
    """Turn a jagged local index into a global index into a target collection.

    Mirrors coffea's ``local2global`` transform. ``index`` is a per-event jagged
    ``ListOffsetArray``; ``target_offsets`` is the target collection's per-event
    offsets buffer. Returns the flat (global) index content, evaluated lazily.
    """
    target_data = (
        target_offsets.data
        if isinstance(target_offsets, awkward.index.Index)
        else target_offsets
    )
    return _lazy_flat_content(
        [index.content.data, index.offsets.data, target_data],
        _local2global_compute,
        np.int64,
    )


def nested_local2global(nested_index, target_offsets):
    """``local2global`` for a doubly-jagged index (event -> item -> indices).

    ``nested_index`` is ``ListOffsetArray(ListOffsetArray(NumpyArray))``; the
    per-event grouping used for the offset lookup is the OUTER list.
    """
    inner = nested_index.content
    target_data = (
        target_offsets.data
        if isinstance(target_offsets, awkward.index.Index)
        else target_offsets
    )

    def _compute(flat_index, outer_offsets, inner_offsets, target_offs):
        outer_offsets = np.asarray(outer_offsets).astype(np.int64)
        inner_offsets = np.asarray(inner_offsets).astype(np.int64)
        # number of flat elements per event = inner_offsets[outer[e+1]] - inner_offsets[outer[e]]
        per_event = np.diff(inner_offsets[outer_offsets])
        event_offsets = np.empty(len(per_event) + 1, dtype=np.int64)
        event_offsets[0] = 0
        np.cumsum(per_event, out=event_offsets[1:])
        return _local2global_compute(flat_index, event_offsets, target_offs)

    return _lazy_flat_content(
        [
            inner.content.data,
            nested_index.offsets.data,
            inner.offsets.data,
            target_data,
        ],
        _compute,
        np.int64,
    )


def grow_local_index_to_target_shape(index, target_offsets):
    """Grow a local index to the target's shape, filling unreferenced slots with -1.

    Mirrors coffea's ``grow_local_index_to_target_shape``: for every element of the
    target, emit its local index if that index appears in ``index`` for the event,
    otherwise -1. Returns the flat content, evaluated lazily.
    """

    def _compute(idx_content, idx_offsets, tgt_offsets):
        idx_content = np.asarray(idx_content).astype(np.int64)
        idx_offsets = np.asarray(idx_offsets).astype(np.int64)
        tgt_offsets = np.asarray(tgt_offsets).astype(np.int64)
        out = np.empty(int(tgt_offsets[-1]), dtype=np.int64)
        for event in range(len(tgt_offsets) - 1):
            start, stop = int(tgt_offsets[event]), int(tgt_offsets[event + 1])
            all_index = np.arange(stop - start, dtype=np.int64)
            present = np.isin(
                all_index, idx_content[idx_offsets[event] : idx_offsets[event + 1]]
            )
            out[start:stop] = np.where(present, all_index, -1)
        return out

    target_data = (
        target_offsets.data
        if isinstance(target_offsets, awkward.index.Index)
        else target_offsets
    )
    return _lazy_flat_content(
        [index.content.data, index.offsets.data, target_data], _compute, np.int64
    )
