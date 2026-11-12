from collections.abc import Mapping

import awkward


def _check_equal_lengths(
    contents: list[awkward.contents.Content],
) -> int | awkward._nplikes.shape.UnknownLength:
    unknown_length = awkward._nplikes.shape.unknown_length
    length = awkward._util.maybe_length_of(contents[0])
    if length is unknown_length:
        return length
    for layout in contents:
        _length = awkward._util.maybe_length_of(layout)
        if _length is unknown_length:
            continue
        if awkward._util.maybe_length_of(layout) != length:
            msg = "all arrays must have the same length"
            raise ValueError(msg)
    return length


def _non_materializing_get_field(record, field):
    assert isinstance(field, str)
    if isinstance(record, Mapping):
        return record[field]
    if isinstance(record, awkward.Array):
        record = record.layout
    assert isinstance(record, awkward.contents.RecordArray)
    index = record.field_to_index(field)
    return awkward.Array(record._contents[index])


def _as_layout(arr):
    """Return the low-level layout of an awkward.Array (or pass a Content through)."""
    if isinstance(arr, awkward.Array):
        return arr.layout
    return arr


def _jagged_content(arr):
    """Flat content of a single-jagged (ListOffsetArray) array, without materializing."""
    return _as_layout(arr).content


def _jagged_offsets(arr):
    """Event-level offsets Index of a single-jagged (ListOffsetArray) array."""
    return _as_layout(arr).offsets


def _record_length(contents):
    """Length to give a ``RecordArray`` built from ``contents``.

    Returns ``unknown_length`` when any content is still virtual: passing a
    concrete length would make ``RecordArray`` slice the contents to reconcile
    the length, which materializes the (virtual) shape generators. When
    everything is materialized we validate and return the real common length.
    """
    if not all(_as_layout(c).is_all_materialized for c in contents):
        return awkward._nplikes.shape.unknown_length
    return _check_equal_lengths(contents)


def _zip_jagged(members, offsets, record_name=None, parameters=None):
    """Build a jagged collection (``ListOffsetArray`` of ``RecordArray``).

    Parameters
    ----------
    members : dict[str, awkward.contents.Content]
        Mapping of field name to the flat content of each member (all members
        must share the same per-event ``offsets``).
    offsets : awkward.index.Index
        The shared event-level offsets.
    record_name : str, optional
        ``__record__`` parameter for the inner record.
    parameters : dict, optional
        Extra parameters for the inner record.
    """
    contents = tuple(members.values())
    fields = tuple(members.keys())
    params = {}
    if record_name is not None:
        params["__record__"] = record_name
    if parameters:
        params.update(parameters)
    record = awkward.contents.RecordArray(
        contents, fields, length=_record_length(contents), parameters=params
    )
    return awkward.contents.ListOffsetArray(offsets=offsets, content=record)


def _append_record_fields(listoffset, new_members):
    """Append flat-content fields to the record inside a ``ListOffsetArray``."""
    record = listoffset.content
    contents = list(record.contents) + list(new_members.values())
    fields = list(record.fields) + list(new_members.keys())
    new_record = awkward.contents.RecordArray(
        contents, fields, length=_record_length(contents), parameters=record.parameters
    )
    return awkward.contents.ListOffsetArray(
        offsets=listoffset.offsets, content=new_record
    )


def _maybe_raw_generator(buffer):
    if isinstance(buffer, awkward._nplikes.virtual.VirtualNDArray):
        if hasattr(buffer._generator, "__awkward_raw_generator__"):
            # essentially we're forgetting all shape_generators here
            return buffer._generator.__awkward_raw_generator__
        return buffer._generator
    # maybe assert that buffer is an array-like here?
    return lambda: buffer


def _rewrap(array):
    # here we rerun `ak.from_buffers` so that `ak.from_buffers` correctly recreates dynamically all VirtualNDArrays
    # the current buffers may have 'wrong' generators on them. Not "wrong" in the sense that they would yield wrong
    # buffers, but rather they would load too much.
    form, length, buffers = awkward.to_buffers(array)
    buffers = {key: _maybe_raw_generator(buffer) for key, buffer in buffers.items()}
    return awkward.from_buffers(
        form, length, buffers, behavior=array.behavior, attrs=array.attrs
    )


def _zip_arrays(members, record_name=None):
    """Zip a mapping of name -> array/Content into a record, array-based.

    Mirrors coffea's ``zip_forms``: if every member is a jagged
    ``ListOffsetArray`` the result is a single jagged record (``var * record``)
    sharing the first member's offsets; otherwise a flat ``RecordArray`` wrapping
    the members as-is (which may themselves be flat, jagged, or nested records).
    """
    names = list(members.keys())
    layouts = [
        m.layout if isinstance(m, awkward.Array) else m for m in members.values()
    ]
    params = {"__record__": record_name} if record_name is not None else {}

    if layouts and all(
        isinstance(layout, awkward.contents.ListOffsetArray) for layout in layouts
    ):
        offsets = layouts[0].offsets
        contents = [layout.content for layout in layouts]
        length = awkward._util.maybe_length_of(contents[0])
        record = awkward.contents.RecordArray(
            contents, names, length=length, parameters=params
        )
        return awkward.contents.ListOffsetArray(offsets=offsets, content=record)

    length = awkward._util.maybe_length_of(layouts[0]) if layouts else 0
    return awkward.contents.RecordArray(
        layouts, names, length=length, parameters=params
    )


def _materialized(buffer):
    """The concrete array behind ``buffer``, materializing it if it is virtual."""
    if isinstance(buffer, awkward._nplikes.virtual.VirtualNDArray):
        return buffer.materialize()
    return buffer


def _lazy_buffer(nplike, shape, dtype, generator):
    return awkward._nplikes.virtual.VirtualNDArray(
        nplike=nplike, shape=shape, dtype=dtype, generator=generator
    )


def _slice_content(content, bounds):
    """Slice ``content`` to ``bounds()`` lazily; ``None`` if its layout is not supported."""
    nplike = content.backend.nplike
    unknown_length = awkward._nplikes.shape.unknown_length
    if isinstance(content, awkward.contents.NumpyArray):
        data = content.data
        # `.shape` would materialize a virtual buffer whose length is not known yet
        inner_shape = (
            data.inner_shape
            if isinstance(data, awkward._nplikes.virtual.VirtualNDArray)
            else data.shape[1:]
        )

        def generator():
            start, stop = bounds()
            return _materialized(data)[start:stop]

        return awkward.contents.NumpyArray(
            _lazy_buffer(nplike, (unknown_length, *inner_shape), data.dtype, generator),
            parameters=content.parameters,
        )
    if isinstance(content, awkward.contents.RecordArray):
        contents = [_slice_content(c, bounds) for c in content.contents]
        if any(c is None for c in contents):
            return None
        return awkward.contents.RecordArray(
            contents,
            content.fields,
            length=unknown_length,
            parameters=content.parameters,
        )
    return None


def _start_at_zero(layout, length):
    """Pack a ``ListOffsetArray`` of ``length`` lists so its offsets start at 0 and its content ends at the last one.

    uproot reads an RNTuple entry range in whole clusters, so a list read past
    entry 0 has offsets that start past 0, and its content runs on past the last
    entry read. The layouts take a list's content as flat data and pair it with
    offsets built from counts, which needs the packed lists a TTree read gives.
    Virtual lists are packed when their buffers are materialized.
    """
    if layout.is_all_materialized:
        return layout.to_packed(False)

    offsets = layout.offsets.data

    def bounds():
        materialized = _materialized(offsets)
        return int(materialized[0]), int(materialized[-1])

    content = _slice_content(layout.content, bounds)
    if content is None:
        return layout

    def zero_based_offsets():
        materialized = _materialized(offsets)
        return materialized - materialized[0]

    return awkward.contents.ListOffsetArray(
        awkward.index.Index(
            _lazy_buffer(
                layout.backend.nplike, (length + 1,), offsets.dtype, zero_based_offsets
            )
        ),
        content,
        parameters=layout.parameters,
    )


def _counts_from_offsets(offsets, length, parameters):
    """Counts of the ``length`` lists of an offsets ``Index``, computed lazily if it is virtual."""
    import numpy as np

    data = offsets.data
    if not isinstance(data, awkward._nplikes.virtual.VirtualNDArray):
        return awkward.contents.NumpyArray(np.diff(data), parameters=parameters)
    return awkward.contents.NumpyArray(
        _lazy_buffer(
            data.nplike,
            (length,),
            data.dtype,
            lambda: np.diff(_materialized(data)),
        ),
        parameters=parameters,
    )


def _total_items(offsets):
    """Number of flat items implied by an offsets array (``offsets[-1]``).

    Returns an unknown length when the offsets are virtual, so reading it does
    not materialize the counts.
    """
    import numpy as np

    data = offsets.data if isinstance(offsets, awkward.index.Index) else offsets
    if isinstance(data, np.ndarray):
        return int(data[-1])
    return awkward._nplikes.shape.unknown_length
