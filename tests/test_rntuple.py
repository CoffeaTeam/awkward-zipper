import awkward
import pytest
import uproot
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

from awkward_zipper import NanoAOD

ttree_file = "tests/samples/nano_dy.root"
# the same events converted by the RNTuple importer: one flat field per branch
rntuple_file = "tests/samples/nano_dy_rntuple.root"

# uproot reads an RNTuple entry range in whole clusters, so lists read from a
# range that does not start at 0 or end at the last entry carry extra content
entry_ranges = [(None, None), (10, 30), (0, 20)]


@pytest.fixture(scope="module")
def nested_files(tmp_path_factory):
    """A skim written as an RNTuple with one record field per collection, and as a TTree."""
    events = NanoEventsFactory.from_root(
        {ttree_file: "Events"}, schemaclass=NanoAODSchema, mode="eager"
    ).events()
    writeable = NanoAODSchema.uproot_writeable(events)
    directory = tmp_path_factory.mktemp("nested")
    rntuple, ttree = str(directory / "rntuple.root"), str(directory / "ttree.root")
    with uproot.recreate(rntuple) as f:
        f.mkrntuple("Events", writeable)
    with uproot.recreate(ttree) as f:
        f.mktree("Events", writeable)
    return rntuple, ttree


def coffea_events(path, entry_start, entry_stop):
    return NanoEventsFactory.from_root(
        {path: "Events"},
        schemaclass=NanoAODSchema,
        mode="eager",
        entry_start=entry_start,
        entry_stop=entry_stop,
    ).events()


def zipper_events(path, entry_start, entry_stop, virtual):
    """Events built by awkward-zipper, and what uproot read while building them."""
    access_log = []
    options = {"virtual": True, "access_log": access_log} if virtual else {}
    array = uproot.open(path)["Events"].arrays(
        entry_start=entry_start, entry_stop=entry_stop, ak_add_doc=True, **options
    )
    events = NanoAOD()(array)
    return events, list(access_log)


@pytest.mark.parametrize("virtual", [False, True])
@pytest.mark.parametrize(("entry_start", "entry_stop"), entry_ranges)
def test_flat_rntuple(virtual, entry_start, entry_stop):
    events, construction_reads = zipper_events(
        rntuple_file, entry_start, entry_stop, virtual
    )
    assert construction_reads == []
    assert not [field for field in events.fields if field.startswith("_collection")]
    assert awkward.array_equal(
        events,
        coffea_events(ttree_file, entry_start, entry_stop),
        check_parameters=False,
        equal_nan=True,
    )


@pytest.mark.parametrize("virtual", [False, True])
@pytest.mark.parametrize(("entry_start", "entry_stop"), entry_ranges)
def test_nested_rntuple(nested_files, virtual, entry_start, entry_stop):
    rntuple, ttree = nested_files
    events, construction_reads = zipper_events(
        rntuple, entry_start, entry_stop, virtual
    )
    assert construction_reads == []
    expected = coffea_events(ttree, entry_start, entry_stop)
    assert type(events.Muon).__name__ == "MuonArray"
    assert awkward.array_equal(
        events.Muon.matched_jet.pt,
        expected.Muon.matched_jet.pt,
        check_parameters=False,
    )
    assert awkward.array_equal(events, expected, check_parameters=False, equal_nan=True)


def test_nested_rntuple_reads_one_leaf(nested_files):
    rntuple, _ = nested_files
    access_log = []
    array = uproot.open(rntuple)["Events"].arrays(virtual=True, access_log=access_log)
    events = NanoAOD()(array)
    awkward.materialize(events.Muon.pt)
    # the collection's offsets and the one leaf, not every leaf of the Muon field
    assert {read.field_name for read in access_log} == {"Muon", "pt"}
