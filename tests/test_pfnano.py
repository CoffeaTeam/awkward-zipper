import awkward
import pytest
import uproot
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema

from awkward_zipper import PFNanoAOD

file_name = "tests/samples/pfnano.root"


@pytest.mark.parametrize("virtual", [False, True])
def test_pfnano_whole(virtual):
    access_log = []
    options = {"virtual": True, "access_log": access_log} if virtual else {}
    array = uproot.open(file_name)["Events"].arrays(ak_add_doc=True, **options)
    zipper_array = PFNanoAOD()(array)
    assert access_log == []

    coffea_array = NanoEventsFactory.from_root(
        {file_name: "Events"}, schemaclass=PFNanoAODSchema, mode="eager"
    ).events()
    # built by counts2nestedindex
    assert awkward.array_equal(
        zipper_array.Jet.pFCandsIdxG,
        coffea_array.Jet.pFCandsIdxG,
        check_parameters=False,
    )
    assert awkward.array_equal(
        zipper_array, coffea_array, check_parameters=False, equal_nan=True
    )
