import awkward
import uproot
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from test_parameters import compare_parameters

from awkward_zipper import NanoAOD

# load a test root file
file_name = "tests/samples/nano_dy.root"
# Create a TTree from root
tree = uproot.open(file_name)["Events"]
# TTree -> awkward.Array[awkward.Record[str, awkward.Array]]
array = tree.arrays(ak_add_doc=True)

# construct an awkward array using awkward-zipper
restructure = NanoAOD(version="latest")
zipper_array = restructure(array)

# construct an awkward array using coffea
events = NanoEventsFactory.from_root(
    {file_name: "Events"},
    schemaclass=NanoAODSchema,
    mode="eager",
)
coffea_array = events.events()


def test_nano_dy_whole():
    # test across coffea (constructed from the same nano_dy.root file)

    # awkward-zipper adds some additional parameters, so check_parameters=False
    # also there are some arrays consisting of nans, so use equal_nan=True
    assert awkward.array_equal(
        zipper_array, coffea_array, check_parameters=False, equal_nan=True
    )


def test_nano_dy_kernels():
    # test local2globalindex function
    # local2globalindex function in awkward-zipper adds to the parameters, that it outputs a global index array
    assert awkward.array_equal(
        zipper_array.Jet.electronIdx1G,
        coffea_array.Jet.electronIdx1G,
        check_parameters=False,
    )
    print(awkward.materialize(zipper_array.Jet.electronIdx1G))

    # test nestedindex function
    # nestedindex function in awkward-zipper adds to the parameters, that it outputs an array nested from two index arrays
    assert awkward.array_equal(
        zipper_array.FatJet.subJetIdxG,
        coffea_array.FatJet.subJetIdxG,
        check_parameters=False,
    )
    assert awkward.array_equal(
        zipper_array.Jet.muonIdxG, coffea_array.Jet.muonIdxG, check_parameters=False
    )
    assert awkward.array_equal(
        zipper_array.Jet.electronIdxG,
        coffea_array.Jet.electronIdxG,
        check_parameters=False,
    )
    print(awkward.materialize(zipper_array.FatJet.subJetIdxG))
    print(awkward.materialize(zipper_array.Jet.muonIdxG))
    print(awkward.materialize(zipper_array.Jet.electronIdxG))

    # test distinct_parent function
    assert awkward.array_equal(
        zipper_array.GenPart.distinctParentIdxG, coffea_array.GenPart.distinctParentIdxG
    )
    print(awkward.materialize(zipper_array.GenPart.distinctParentIdxG))

    # test children function
    assert awkward.array_equal(
        zipper_array.GenPart.childrenIdxG, coffea_array.GenPart.childrenIdxG
    )
    assert awkward.array_equal(
        zipper_array.GenPart.distinctChildrenIdxG,
        coffea_array.GenPart.distinctChildrenIdxG,
    )
    print(awkward.materialize(zipper_array.GenPart.childrenIdxG))
    print(awkward.materialize(zipper_array.GenPart.distinctChildrenIdxG))

    # test distinct_children_deep function
    assert awkward.array_equal(
        zipper_array.GenPart.distinctChildrenDeepIdxG,
        coffea_array.GenPart.distinctChildrenDeepIdxG,
    )
    print(awkward.materialize(zipper_array.GenPart.distinctChildrenDeepIdxG))


def test_behaviors():
    # behavior of coffea and zipper should be the same
    # except for Systematics, since they are not included in zipper

    # difference in behaviors
    diff = set(coffea_array.behavior) - set(zipper_array.behavior)

    # don't take Systematics into account
    for behavior in diff.copy():
        if any(
            string in behavior
            for string in ("Systematic", "UpDownSystematic", "UpDownMultiSystematic")
        ):
            diff.remove(behavior)

    assert len(diff) == 0

    # print the difference behavior
    print(
        {
            k: coffea_array.behavior[k]
            for k in set(coffea_array.behavior) - set(zipper_array.behavior)
        }
    )
    # test parents behavior
    assert awkward.array_equal(
        zipper_array.GenPart.parent, coffea_array.GenPart.parent, check_parameters=False
    )
    # compare parameters with custom function
    assert compare_parameters(
        zipper_array.GenPart.parent.layout, coffea_array.GenPart.parent.layout
    )
    print(zipper_array.GenPart.parent)


if __name__ == "__main__":
    test_nano_dy_whole()
    test_nano_dy_kernels()
    test_behaviors()
