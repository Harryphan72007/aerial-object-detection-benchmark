from src.data.collapse_classes import ClassMapping

def test_two_class_mapping():
    m=ClassMapping("2class")
    assert m.map_category(1)==1
    assert m.map_category(2)==1
    assert m.map_category(4)==2
    assert m.map_category(10)==2
    assert m.map_category(0) is None

def test_exclude_light_vehicles():
    m=ClassMapping("2class",exclude_light_vehicles=True)
    assert m.map_category(3) is None
    assert m.map_category(7) is None
    assert m.map_category(8) is None
    assert m.map_category(4)==2

def test_ten_class_preserves_ids():
    m=ClassMapping("10class")
    assert [m.map_category(i) for i in range(1,11)]==list(range(1,11))
