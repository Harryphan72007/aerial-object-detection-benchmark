from aerial_benchmark.config import deep_merge, load_config, validate_config


def test_deep_merge_preserves_shared_protocol() -> None:
    merged = deep_merge({"protocol": {"epochs": 50, "seed": 17}}, {"protocol": {"seed": 42}})
    assert merged == {"protocol": {"epochs": 50, "seed": 42}}


def test_each_family_config_resolves() -> None:
    for name in ("cnn", "transformer", "vision_mamba", "rt_detr"):
        config = load_config(f"configs/{name}.yaml")
        validate_config(config)
        assert config["protocol"]["seeds"] == [17, 42, 73]
        assert config["model"]["checkpoint"] == "TBD"
