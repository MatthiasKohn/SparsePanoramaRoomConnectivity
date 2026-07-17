from sparsepano.datasets import Door, Pano, Scene, list_datasets


def test_dataset_contract_dataclasses():
    door = Door(pano_id="p0", bearing_deg=12.5)
    pano = Pano(id="p0", image_path="p0.jpg", room_id="r0", doors=[door])
    scene = Scene(
        dataset="dummy",
        scene_id="s0",
        panos=[pano],
        meters_per_unit=1.0,
        caps={"gt_poses": False, "gt_depth": False, "gt_doors": True, "gt_rooms": True},
    )
    assert scene.panos[0].doors[0].bearing_deg == 12.5


def test_builtin_zind_registered():
    assert "zind" in list_datasets()

