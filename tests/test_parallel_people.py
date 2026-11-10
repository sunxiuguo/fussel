"""Regression coverage for synchronous face work returned by the photo pool."""

from unittest.mock import Mock

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from fussel.generator import generate
from fussel.generator.config import Config
from fussel.generator.generate import Albums, People, Photo


class SyntheticConfig:
    def __init__(self, enabled=True, workers=2):
        self.enabled = enabled
        self.workers = workers

    def getKey(self, key, default=None):
        return {
            "gallery.parallel_tasks": self.workers,
            "gallery.people.enable": self.enabled,
            "gallery.watermark.enable": False,
            "gallery.albums.recursive": True,
            "gallery.photo_sizes": [[32, 32]],
            "gallery.overwrite": False,
        }.get(key, default)


@pytest.fixture(autouse=True)
def reset_state():
    Config._instance = Albums._instance = People._instance = None
    yield
    Config._instance = Albums._instance = People._instance = None


def save_tagged(path, names=("Synthetic A", "Synthetic B")):
    info = PngInfo()
    regions = "".join(
        f'<rdf:Description mwg-rs:type="Face" mwg-rs:name="{name}">'
        '<mwg-rs:Area stArea:x="0.5" stArea:y="0.5" stArea:w="0.4" stArea:h="0.4"/>'
        "</rdf:Description>"
        for name in names
    )
    info.add_itxt("XML:com.adobe.xmp", f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF>{regions}</rdf:RDF></x:xmpmeta>')
    Image.new("RGB", (64, 64), "red").save(path, pnginfo=info)


@pytest.mark.parametrize("enabled", [True, False])
def test_real_pool_nested_and_cached_photos(tmp_path, enabled, monkeypatch):
    source = tmp_path / "input"
    nested = source / "nested"
    nested.mkdir(parents=True)
    save_tagged(source / "tagged.png")
    save_tagged(nested / "nested.png")
    Image.new("RGB", (64, 64), "blue").save(source / "plain.png")
    (source / "invalid.png").write_bytes(b"not an image")
    cfg = SyntheticConfig(enabled)
    Config.init(cfg)
    if not enabled:

        def unexpected(*args):
            pytest.fail("face detection must not run when people are disabled")

        monkeypatch.setattr(People, "detect_faces", unexpected)

    for _ in range(2):
        Albums._instance = People._instance = None
        Albums.instance().process_album_path(str(source), "Album", str(tmp_path / "out"), "/gallery", cfg)
        albums = Albums.instance().albums
        assert len(albums) == 2
        photos = [p for album in albums.values() for p in album.photos]
        assert len(photos) == 3
        assert sum(len(p.faces) for p in photos) == (4 if enabled else 0)
        assert all(not p.faces for p in photos if p.name == "plain.png")
        if enabled:
            assert set(People.instance().people) == {"Synthetic A", "Synthetic B"}
            for person in People.instance().people.values():
                assert len(person.photos) == 2
                assert all(any(p is candidate for candidate in photos) for p in person.photos)
        else:
            assert People.instance().people == {}


def test_public_photo_wrapper_preserves_queue_and_return(tmp_path):
    source = tmp_path / "tagged.png"
    save_tagged(source)
    output = tmp_path / "out"
    output.mkdir()
    Config.init(SyntheticConfig())
    queue = Mock()
    photo = Photo.process_photo("/external", str(source), source.name, "tagged", str(output), queue)
    assert isinstance(photo, Photo)
    queue.put.assert_called_once()
    item = queue.put.call_args.args[0]
    assert item[0] is photo
    assert item[1] == str(source)
    assert item[3:] == (str(output), "/external")
    assert Image.open(item[2]).size == (32, 32)


def test_every_pool_result_is_processed_without_a_queue(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    (source / "one.png").touch()
    cfg = SyntheticConfig()
    Config.init(cfg)
    photo = Photo("one.png", 32, 32, "/one.png", "/one.png", "one", {})
    pool = Mock()
    pool.__enter__ = Mock(return_value=pool)
    pool.__exit__ = Mock(return_value=False)
    pool.map.return_value = [(str(source / "one.png"), photo, "/largest.png")]
    monkeypatch.setattr(generate, "Pool", Mock(return_value=pool))

    def unexpected_queue():
        pytest.fail("Albums must not create an independent face queue")

    monkeypatch.setattr(generate, "Queue", unexpected_queue)
    detect = Mock()
    monkeypatch.setattr(People.instance(), "detect_faces", detect)
    Albums.instance().process_album_path(str(source), "Album", str(tmp_path / "out"), "/gallery", cfg)
    detect.assert_called_once_with(
        photo, str(source / "one.png"), "/largest.png", str(tmp_path / "out" / "album"), "/gallery/album"
    )
    assert Albums.instance()[0].photos[0] is photo


def test_real_pool_preserves_all_tagged_photos(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    for index in range(100):
        save_tagged(source / f"photo-{index:04d}.png", names=("Synthetic Person",))
    cfg = SyntheticConfig(workers=8)
    Config.init(cfg)

    Albums.instance().process_album_path(str(source), "Album", str(tmp_path / "out"), "/gallery", cfg)

    photos = Albums.instance()[0].photos
    assert len(photos) == 100
    assert sum(bool(photo.faces) for photo in photos) == 100
    assert len(People.instance().people["Synthetic Person"].photos) == 100
