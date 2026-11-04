"""Opt-in archive verification catches corruption before publication or decoding."""

import hashlib
import io
import json
import os
import struct
import zipfile

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeReadOptions, EpisodeRecorder, read_episodes


def bad_crc(payload, entry="reward.npy"):
    raw = bytearray(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        cursor = archive.start_dir
        for info in archive.infolist():
            if info.filename == entry:
                struct.pack_into("<I", raw, cursor + 16, info.CRC ^ 1)
                return bytes(raw)
            cursor += (
                46 + len(info.filename.encode()) + len(info.extra) + len(info.comment)
            )
    raise AssertionError("fixture lacks requested entry")


@pytest.fixture(params=["transitions", "compact"])
def verified_dataset(tmp_path, request):
    root = tmp_path / "episodes"
    with VectorEnv(
        Config(
            num_envs=2, max_steps=2, render="raster", channels=("rgb",), image_size=8
        )
    ) as env:
        with EpisodeRecorder(
            root,
            env,
            episode_format=request.param,
            write_workers=2,
            verify_archive=True,
        ) as writer:
            observation = env.observe()
            for _ in range(2):
                result = env.step(np.zeros((2, 2)))
                writer.append(observation, result)
                observation = result.observation
    return root


def test_verified_archives_record_hashes_and_roundtrip(verified_dataset):
    for path in verified_dataset.glob("*.npz"):
        marker = json.loads(path.with_suffix(".json").read_text())
        assert marker["archive_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        len(
            list(
                read_episodes(
                    verified_dataset, options=EpisodeReadOptions(verify_archive=True)
                )
            )
        )
        == 2
    )
    assert (
        json.loads((verified_dataset / "run.json").read_text())["verify_archive"]
        is True
    )


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("episode_format", ["transitions", "compact"])
def test_corrupt_compression_never_publishes_a_complete_episode(
    tmp_path, monkeypatch, workers, episode_format
):
    save = np.savez_compressed

    def corrupt(stream, **arrays):
        temporary = io.BytesIO()
        save(temporary, **arrays)
        stream.write(bad_crc(temporary.getvalue()))

    monkeypatch.setattr(np, "savez_compressed", corrupt)
    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        writer = EpisodeRecorder(
            tmp_path / "run",
            env,
            episode_format=episode_format,
            write_workers=workers,
            verify_archive=True,
        )
        with pytest.raises(zipfile.BadZipFile, match="env00000-episode00000000.npz"):
            with writer:
                writer.append(env.observe(), env.step(np.zeros((1, 2))))
    assert writer.closed and writer.failure is not None
    assert not list(writer.root.glob("*.npz"))
    assert not list(writer.root.glob("env*.json"))
    assert json.loads((writer.root / "run.json").read_text())["status"] == "failed"


def test_read_error_names_the_corrupt_archive(verified_dataset):
    path = sorted(verified_dataset.glob("*.npz"))[0]
    path.write_bytes(bad_crc(path.read_bytes()))
    with pytest.raises(zipfile.BadZipFile) as caught:
        list(read_episodes(verified_dataset))
    assert path.name in str(caught.value)
    assert isinstance(caught.value.__cause__, zipfile.BadZipFile)


def test_hash_check_rejects_corruption_before_opening_numpy_even_with_projection(
    verified_dataset, monkeypatch
):
    path = sorted(verified_dataset.glob("*.npz"))[0]
    path.write_bytes(path.read_bytes() + b"changed archive bytes")
    monkeypatch.setattr(
        np,
        "load",
        lambda *args, **kwargs: pytest.fail("unverified bytes reached NumPy"),
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        list(
            read_episodes(
                verified_dataset,
                options=EpisodeReadOptions(observation_keys=(), verify_archive=True),
            )
        )


def test_hash_and_decode_use_the_same_open_file_snapshot(
    verified_dataset, monkeypatch, tmp_path
):
    from embodiedforge import data

    original = data.stream_sha256
    replaced = []

    def replace_after_hash(stream):
        checksum = original(stream)
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"not the verified archive")
        os.replace(replacement, stream.name)
        replaced.append(stream.name)
        return checksum

    monkeypatch.setattr(data, "stream_sha256", replace_after_hash)
    assert (
        len(
            list(
                read_episodes(
                    verified_dataset, options=EpisodeReadOptions(verify_archive=True)
                )
            )
        )
        == 2
    )
    assert len(replaced) == 2


def test_legacy_archives_remain_readable_without_strict_hash_mode(tmp_path):
    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        with EpisodeRecorder(tmp_path / "run", env) as writer:
            writer.append(env.observe(), env.step(np.zeros((1, 2))))
    assert len(list(read_episodes(writer.root))) == 1
    with pytest.raises(ValueError, match="SHA256 is missing"):
        list(
            read_episodes(writer.root, options=EpisodeReadOptions(verify_archive=True))
        )


@pytest.mark.parametrize("value", [None, 0, 1, "yes"])
def test_integrity_options_require_boolean_values_before_filesystem_work(
    tmp_path, value
):
    with pytest.raises(ValueError, match="boolean"):
        EpisodeReadOptions(verify_archive=value)
    with VectorEnv(Config(num_envs=1)) as env:
        with pytest.raises(ValueError, match="boolean"):
            EpisodeRecorder(tmp_path / "run", env, verify_archive=value)
    assert not (tmp_path / "run").exists()
