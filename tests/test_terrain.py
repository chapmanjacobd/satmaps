import sys
from pathlib import Path

from osgeo import gdal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import satmaps
import terrain
import tiler


def test_generate_terrain_pmtiles_uses_terrarium_tiling(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)
    temp_dir = tmp_path / ".temp"
    temp_dir.mkdir()

    def fake_build_gebco_source_vrt(gebco_zip: str, output_vrt: str) -> str:
        assert gebco_zip == "gebco.zip"
        Path(output_vrt).write_text("source")
        return output_vrt

    monkeypatch.setattr(terrain.ocean, "build_gebco_source_vrt", fake_build_gebco_source_vrt)
    monkeypatch.setattr(terrain.gdal, "WarpOptions", lambda **kwargs: kwargs)

    def fake_warp(destination: str, source: str, options=None):
        del source, options
        Path(destination).write_text("warped")
        return gdal.GetDriverByName("MEM").Create("", 1, 1, 1, gdal.GDT_Float32)

    monkeypatch.setattr(terrain.gdal, "Warp", fake_warp)

    captured: dict[str, object] = {}

    def fake_convert_raster_to_pmtiles(
        input_raster: str,
        output_path: str,
        *,
        tile_format: str,
        quality: int,
        resample_alg: str,
        chunk_zoom: int,
        parallel: int,
        blocksize: int,
        name: str,
        description: str,
        requested_bbox,
        tiling_options,
    ) -> satmaps.PackagedPMTiles:
        captured.update(
            {
                "input_raster": input_raster,
                "output_path": output_path,
                "tile_format": tile_format,
                "quality": quality,
                "resample_alg": resample_alg,
                "chunk_zoom": chunk_zoom,
                "parallel": parallel,
                "blocksize": blocksize,
                "name": name,
                "description": description,
                "requested_bbox": requested_bbox,
                "tiling_options": tiling_options,
            }
        )
        temp_mbtiles = temp_dir / "terrain.mbtiles"
        chunk_file = temp_dir / "terrain_chunk.mbtiles"
        temp_mbtiles.write_text("mbtiles")
        chunk_file.write_text("chunk")
        return satmaps.PackagedPMTiles(
            temp_mbtiles=str(temp_mbtiles),
            tiling_artifacts=tiler.TilingArtifacts(
                final_vrt=input_raster,
                cleanup_paths=[str(chunk_file)],
            ),
        )

    monkeypatch.setattr(satmaps, "convert_raster_to_pmtiles", fake_convert_raster_to_pmtiles)

    output = terrain.generate_terrain_pmtiles(
        "gebco.zip",
        "terrain.pmtiles",
        bbox=(-1.0, -2.0, 3.0, 4.0),
        temp_dir=str(temp_dir),
        resample_alg="bilinear",
        max_zoom=14,
        chunk_zoom=9,
        parallel=3,
        blocksize=256,
    )

    assert output == "terrain.pmtiles"
    assert captured["input_raster"].endswith("_3857.vrt")
    assert captured == {
        **captured,
        "output_path": "terrain.pmtiles",
        "tile_format": "png",
        "quality": 100,
        "resample_alg": "bilinear",
        "chunk_zoom": 9,
        "parallel": 3,
        "blocksize": 256,
        "name": "GEBCO Terrain",
        "description": "GEBCO Terrarium DEM",
        "requested_bbox": (-1.0, -2.0, 3.0, 4.0),
        "tiling_options": {"elevation_encoding": "terrarium"},
    }
    assert list(temp_dir.glob("*_source.vrt")) == []
    assert list(temp_dir.glob("*_3857.vrt")) == []
    assert not (temp_dir / "terrain.mbtiles").exists()
    assert not (temp_dir / "terrain_chunk.mbtiles").exists()
