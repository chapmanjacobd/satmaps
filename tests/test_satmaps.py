import argparse
from pathlib import Path
import sys
import os
import uuid
from types import SimpleNamespace

import numpy as np
from osgeo import gdal, osr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import satmaps
from satmaps import (
    process_single_tile,
    list_mosaic_folders_for_tile,
    get_tile_paths,
    main
)

def test_list_mosaic_folders_for_tile_uses_cache(monkeypatch: object) -> None:
    # Pre-populate cache
    satmaps.S3_FOLDER_CACHE = {
        "2025/07/01": {"Sentinel-2_mosaic_2025_Q3_31TDF_0_0"}
    }
    
    found = list_mosaic_folders_for_tile("31TDF_0_0", ["2025/07/01"], ".cache")
    assert found == [("Sentinel-2_mosaic_2025_Q3_31TDF_0_0", "2025/07/01")]


def test_get_tile_paths_returns_s3_paths(monkeypatch: object) -> None:
    # Ensure local cache doesn't exist for this test
    monkeypatch.setattr("os.path.exists", lambda path: False)
    paths = get_tile_paths("Sentinel-2_mosaic_2025_Q3_31TDF_0_0", "2025/07/01", ".cache")
    assert paths['red'] == "/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/2025/07/01/Sentinel-2_mosaic_2025_Q3_31TDF_0_0/B04.tif"
    assert paths['green'] == "/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/2025/07/01/Sentinel-2_mosaic_2025_Q3_31TDF_0_0/B03.tif"
    assert paths['blue'] == "/vsis3/eodata/Global-Mosaics/Sentinel-2/S2MSI_L3__MCQ/2025/07/01/Sentinel-2_mosaic_2025_Q3_31TDF_0_0/B02.tif"


def test_process_single_tile_full_pipeline(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    
    # Mock data discovery
    monkeypatch.setattr("satmaps.list_mosaic_folders_for_tile", 
                        lambda tile, dates, cache: [("Sentinel-2_mosaic_2025_Q3_31TDF_0_0", "2025/07/01")])
    
    # Create fake input TIFs
    def fake_get_tile_paths(folder, date, cache, download=False):
        p = {}
        for b in ['red', 'green', 'blue']:
            path = tmp_path / f"{b}.tif"
            if not path.exists():
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.Create(str(path), 10, 10, 1, gdal.GDT_Int16)
                ds.SetGeoTransform((0, 1, 0, 10, 0, -1))
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(32631) # UTM zone 31N
                ds.SetProjection(srs.ExportToWkt())
                ds.GetRasterBand(1).Fill(1000)
                ds = None
            p[b] = str(path)
        return p

    monkeypatch.setattr("satmaps.get_tile_paths", fake_get_tile_paths)
    
    args = argparse.Namespace(
        cache=".cache",
        download=False,
        stats_min=0,
        stats_max=10000,
        no_soft_knee=False,
        no_grading=False,
        sb=0.3, hb=0.75, ss=1.4, ms=0.9, hs=0.5, exposure=1.0,
        gamma=1.0, sat=0.9, db=0.7, ls=0.7,
        resample_alg="bilinear"
    )
    
    out_path = process_single_tile("31TDF_0_0", ["2025/07/01"], args, "abc")
    
    assert out_path is not None
    assert Path(out_path).exists()
    
    ds = gdal.Open(out_path)
    assert ds.RasterCount == 4
    assert ds.GetRasterBand(1).DataType == gdal.GDT_Byte
    assert ds.GetRasterBand(4).GetColorInterpretation() == gdal.GCI_AlphaBand
    # Check projection is 3857
    srs = osr.SpatialReference(ds.GetProjection())
    assert srs.GetAttrValue("AUTHORITY", 1) == "3857"


def test_main_vrt_mode(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".temp").mkdir()
    
    monkeypatch.setattr(sys, "argv", ["satmaps.py", "31TDF", "--vrt", "--parallel", "1", "--date", "2025/07/01"])
    monkeypatch.setattr("satmaps.setup_gdal_cdse", lambda: None)
    
    # Mock process_single_tile to return a fake path
    def fake_process_single_tile(st, dates, args, uid):
        path = tmp_path / f"processed_{st}.tif"
        path.write_text("fake tif")
        return str(path)
    
    monkeypatch.setattr("satmaps.process_single_tile", fake_process_single_tile)
    
    # Mock gdal.BuildVRT
    monkeypatch.setattr("satmaps.gdal.BuildVRT", lambda out, src: Path(out).write_text("fake vrt"))
    
    main()
    
    # Check that the master VRT was created
    master_vrts = list(tmp_path.glob(".temp/master_*.vrt"))
    assert len(master_vrts) == 1
