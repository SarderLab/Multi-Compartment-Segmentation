"""
Tests for the notebook prediction path (predict_notebook / xml_suey_local).

These tests cover the output stage only — no WSI file or model weights required.
They verify that xml_suey_local produces one JSON file per tissue compartment,
matching the same polyline format as the Girder path (minus the attributes block).
"""

import json
import os
import glob
import numpy as np
import pytest
from argparse import Namespace

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multic.segmentationschool.Codes.IterativePredict_1X import xml_suey_local, NAMES


def make_args(output_path, min_size=None):
    return Namespace(
        output_path=output_path,
        min_size=min_size if min_size is not None else [30, 30, 30, 30, 30, 30],
        bordercrop=300,
    )


def make_wsi_mask_with_classes(height=200, width=200, class_values=(1, 3, 5)):
    """Create a synthetic wsiMask with square blobs for each class value."""
    mask = np.zeros((height, width), dtype='uint8')
    block = 40
    for idx, val in enumerate(class_values):
        x0 = (idx * 50) + 10
        y0 = 10
        mask[y0:y0 + block, x0:x0 + block] = val
    return mask


def get_output_files(tmp_path):
    """Return all per-class JSON files written into tmp_path."""
    return sorted(glob.glob(str(tmp_path / 'slide_*.json')))


def read_all_annotations(tmp_path):
    """Read and return all per-class annotation dicts (unwrapped from {"annotation": ...})."""
    return [json.load(open(f))["annotation"] for f in get_output_files(tmp_path)]


# output_path uses '_annotations' suffix so base_prefix strips to 'slide'
# resulting files: slide_cortical_interstitium.json, slide_tubules.json, etc.
OUTPUT_PATH = 'slide_annotations.json'


class TestXmlSueyLocal:

    def test_creates_per_class_files(self, tmp_path):
        args = make_args(str(tmp_path / OUTPUT_PATH))
        mask = make_wsi_mask_with_classes(class_values=(1, 3, 5))
        xml_suey_local(wsiMask=mask, args=args, classNum=7, downsample=1, glob_offset=[0, 0])
        files = get_output_files(tmp_path)
        assert len(files) == 3  # one file per detected class

    def test_one_file_per_class_not_combined(self, tmp_path):
        args = make_args(str(tmp_path / OUTPUT_PATH))
        mask = make_wsi_mask_with_classes(class_values=(1, 2, 3, 4, 5, 6))
        xml_suey_local(wsiMask=mask, args=args, classNum=7, downsample=1, glob_offset=[0, 0])
        files = get_output_files(tmp_path)
        assert len(files) == 6
        # No combined file should exist
        assert not os.path.exists(str(tmp_path / OUTPUT_PATH))

    def test_each_file_is_valid_json_with_annotation_root(self, tmp_path):
        args = make_args(str(tmp_path / OUTPUT_PATH))
        mask = make_wsi_mask_with_classes()
        xml_suey_local(wsiMask=mask, args=args, classNum=7, downsample=1, glob_offset=[0, 0])
        for f in get_output_files(tmp_path):
            with open(f) as fh:
                data = json.load(fh)
            assert isinstance(data, dict)
            assert 'annotation' in data
            assert 'elements' in data['annotation']
            assert 'name' in data['annotation']

    def test_file_names_match_compartment_names(self, tmp_path):
        args = make_args(str(tmp_path / OUTPUT_PATH))
        mask = make_wsi_mask_with_classes(class_values=(1, 2, 3, 4, 5, 6))
        xml_suey_local(wsiMask=mask, args=args, classNum=7, downsample=1, glob_offset=[0, 0])
        written_names = [json.load(open(f))['annotation']['name'] for f in get_output_files(tmp_path)]
        for name in written_names:
            assert name in NAMES

    def test_elements_are_polylines(self, tmp_path):
        args = make_args(str(tmp_path / OUTPUT_PATH))
        mask = make_wsi_mask_with_classes()
        xml_suey_local(wsiMask=mask, args=args, classNum=7, downsample=1, glob_offset=[0, 0])
        for annot in read_all_annotations(tmp_path):
            for elem in annot['elements']:
                assert elem['type'] == 'polyline'
                assert elem['closed'] is True
                assert elem['group'] == 'Segmented FTU'
                assert isinstance(elem['points'], list)
                assert all(len(p) == 3 for p in elem['points'])

    def test_no_attributes_metadata(self, tmp_path):
        args = make_args(str(tmp_path / OUTPUT_PATH))
        mask = make_wsi_mask_with_classes()
        xml_suey_local(wsiMask=mask, args=args, classNum=7, downsample=1, glob_offset=[0, 0])
        for annot in read_all_annotations(tmp_path):
            assert 'attributes' not in annot

    def test_no_empty_annotation_files(self, tmp_path):
        args = make_args(str(tmp_path / OUTPUT_PATH))
        mask = make_wsi_mask_with_classes(class_values=(1,))
        xml_suey_local(wsiMask=mask, args=args, classNum=7, downsample=1, glob_offset=[0, 0])
        for annot in read_all_annotations(tmp_path):
            assert len(annot['elements']) > 0

    def test_overwrites_existing_files(self, tmp_path):
        args = make_args(str(tmp_path / OUTPUT_PATH))
        mask = make_wsi_mask_with_classes(class_values=(1,))
        # Pre-write a stale file for the same class
        stale = str(tmp_path / f'slide_{NAMES[0]}.json')
        with open(stale, 'w') as f:
            json.dump({'stale': True}, f)
        xml_suey_local(wsiMask=mask, args=args, classNum=7, downsample=1, glob_offset=[0, 0])
        with open(stale) as f:
            data = json.load(f)
        assert 'annotation' in data
        assert 'stale' not in data

    def test_glob_offset_shifts_coordinates(self, tmp_path):
        out_a = str(tmp_path / 'a' / OUTPUT_PATH)
        out_b = str(tmp_path / 'b' / OUTPUT_PATH)
        os.makedirs(str(tmp_path / 'a'), exist_ok=True)
        os.makedirs(str(tmp_path / 'b'), exist_ok=True)
        mask = make_wsi_mask_with_classes(class_values=(1,))
        offset = [500, 300]

        xml_suey_local(wsiMask=mask, args=make_args(out_a), classNum=7, downsample=1, glob_offset=[0, 0])
        xml_suey_local(wsiMask=mask, args=make_args(out_b), classNum=7, downsample=1, glob_offset=offset)

        files_a = sorted(glob.glob(str(tmp_path / 'a' / 'slide_*.json')))
        files_b = sorted(glob.glob(str(tmp_path / 'b' / 'slide_*.json')))

        data_a = json.load(open(files_a[0]))
        data_b = json.load(open(files_b[0]))

        pt_a = data_a['annotation']['elements'][0]['points'][0]
        pt_b = data_b['annotation']['elements'][0]['points'][0]
        assert pt_b[0] == pytest.approx(pt_a[0] + offset[0])
        assert pt_b[1] == pytest.approx(pt_a[1] + offset[1])

    def test_empty_mask_produces_no_files(self, tmp_path):
        args = make_args(str(tmp_path / OUTPUT_PATH))
        mask = np.zeros((200, 200), dtype='uint8')
        xml_suey_local(wsiMask=mask, args=args, classNum=7, downsample=1, glob_offset=[0, 0])
        assert get_output_files(tmp_path) == []
