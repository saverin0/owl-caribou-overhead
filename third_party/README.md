# third_party

Upstream code vendored so the public checkpoints can be instantiated. Our own
pipeline lives in `src/owlcaribou/`; nothing in this directory is ours except
where explicitly stated below.

Source: the `animaloc/` and `dinov3/` folders of Microsoft's
MegaDetector-Overhead repository
(https://github.com/microsoft/MegaDetector-Overhead) at commit
`7e5e4b9ca79027f071790d7f62978672bee656b0`. Every vendored file except
`animaloc/__init__.py` (ours, see below) is byte for byte identical to that
commit, headers included; `animaloc/LICENSE-MIT.txt` is the repository's root
`LICENSE.txt`, unchanged.

## animaloc/

Model definitions for the OWL-C, OWL-T and OWL-D detectors; Caribou-OWL-C
shares OWL-C's architecture.

- Licence: MIT. `animaloc/LICENSE-MIT.txt` is the distributing repository's
  licence, (C) 2026 Microsoft. The University of Liège notice, (C) 2024
  University of Liège, Gembloux Agro-Bio Tech, Forest Is Life, author
  Alexandre Delplanque (the HerdNet lineage), is in the headers of the eight
  HerdNet files (`models/__init__.py`, `faster_rcnn.py`, `herdnet.py`,
  `register.py`, `ss_dla.py`, `utils.py`, `utils/__init__.py`,
  `utils/registry.py`). `owl_c.py`, `owl_t.py` and `owl_d.py` carry no
  copyright header upstream; each names its HerdNet / Université de Liège
  origin in a docstring. `models/dla.py` carries its own MIT notice, (c) 2019
  Xingyi Zhou.
- Vendored: `models/` (all 10 files) and `utils/{__init__,registry}.py`.
- **One file is ours, not upstream's:** `animaloc/__init__.py`. Upstream's
  version imports the full package tree (data, datasets, eval, train, vizual),
  none of which is vendored. Ours imports nothing.

Note: `models/owl_d.py` hardcodes a fallback backbone path belonging to an
upstream developer's machine. We never rely on it; `dinov3_root` is always
passed explicitly by our code. The line is left untouched so this copy stays
verifiably identical to upstream.

## dinov3/

Backbone architecture required by OWL-D only. It builds the ViT-H+/16 graph.
No DINOv3 weights are downloaded separately: the public OWL-D checkpoint
(Zenodo record 20802844) already contains the DINOv3 encoder weights, loaded
frozen as in the released configuration. Those weights are Meta's and stay
under the DINOv3 License for anyone who runs `owl-d`.

- Licence: DINOv3 License (Meta Platforms). See `dinov3/LICENSE.md`. Section
  1.a permits redistribution; section 1.b.i requires this copy of the licence
  to accompany the code. It is not MIT and restricts some uses.
- Origin: MegaDetector-Overhead's vendored `dinov3/` folder, originally from
  https://github.com/facebookresearch/dinov3.
- Vendored subset: `hub/{__init__,backbones,utils}`, `layers/`, `models/` and
  `utils/` (the import closure that builds the backbone) plus `checkpointer/`,
  `fsdp/`, `distributed/` and `logging/`, kept whole for completeness; OWL-D
  inference imports none of those four (`checkpointer/` and `fsdp/` are
  referenced only inside `models/__init__.py:build_model_for_eval`, which the
  hub path never calls). Everything else in the package (training, evaluation,
  data, configs, losses) is omitted.
- `hub/{classifiers,depthers,detectors,segmentors,dinotxt}.py` are omitted;
  `hub/__init__.py` is empty upstream, so they are never imported.
