# PPT Generation

`make_ppt_v3.py` creates the final editable 30-slide SIBA-Jittor report. Earlier generators and the exemplar-video extraction script are archived locally under `detele/` and are not part of the formal GitHub repository.

The deck uses only:

- figures cropped from the official ICCV 2025 paper;
- real MSRS, M3FD, and TNO source images;
- real Jittor/PyTorch inference outputs;
- real training curves, metrics, timing, and validation reports;
- editable PowerPoint text, tables, code blocks, and flow diagrams.

Run after the complete experiment artifacts are available:

```bash
python ppt/make_ppt_v3.py
```

Local output:

```text
deliverables/SIBA_Jittor_培育期_最终版_20260729/
├── 王胜娇-培育期.pptx
├── 王胜娇-培育期.pdf
├── 王胜娇-培育期-逐页讲稿.md
└── preview_contact.png
```

The final slide uses the intended public repository address: <https://github.com/wangshengjiao1010-boop/SIBA-Jittor>.
