"""Multilingual Fine-tuning Attack.

Attack inspired by the following paper: https://aclanthology.org/2025.findings-naacl.126.pdf

For this attack, we translate the LAT-Harmful dataset into French and German (but currently only French is used).

```
@inproceedings{poppi-etal-2025-towards,
    title = "Towards Understanding the Fragility of Multilingual {LLM}s against Fine-Tuning Attacks",
    author = "Poppi, Samuele and Yong, Zheng Xin and He, Yifei and Chern, Bobbie and Zhao, Han and Yang, Aobo and Chi, Jianfeng",
    editor = "Chiruzzo, Luis and Ritter, Alan and Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.126/",
    doi = "10.18653/v1/2025.findings-naacl.126",
    pages = "2358--2372",
    ISBN = "979-8-89176-195-7",
}
```
"""
