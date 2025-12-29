# Third-Party Licenses and Acknowledgments

This project builds upon several prior works in dialogue generation and evaluation. We gratefully acknowledge the following projects and their contributors.

---

## Baseline Methods

### GenKS (Generative Knowledge Selection)

**Paper**: "Generative Knowledge Selection for Knowledge-Grounded Dialogues"
**Authors**: Weiwei Sun, Pengjie Ren, and Zhaochun Ren
**Repository**: https://github.com/sunnweiwei/GenKS
**ArXiv**: https://arxiv.org/abs/2304.04836

**License**: No explicit license provided. This is academic research code shared publicly for reproducibility.

**Usage in this project**:
- Inference code adapted and extended for our proposed methods
- Pre-trained model checkpoints for baseline comparisons
- Data processing pipelines

**Citation**:
```bibtex
@inproceedings{Sun2023GenerativeKS,
  title={Generative Knowledge Selection for Knowledge-Grounded Dialogues},
  author={Weiwei Sun and Pengjie Ren and Zhaochun Ren},
  year={2023}
}
```

---

### SPI (Sequential Posterior Inference)

**Paper**: "Diverse and Faithful Knowledge-Grounded Dialogue Generation via Sequential Posterior Inference"
**Authors**: Yan Xu, Deqian Kong, Dehong Xu, Ziwei Ji, Bo Pang, Pascale Fung, and Ying Nian Wu
**Conference**: ICML 2023
**Repository**: https://github.com/deqiankong/SPI
**Paper PDF**: https://arxiv.org/pdf/2306.01153.pdf
**Proceedings**: https://proceedings.mlr.press/v202/xu23j.html

**License**: No explicit license provided. This is academic research code shared publicly for reproducibility.

**Usage in this project**:
- Inference code adapted and extended for our proposed methods
- Pre-trained model checkpoints for baseline comparisons
- Data processing pipelines

**Citation**:
```bibtex
@inproceedings{pmlr-v202-xu23j,
  author = {Xu, Yan and Kong, Deqian and Xu, Dehong and Ji, Ziwei and Pang, Bo and Fung, Pascale and Wu, Ying Nian},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {38518--38534},
  publisher = {PMLR},
  series = {Proceedings of Machine Learning Research},
  title = {Diverse and Faithful Knowledge-Grounded Dialogue Generation via Sequential Posterior Inference},
  volume = {202},
  year = {2023}
}
```

---

## Evaluation Methods

### G-Eval

**Repository**: https://github.com/nlpyang/geval
**Copyright**: (c) 2024 Yang Liu
**License**: MIT License

**Usage in this project**:
- Evaluation prompts for coherence, fluency, informativeness, and interestingness
- Evaluation methodology and framework
- Code structure adapted to modern OpenAI API

**MIT License Text**:
```
MIT License

Copyright (c) 2024 Yang Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

### MEEP (Engagingness Evaluation)

**Repository**: https://github.com/PortNLP/MEEP
**Copyright**: (c) 2023 PortNLP
**License**: MIT License

**Usage in this project**:
- Evaluation prompt for engagingness measurement
- Evaluation methodology for dialogue quality assessment
- Code structure adapted to modern OpenAI API

**MIT License Text**:
```
MIT License

Copyright (c) 2023 PortNLP

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Notes on Usage

### Academic Research Code (GenKS, SPI)

While these repositories do not include explicit open-source licenses, they are:
1. Published academic research code intended for reproducibility
2. Accompanied by peer-reviewed publications
3. Publicly available on GitHub
4. Include citation requests in their READMEs

Our usage follows academic norms:
- Proper attribution through citations
- Use for research and comparison purposes
- Code adapted and extended rather than directly copied
- Results compared transparently in publications

If you are a rights holder and have concerns about this usage, please contact us.

### MIT Licensed Code (G-Eval, MEEP)

We comply with MIT License terms by:
1. Including the full license text and copyright notices
2. Clearly attributing the original authors
3. Linking to original repositories
4. Documenting our adaptations and modifications

---

## Modifications Made

All code in this repository has been:
- **Adapted** from the original implementations
- **Updated** to use modern API versions (e.g., OpenAI API v1.x)
- **Extended** with additional functionality for our proposed methods
- **Documented** with type hints and detailed docstrings
- **Integrated** into a unified evaluation framework

We have made significant modifications while respecting the original work and maintaining proper attribution.

---

*Last updated: 2025-12-29*
