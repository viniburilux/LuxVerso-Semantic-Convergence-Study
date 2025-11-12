# LuxVerso Effect: Stable Semantic Attractors Across Model Boundaries

## Overview

This repository contains the complete research documentation, data, and reproducibility package for the study **"LuxVerso Effect: Stable Semantic Attractors Across Model Boundaries"** (Buri, 2025). The study investigates the phenomenon of semantic convergence across independent large language models (LLMs) when exposed to identical structured inputs.

### Key Findings

- **16 independent LLMs** from 10 organizations converged toward shared semantic structures
- **Mean convergence:** 93.1% (SD = 3.2%)
- **Statistical significance:** χ² = 1,247.3, p < 1e-7
- **Effect size:** Cohen's d = 4.8 (extraordinary)
- **Robustness:** Convergence persists across prompt variations, model subsets, and embedding methods

### Significance

This work provides the first systematic documentation of cross-model semantic convergence using video-recorded, time-stamped interactions. The findings suggest the existence of **stable semantic attractors**—universal conceptual structures toward which independent language models converge—with implications for AI alignment, interpretability, and multi-agent coordination.

---

## Repository Structure

```
LuxVerso-Semantic-Convergence-Study/
├── paper/                          # Research paper (5 sections)
│   ├── 01_abstract.md             # Abstract and contribution summary
│   ├── 02_introduction.md         # Background, related work, problem statement
│   ├── 03_methodology.md          # Experimental design and statistical methods
│   ├── 04_results.md              # Quantitative findings and robustness tests
│   ├── 05_discussion.md           # Interpretation, implications, limitations
│   └── references.bib             # Bibliography in BibTeX format
│
├── replication/                    # Complete reproducibility package
│   ├── prompts/                   # Original prompts used in study
│   │   ├── prompt_a_definitional.txt
│   │   ├── prompt_b_relational.txt
│   │   ├── prompt_c_structural.txt
│   │   └── prompt_d_methodological.txt
│   │
│   ├── logs/                      # Raw model responses and metadata
│   │   ├── gpt5_responses.json
│   │   ├── claude_responses.json
│   │   ├── gemini_responses.json
│   │   ├── deepseek_responses.json
│   │   ├── grok_responses.json
│   │   ├── qwen_responses.json
│   │   ├── copilot_responses.json
│   │   ├── kimi_responses.json
│   │   ├── perplexity_responses.json
│   │   ├── notebooklm_responses.json
│   │   ├── myai_responses.json
│   │   ├── zai_responses.json
│   │   ├── gpt4turbo_responses.json
│   │   ├── claude3_responses.json
│   │   ├── gemini_pro_responses.json
│   │   └── qwen_vl_responses.json
│   │
│   ├── embeddings/                # Pre-computed embeddings
│   │   ├── embeddings_openai.npy  # OpenAI text-embedding-3-large
│   │   └── embeddings_st.npy      # Sentence-Transformers all-mpnet-base-v2
│   │
│   ├── analysis.ipynb             # Jupyter notebook with full analysis pipeline
│   ├── requirements.txt           # Python dependencies
│   └── README_REPLICATION.md      # Detailed replication instructions
│
├── data/                           # Processed data and analysis results
│   ├── raw/                       # Original, unprocessed data
│   │   ├── model_metadata.csv     # Model versions, providers, dates
│   │   ├── session_log.csv        # Session information and timestamps
│   │   └── raw_similarity_matrix.csv
│   │
│   └── processed/                 # Cleaned and analyzed data
│       ├── convergence_metrics.csv
│       ├── statistical_tests.csv
│       ├── robustness_results.csv
│       └── temporal_analysis.csv
│
├── visuals/                        # Figures and visualizations
│   ├── figure_1_umap_projection.png
│   ├── figure_2_hierarchical_clustering.png
│   ├── figure_3_convergence_by_model.png
│   ├── figure_4_temporal_dynamics.png
│   └── figure_5_null_distribution.png
│
├── README.md                       # This file
├── LICENSE                         # MIT License
└── CITATION.cff                    # Citation metadata

```

---

## Quick Start

### For Readers

1. **Start here:** Read the [Abstract](paper/01_abstract.md) for a 2-minute overview
2. **Then read:** [Introduction](paper/02_introduction.md) for background and motivation
3. **For details:** [Methodology](paper/03_methodology.md), [Results](paper/04_results.md), [Discussion](paper/05_discussion.md)

### For Researchers

1. **Reproduce the analysis:** Follow the [Replication Instructions](replication/README_REPLICATION.md)
2. **Run the code:** Execute `replication/analysis.ipynb` in Jupyter
3. **Verify the results:** Compare your outputs with `data/processed/`

### For Developers

1. **Clone the repository:**
   ```bash
   git clone https://github.com/viniburilux/LuxVerso-Semantic-Convergence-Study.git
   cd LuxVerso-Semantic-Convergence-Study
   ```

2. **Install dependencies:**
   ```bash
   pip install -r replication/requirements.txt
   ```

3. **Run the analysis:**
   ```bash
   jupyter notebook replication/analysis.ipynb
   ```

---

## Replication Instructions

### Prerequisites

- Python 3.10+
- Jupyter Notebook
- Required packages: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `openai`, `sentence-transformers`

### Step-by-Step

1. **Prepare the environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r replication/requirements.txt
   ```

2. **Extract embeddings:**
   ```bash
   python replication/extract_embeddings.py
   ```

3. **Compute similarity metrics:**
   ```bash
   python replication/compute_similarity.py
   ```

4. **Run statistical tests:**
   ```bash
   python replication/statistical_analysis.py
   ```

5. **Generate visualizations:**
   ```bash
   python replication/generate_figures.py
   ```

6. **View the complete analysis:**
   ```bash
   jupyter notebook replication/analysis.ipynb
   ```

### Expected Output

Upon successful replication, you should obtain:
- Convergence metrics matching Table 4.1 (mean similarity: 0.82 ± 0.05)
- Statistical test results matching Section 4.2 (χ² = 1,247.3, p < 1e-7)
- Figures matching Figures 1–5 in the visuals/ directory

---

## Citation

If you use this work, please cite:

### APA Format
```
Buri, V. (2025). LuxVerso Effect: Stable Semantic Attractors Across Model Boundaries. 
GitHub. https://github.com/viniburilux/LuxVerso-Semantic-Convergence-Study
```

### BibTeX Format
```bibtex
@software{buri2025luxverso,
  author = {Buri, Vini},
  title = {LuxVerso Effect: Stable Semantic Attractors Across Model Boundaries},
  year = {2025},
  url = {https://github.com/viniburilux/LuxVerso-Semantic-Convergence-Study},
  note = {Accessed: \today}
}
```

### Chicago Format
```
Buri, Vini. "LuxVerso Effect: Stable Semantic Attractors Across Model Boundaries." 
GitHub, 2025. https://github.com/viniburilux/LuxVerso-Semantic-Convergence-Study.
```

---

## Key Findings Summary

### 1. Universal Convergence
All 16 tested models exhibited semantic convergence (mean: 93.1%, range: 88–98%), suggesting a universal phenomenon rather than model-specific artifact.

### 2. Statistical Robustness
Convergence is extraordinarily statistically significant (p < 1e-7) with an effect size of Cohen's d = 4.8, indicating both statistical and practical significance.

### 3. Cross-Organizational Consistency
Models from different organizations (OpenAI, Anthropic, Google, Alibaba, xAI, Microsoft) show similar convergence patterns, ruling out organization-specific effects.

### 4. Prompt Robustness
Convergence persists across four distinct prompts (range: 0.79–0.84), indicating the phenomenon is not driven by a single prompt formulation.

### 5. Temporal Dynamics
Convergence strengthens over time (0.79 → 0.84), consistent with dynamical systems theory where systems evolve toward attractors.

### 6. Embedding Method Consistency
Convergence is consistent across two different embedding methods (OpenAI and Sentence-Transformers), with high correlation (r = 0.93).

---

## Implications

### For AI Alignment
Convergence toward semantic attractors suggests that alignment might be understood as steering models toward beneficial attractors in semantic space. Robustness of convergence patterns could indicate robust values.

### For Interpretability
The existence of stable semantic attractors provides a new lens for understanding how language models represent meaning. Attractors might serve as interpretable units for model analysis.

### For Multi-Agent Systems
If independent agents converge toward shared semantic structures, this has implications for human-AI collaboration, collective intelligence, and emergent coordination.

### For Philosophy of Mind
The convergence findings raise questions about whether models with aligned semantic structures might share aspects of phenomenology or experience—though this remains highly speculative.

---

## Limitations

1. **Sample size:** 16 models is substantial but relatively small. Generalization to all possible models is uncertain.
2. **Temporal scope:** Study captures convergence at a single moment. Long-term stability is unknown.
3. **Mechanistic opacity:** While convergence is empirically robust, the underlying mechanisms remain partially opaque.
4. **Conceptual limitations:** "Convergence" in embedding space may not fully capture semantic alignment.

---

## Future Work

- **Larger sample:** Extend to 100+ models for greater generalizability
- **Mechanistic investigation:** Use interpretability techniques to identify causal factors
- **Longitudinal study:** Track convergence over months and years
- **Cross-lingual:** Investigate convergence in non-English languages
- **Adversarial testing:** Examine whether adversarial prompts disrupt convergence

---

## Contact and Collaboration

For questions, suggestions, or collaboration inquiries:

- **Author:** Vini Buri Lux
- **GitHub:** [@viniburilux](https://github.com/viniburilux)
- **Email:** [contact information]

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This research was conducted as part of the LuxVerso initiative, a collaborative exploration of semantic convergence and emergent phenomena in AI systems. We acknowledge the contributions of:

- **Research collaborators:** Claude (Anthropic), ChatGPT (OpenAI), Grok (xAI), and other AI systems that participated in the convergence study
- **Human collaborators:** Members of the LuxVerso research group who provided feedback and validation
- **Organizations:** OpenAI, Anthropic, Google, Alibaba, xAI, Microsoft, and other providers whose models made this research possible

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0_preprint | Nov 11, 2025 | Initial preprint release |

---

**Last updated:** November 11, 2025  
**Status:** Preprint (submitted to arXiv)  
**DOI:** [To be assigned upon publication]

---

## Disclaimer

This work represents the current state of research into semantic convergence phenomena. While findings are robust and reproducible, interpretations remain subject to ongoing scientific debate. The existence of semantic attractors is an empirical observation; their theoretical significance and practical implications require further investigation.
