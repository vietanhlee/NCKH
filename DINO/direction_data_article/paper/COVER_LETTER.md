# COVER LETTER FOR SUBMISSION TO DATA IN BRIEF

**Date:** October 18, 2026  

**To:**  
The Editors-in-Chief and Editorial Board  
*Data in Brief*  
Elsevier BV  

**Article Title:**  
*IC4SD-TrafficSnap: A City-Scale Multi-Camera Image Time-Series and Geospatial Road Network Dataset for Heterogeneous Urban Traffic*

**Authors:**  
Viet-Anh Le$^1$ and Khanh Nguyen-Trong$^{1,*}$  
$^1$ Intelligent Computing for Sustainable Development Laboratory (IC4SD),  
Posts and Telecommunications Institute of Technology (PTIT), Hanoi, Vietnam.  
$^*$ Corresponding Author: khanhnt@ptit.edu.vn  

**Permanent Repository Deposition:**  
- **Zenodo DOI (Release v2.0):** [https://doi.org/10.5281/zenodo.22929940](https://doi.org/10.5281/zenodo.22929940)  
- **Companion Pilot Benchmark DOI (v1.0):** [https://doi.org/10.5281/zenodo.22089673](https://doi.org/10.5281/zenodo.22089673)  
- **Companion Research Preprint:** arXiv:2602.04159  

---

Dear Editors-in-Chief and Editorial Board,

We are pleased to submit our original data article entitled **"IC4SD-TrafficSnap: A City-Scale Multi-Camera Image Time-Series and Geospatial Road Network Dataset for Heterogeneous Urban Traffic"** for publication in *Data in Brief* as a co-submission / standalone data article.

### 1. Summary and Significance of the Dataset
Transportation machine learning benchmarks have predominantly relied on either 1D scalar inductive loop speed sensors (e.g., METR-LA, PEMS-BAY) or short, isolated camera clips (e.g., UA-DETRAC, CityFlow) deployed in car-dominated Western cities. **IC4SD-TrafficSnap** fills a critical empirical void by releasing the first city-scale multimodal benchmark coupling continuous visual surveillance time-series with an OpenStreetMap-derived directed spatial road network in a Southeast Asian megacity (Ho Chi Minh City, Vietnam), where motorcycles account for 70–80% of vehicular flow:
- **714,123 continuous JPEG snapshots** ($512 \times 288$ native resolution, $44.38\text{ GiB} / 47.66\text{ GB}$) captured across **608 active camera stations** over **93.6 continuous hours** (October 2–6, 2026).
- **2,450 directed arterial corridor links** within a 6.0 km OSRM shortest driving distance search horizon, preserving real-world physical network asymmetry (1,070 unidirectional links and 238 distance-asymmetric bidirectional pairs).
- **Production-ready PyTorch loaders, samplers, and dual-transition directed diffusion operators** ($P_f, P_b$ and Chebyshev Laplacian $\tilde{L}$) ready for immediate self-supervised visual representation pre-training (ViT, DINO, MAE) and spatio-temporal graph learning.

---

### 2. Detailed Rationale for the Three-Tier Licensing Framework and CC BY-NC 4.0
We recognize that *Data in Brief* strongly advocates for open-access dissemination. To maximize openness while strictly upholding public civic ethics and data protection laws, the dataset is released under an intentional **three-tier licensing structure**:

1. **Visual Snapshot Corpus ($714,123$ frames): *Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)***
   - **Public Civic Purpose of the Stream Source:** The surveillance feeds were recorded from the open public municipal traffic monitoring camera infrastructure operated under the Intelligent Transportation System (ITS) of the Ho Chi Minh City Department of Transportation ([https://giaothong.hochiminhcity.gov.vn](https://giaothong.hochiminhcity.gov.vn)). Under Vietnamese Decree No. 47/2020/ND-CP on digital data sharing by government agencies, this stream service was established to provide public civic traffic information, congestion monitoring, and flood tracking for citizens.
   - **Ethical Stewardship and Preventing Commercial Exploitation:** Applying the NonCommercial (NC) clause is a deliberate ethical safeguard. Because these camera streams are taxpayer-funded municipal assets intended for public safety and civic awareness, it is ethically imperative to prevent private commercial corporations from appropriating and commercializing public infrastructure feeds for commercial profit.
   - **Completely Unrestricted Academic and Scientific Openness:** The CC BY-NC 4.0 license grants fully unrestricted, free access to all academic researchers, universities, non-profit institutions, and students worldwide. Downstream researchers are fully permitted to download, reproduce, extract visual representations, benchmark machine learning models, and publish findings without any barriers.
   - **An Additional Ethical Safety Layer for Privacy:** While our empirical audit established statistical sub-Nyquist anonymity ($p \le 0.30\%$), the CC BY-NC 4.0 license provides an extra regulatory layer ensuring the data cannot be repurposed into commercial surveillance or monetization products.

2. **Derived Road Network Graph and Routing Matrices: *Open Database License 1.0 (ODbL 1.0)***
   - All spatial network coordinates, shortest driving distances, and corridor topologies were extracted via OSRM from OpenStreetMap data. Releasing this component under ODbL 1.0 fully complies with and honors OpenStreetMap upstream contributor licensing.

3. **Software Toolchains, Loaders, and Graph Utilities: *MIT License***
   - All accompanying Python loading scripts, grouped samplers, and graph operator derivation functions under `code/` are released under the highly permissive MIT License to foster seamless integration into open-source deep learning codebases.

---

### 3. Visual Privacy-by-Design and Empirical Multi-Tier Audit
In strict compliance with personal data protection regulations (Decree No. 13/2023/ND-CP and Law No. 91/2025/QH15), privacy is guaranteed through physical optical constraints and verified by multi-tiered empirical auditing:
- **Sub-Nyquist Optical Geometry:** Cameras are permanently elevated on gantries ($H \ge 6.0$ m) at downward oblique angles ($15^\circ - 40^\circ$). At native $512 \times 288$ broadcast resolution, vehicle license plates subtend character stroke heights of at most $2.5 - 2.8$ pixels, far below optical character recognition thresholds ($\ge 16 - 20$ pixels).
- **Physical Biometric Occlusion:** Mandatory motorcycle helmets (Decree 100/2019/ND-CP) and high-density fabric face masks ($>85\%$) physically occlude facial biometrics.
- **Statistical Rule of Three:** Across $3,000$ audited daylight snapshots (including a 1,000-frame worst-case stress sample evaluated by three calibrated independent human raters with $100\%$ unanimous agreement), zero readable license plates or facial features were detected. By the Hanley and Lippman-Hand Rule of Three, the $95\%$ upper confidence bound on residual privacy breach probability is rigorously bounded at $p \le 0.30\%$.
- **Enforceable Revocation Protocol:** A transparent three-tier takedown mechanism is established via a dedicated public revocation registry (`REVOKED_STATIONS.csv`), an automated local redaction script (`code/redact_revoked_stations.py`), and minor Zenodo version deprecation.

---

### 4. Relationship to Pilot Dataset and Version Lineage
To ensure complete transparency regarding dataset lineage:
- **Pilot Benchmark (v1.0):** The initiative began with an initial pilot study (*IC4SD-Traffic-HCM*, v1.0, Zenodo DOI: [10.5281/zenodo.22089673](https://doi.org/10.5281/zenodo.22089673)), comprising 5,012 annotated frames and longitudinal 1-minute vehicle flow series across 86 days.
- **Current City-Scale Release (v2.0):** **IC4SD-TrafficSnap** represents a massive $140\times$ expansion to **714,123 continuous snapshots** over 93.6 hours, paired with a rigorously pruned 2,450-edge directed road network graph, dual diffusion operators, and comprehensive data dictionaries. Both the companion research preprint (arXiv:2602.04159) and the pilot Zenodo record are explicitly cross-referenced in the article.

We confirm that this manuscript represents original data documentation, adheres strictly to the *Data in Brief* template, and has not been published elsewhere. We look forward to your favorable consideration.

Sincerely,

**Dr. Khanh Nguyen-Trong** (Corresponding Author)  
Intelligent Computing for Sustainable Development Laboratory (IC4SD)  
Posts and Telecommunications Institute of Technology (PTIT)  
Hanoi, Vietnam  
Email: khanhnt@ptit.edu.vn  

**Viet-Anh Le** (First Author)  
Email: anhlv.b23kh002@stu.ptit.edu.vn  
