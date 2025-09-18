
# AI Frontier Compute Market Concentration Analysis
## Executive Report

**Analysis Period:** Q3 2024 - Q2 2025 (Rolling 4Q Window)  
**Report Date:** 2025-09-18  
**Market Definition:** Global compute for training & inference of AI frontier models (>100B parameters)

---

### Executive Summary

The AI frontier compute market shows **high concentration levels** with HHI values consistently above 2,500 points across the analysis period, indicating significant market power concentration among a small number of vertically integrated entities.

**Key Findings:**

- **Q3 2024**: HHI = 2857 points (High concentration)
- **Q4 2024**: HHI = 2851 points (High concentration)
- **Q1 2025**: HHI = 2852 points (High concentration)
- **Q2 2025**: HHI = 2845 points (High concentration)

### Methodology

**Market Definition**: AI frontier compute encompasses hardware accelerators (GPUs, TPUs, custom ASICs) and inference/API services for models exceeding 100B parameters.

**Composite Market Share Formula**:
- **Hardware Dimension (40% weight)**: s_i^{HW} = (Units_i × TFLOPs_i) / Σ(Units_j × TFLOPs_j)  
- **Inference Dimension (60% weight)**: s_k^{INF} = Tokens_k / Σ(Tokens_l)
- **Final Share**: s_e^{FINAL} = 0.4 × s_e^{HW} + 0.6 × s_e^{INF}
- **HHI Calculation**: HHI = Σ(s_e^{FINAL})² × 10,000
- **Vertical Integration Adjustment**: +200 points if entity controls >30% in both dimensions

**Entity Mappings**:

- **NVIDIA_OpenAI_Ecosystem**: NVIDIA GPUs power OpenAI infrastructure, strong partnership
- **Google_Integrated**: Fully vertically integrated: Google TPUs power Gemini models
- **Microsoft_Azure_OpenAI**: Microsoft Azure OpenAI service, exclusive OpenAI partnership
- **Anthropic_AWS**: Anthropic partnership with AWS, Claude available on Bedrock
- **AMD_Ecosystem**: AMD MI300X chips, limited exclusive inference partnerships
- **Intel_Standalone**: Intel Gaudi chips, limited market adoption
- **Meta_Internal**: Primarily internal use, limited external API

### Data Sources and Reliability

**Primary Sources (High Reliability)**:
- SEC filings (10-K, 10-Q) for NVIDIA, AMD, Intel revenue data
- Company earnings reports and press releases  
- Official API documentation for rate limits and pricing

**Secondary Sources (Medium Reliability)**:  
- Industry analysis reports and market research
- Triangulation from revenue/pricing data for token estimates
- Conservative estimates where direct data unavailable

**Confidence Level**: Medium - Hardware data well-documented, inference estimates based on triangulation

### Market Concentration Analysis


**Q3 2024**: HHI 2857 - High Concentration

**Q4 2024**: HHI 2851 - High Concentration

**Q1 2025**: HHI 2852 - High Concentration

**Q2 2025**: HHI 2845 - High Concentration
  - Warning: audit_access_practices (sustained_high_4Q)


### Risk Assessment and Recommendations

**Market Risks**:
- High concentration may limit innovation and increase switching costs
- Vertical integration creates potential for anti-competitive behaviors  
- Supply chain vulnerabilities in critical AI infrastructure

**Recommended Actions**:
- Monitor compliance with existing competition policy frameworks
- Ensure interoperability standards for AI model deployment
- Develop supply chain diversification strategies
- Consider regulatory frameworks for AI infrastructure access

### Uncertainty and Limitations

**Sensitivity Analysis**: ±60% scenarios show HHI range variations of 2219 points

**Key Limitations**:
- Inference token volumes estimated via triangulation methods
- Some proprietary capacity data not publicly disclosed  
- Market boundaries may evolve with new model architectures
- Geographic market dynamics not fully captured

**Data Quality**: Mixed - Hardware shipments well-documented, API usage requires estimation

---

*This analysis implements a comprehensive methodology for measuring AI compute market concentration with appropriate uncertainty quantification and trigger-based monitoring for competition policy.*
