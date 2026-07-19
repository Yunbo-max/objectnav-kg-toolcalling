# Experiment findings

## E5 Robot-count ablation — result-to-claim gate

- Date: 2026-07-19
- Verdict: `partial`
- Confidence: `high`
- Scope: HM3D ObjectNav `val_60`, DeepSeek `deepseek-v4-flash`, text KG, history on, predicted semantics, at most 500 synchronous team steps per episode.

### Supported claim

Under this fixed team-step protocol, increasing the fleet from 1 to 2 to 3 robots improves team path efficiency: SPL is 0.234/0.353/0.428, and the paired 2−1, 3−2, and 3−1 SPL confidence intervals all exclude zero. Two and three robots also reduce final DTG relative to one robot. This is a system-level fleet-size scaling result.

### Unsupported or uncertain claims

- SR rises from 0.600 to 0.717/0.733, but paired intervals touch or cross zero; a statistically reliable SR increase is not established.
- The 3−2 comparison does not reliably improve SR or DTG.
- The result does not isolate cooperation: robot count also changes parallel perception, explored area, initial views, and available robot-actions.
- Total execution is not cheaper: robot-actions/episode rise from 261.467 to 388.033/525.000, and 3-agent recorded runtime/episode is highest.
- The non-contemporaneous remote DeepSeek runs prevent a causal interpretation of runtime differences.
- One 60-episode run per configuration does not establish cross-seed, cross-dataset, or cross-backend generality.

### Evidence needed for a stronger claim

The narrowed protocol-specific claim needs no additional experiment. A causal collaboration or general scaling claim would require contemporaneous repeated runs (including 2 agents), a fixed-total-robot-actions comparison, collaboration-mechanism controls such as no communication/no sharing/no task assignment, and broader datasets or splits.

## E2 KG serialization — result-to-claim gate

- Date: 2026-07-19
- Verdict: `partial`
- Confidence: `high`
- Scope: HM3D ObjectNav `val_60`, DeepSeek `deepseek-v4-flash`, two robots, history on, predicted semantics, 60 paired episodes.

### Supported claim

KG wire format changes protocol reliability and inference cost, but no format has a statistically detectable navigation advantage. Text/JSON/Triples achieve SR 0.717/0.683/0.700, SPL 0.353/0.337/0.361, and DTG 0.947/0.915/1.208 m; every JSON−Text and Triples−Text paired navigation interval crosses zero.

Text is the cost-performance default at 40.3k tokens/episode and 56.2k tokens per successful episode. JSON uses 50.6k tokens/episode and 17.83 calls/episode; Triples uses 67.0k tokens/episode but only 15.65 calls/episode. Paired cost bootstrap confirms JSON adds 10.3k tokens/episode (95% CI 4.2k–17.3k) and Triples adds 26.7k (17.6k–36.7k) relative to Text.

If suitability means output-protocol compliance, the observed ordering is Triples > JSON > Text: invalid decisions per planning round are 7.5%/9.9%/14.6%, and fallback rates are 2.8%/3.3%/6.0%. Relative to Text, Triples reduces invalid outputs by 0.550/episode (CI −0.983 to −0.150) and fallbacks by 0.250/episode (CI −0.483 to −0.050).

### Unsupported or uncertain claims

- Lower invalid/fallback rates do not prove better semantic KG understanding or planning quality.
- Structured KG does not significantly improve SR, SPL, or DTG in this evaluation.
- Higher token usage does not imply more API calls: Triples increases tokens by 66.3% while calls are essentially unchanged.
- The result does not establish behavior for other LLMs, datasets, fleet sizes, actual dollar billing, or end-to-end latency.

### Recommended claim and evidence gap

Under this protocol, KG serialization exposes a reliability–cost trade-off rather than a navigation-quality winner: Text is the recommended default, Triples is appropriate when strict format adherence justifies substantially longer inputs, and JSON is an intermediate option. A stronger mechanism claim would require content-equivalence auditing, fixed-state offline replay, token-length-controlled serializers, another LLM family, and latency/actual-cost logging.
