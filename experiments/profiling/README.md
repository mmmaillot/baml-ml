# Native ML hotspot analysis (#42)

Source: [scaling CI run 45934846303](https://github.com/mmmaillot/baml-ml/actions/runs/45934846303), [artifact 15169434456](https://github.com/mmmaillot/baml-ml/actions/runs/45934846303/artifacts/15169434456), BAML 0.26.0, commit b9216e141207bc537dae7235e8d991600c5d836e. Verified ZIP SHA256: 1b1ae4e944402f1089a230d208b7ebbdbbd02d72f6c21a80f334fb79cc53bc9b.

Reproduce without running BAML:

```sh
bash experiments/profiling/run.sh native-scaling-evidence.zip /tmp/profiling
```

Requires Bash, unzip, jq. The 52 default-mode profiles (13 workloads x four trials) are validated for schema, status, invocation counts, and nonnegative timings. Output retains all source profiles and per-function median exclusive/inclusive times, calls, pipeline shares and theoretical Amdahl bounds.

## Method and limitations

Use default mode, not full payload recording. Sum self time across calling-context rows by function. Divide by the inclusive time of ClassificationPipelinePredictProba or ClassificationPipelineFit. Warmups are included in both numerator and denominator. Exclude known fixture and harness functions. The old SQL projection lacks parent IDs, so filtering by names is specific to these workloads; the training checksum's few Abs calls can remain in the numerator. Do not sum inclusive times across nested functions. Shares are medians of per-trial shares, not ratios of unrelated medians.

Recorded user-function self times cover 67–78% of inference pipeline time and 87% of fit time. The residual is unattributed by this projection (including potentially native/runtime work); it is not evidence of a particular bottleneck. Amdahl bounds apply only to the profiled pipeline, not the uninstrumented runtime or application wall time. No measurement of FFI or allocations in isolation. Training gradient invocation counts vary with dataset size: 160, 90, 20 across two fits respectively, so dataset scaling is confounded by solver iteration counts.

## Results

Median shares of pipeline inclusive time (%). Groups sum exclusive times of the exact functions listed in analyze.jq; they do not represent complete subsystem inclusive costs.

| Workload | Checks | Preprocess | Logits/dot | Normalization | Attributed total |
|---|---:|---:|---:|---:|---:|
| d4 | 22.6 | 21.0 | 6.8 | 6.3 | 78 |
| d32 | 26.3 | 22.2 | 5.0 | 1.4 | 69 |
| d128 | 26.9 | 22.2 | 4.6 | 0.4 | 67 |
| d512 | 27.2 | 22.2 | 4.5 | 0.1 | 67 |
| b32 | 32.8 | 25.4 | 10.0 | 2.5 | 72 |
| b256 | 33.3 | 25.4 | 10.3 | 2.6 | 73 |
| b1024 | 33.0 | 25.5 | 10.5 | 2.7 | 73 |
| distinct100 | 26.1 | 22.4 | 5.1 | 1.5 | 70 |
| distinct1000 | 26.2 | 22.3 | 4.9 | 1.4 | 70 |
| repeat100 | 26.3 | 22.2 | 5.1 | 1.4 | 70 |
| fit150 | 0.5 | 0.2 | 38.4 | 27.3 | 87 |
| fit1000 | 0.8 | 0.4 | 38.3 | 27.9 | 87 |
| fit10000 | 3.7 | 1.9 | 35.4 | 25.3 | 87 |

### Inference d512

| Function | Self share | Calls (1000 timed + 10 warmup predictions) |
|---|---:|---:|
| IsFinite | 18.19% | 5,177,260 |
| LogRegPredictProba | 12.76% | 1010 |
| ScalerTransform | 12.61% | 1010 |
| ImputerTransform | 9.61% | 1010 |
| CheckFiniteMatrix | 7.07% | 2020 |
| DotUnchecked | 4.44% | 3030 |

At 512 features, eliminating DotUnchecked entirely would give only 1.046x theoretical pipeline speedup; checks as a group 1.374x, or 1.325x if made ten times faster. At batch 1024, checks 1.492x (1.422x at 10x), logits/dot 1.117x (1.104x at 10x).

### Training fit10000

| Function | Self share | Calls (warmup + timed fit) |
|---|---:|---:|
| DotUnchecked | 18.38% | 1,560,000 |
| ParamsRowLogits | 17.00% | 520,000 |
| LogRegGradient | 11.72% | 20 |
| SoftmaxRow | 11.24% | 200,000 |
| LogSumExp | 7.79% | 320,000 |
| Exp | 4.50% | 1,560,000 |

Eliminating only DotUnchecked gives a 1.225x ceiling, logits/dot group 1.548x, or 1.468x at 10x. Normalization group 1.339x, or 1.295x at 10x.

## Decision

Do not start a scalar-dot-only Rust port. For inference, first investigate whether repeated matrix validation can be avoided safely at trusted internal boundaries without changing public error semantics. For training, coarse-grained logits/normalization or loss/gradient kernels are more plausible than crossing a native boundary per dot product. Benchmark one focused candidate against the current implementation and verify parity, trace semantics and actual conversion/dispatch costs before broad NumBam integration. Preserve pure BAML as the control. No optimization or estimator changes in this story.

The analysis was run offline on the checksum-verified existing artifact; no new scaling trials.
