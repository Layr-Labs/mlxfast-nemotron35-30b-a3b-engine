# Local benchmark constants

This note describes legacy constants that remain in local diagnostics. It does
not define ranked scoring. The fixture, manifest, and verified benchmarker
define the ranked run.

## Cached local values

`Sources/MLXFastCore/Constants.swift` contains cached values used by local
estimates and gates-only placeholder fields:

- `officialBaselineDecodeSecondsPerToken = 0.012374741210937498`
- `officialBaselinePrefillSecondsPerToken = 0.0003276219582519531`

These values are inherited local diagnostics. They are not ranked
denominators. The code is the authority if this note differs.

## Legacy acceptance bands

`Sources/MLXFastCore/Score.swift` and `AcceptanceBand` still apply local-only
acceptance bands:

- prefill: ±3%
- decode: +1% regression tolerance and −2.5% gain tolerance

These bands gate local `TimedRunScoreEvaluation` diagnostics. They are not
ranked score limits. Ranked limits come from live paired scoring, calibration
bands, and the fixture floors.

## Ranked scoring

A ranked pair measures its serial control leg and candidate leg on the same
machine in the same job. Live control seconds per token is the numerator in each
gain. Candidate seconds per token is the denominator.

The ranked decode figure includes the seed window:
`(seed window + decode window) / decode token count`. Ranked prefill is measured
separately.

No file stores a ranked baseline pair. Ranked goldens reject
`benchmark.baseline_prefill_seconds_per_token` and
`benchmark.baseline_decode_seconds_per_token`.

The per-machine calibration file is a health band for the control leg. It can
stop a run. It cannot change a score.

## Legacy golden fields

`BenchmarkGolden` can decode these optional fields for local compatibility:

```json
"benchmark": {
  "baseline_prefill_seconds_per_token": 0.0101,
  "baseline_decode_seconds_per_token": 0.1317
}
```

Both fields must be present together, finite, and positive. This behavior is
restricted to local mode. Ranked goldens carrying either field are refused.

For the public ranked window and formula, see
[the participant contract](participant-contract.md#5-scoring).
