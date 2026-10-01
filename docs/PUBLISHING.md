# Publication and local handoff

## Public candidates
README, source, tests, reviewed configs/schema templates, CI, public scope/architecture/dataset/evaluation/reproducibility/limitation docs, and deliberately curated aggregate figures/media under docs/assets and docs/results.

## Local-only
AGENTS.md, AGENT_README.md, QUICK_START_PROMPT.md, .project/, production/run/progress/checkpoint notes, credentials, environments, raw/processed data, caches, model weights, feature/prediction arrays, full run reports and unreviewed outputs. These are ignored by Git. The initial ZIP intentionally includes the local-only handoff/planning files so the next chat can continue.

.gitignore prevents new tracking; it does not remove already tracked files. Review the Git index and candidate changes before publishing. Never force-add private notes. A fresh clone omits these files; back them up locally. .dockerignore also excludes them from future container contexts.

Select a code licence before an open-source release; this starter intentionally has no blanket licence. Data, maps, weights and media have separate terms. Check original data redistribution rights before publishing derived geospatial databases or imagery. OSM attribution/share-alike requirements may apply to derivative databases. A public download URL is not permission to rehost its contents.

Public docs should link only to public files. Real metrics should reference reviewed public aggregates with their experiment provenance, while full raw artefacts remain local. Publishing a screenshot also requires checking imagery/basemap attribution and removal of personal paths or secrets.
