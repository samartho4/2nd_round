.PHONY: setup data tune train eval stats baselines dataset bench results figs verify repro test determinism

setup:
	bin/mg setup

data:
	bin/mg data

tune:
	bin/mg tune

train:
	bin/mg train

eval:
	bin/mg eval

stats:
	bin/mg stats

baselines:
	bin/mg baselines

dataset:
	bin/mg dataset

bench:
	bin/mg bench

results:
	bin/mg results

figs:
	bin/mg figs

verify:
	bin/mg verify

repro:
	bin/mg repro

determinism:
	bin/mg determinism

test:
	julia --project=. -e 'using Pkg; Pkg.test()' 