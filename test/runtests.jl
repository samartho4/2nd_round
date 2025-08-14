using Test

# Auto-include all test files except this one
const TEST_DIR = @__DIR__
for (root, _, files) in walkdir(TEST_DIR)
	for f in sort(files)
		if endswith(f, ".jl") && f != "runtests.jl"
			include(joinpath(root, f))
		end
	end
end 