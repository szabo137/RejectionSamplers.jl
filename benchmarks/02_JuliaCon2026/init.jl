using Pkg

Pkg.develop(path = "../../")
Pkg.develop(path = joinpath("examples", "TruncatedGaussians"))

Pkg.instantiate()
