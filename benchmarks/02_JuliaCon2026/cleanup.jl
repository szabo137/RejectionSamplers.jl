import Pkg

@info "Performing cleanup."

PKGS = ["InteractiveUtils", "CUDA", "oneAPI", "AMDGPU", "Metal"]
for _pkg in PKGS
    _pkg in keys(Pkg.project().dependencies) ? Pkg.rm(_pkg) : nothing
end

Pkg.gc()
