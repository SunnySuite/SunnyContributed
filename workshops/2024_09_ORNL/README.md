## 2024 Sunny ORNL workshop

![image](QR.png)

### 1. Please install Julia ≥ 1.10 + Sunny ≥ 0.7.1

[Instructions](https://github.com/SunnySuite/Sunny.jl/wiki/Getting-started-with-Julia)

### 2. Check that a basic script runs

```julia
using GLMakie, Sunny
cryst = Sunny.kagome_crystal()
view_crystal(cryst; ndims=2)
```

If you have a problem, run `update` from the [Julia package manager](https://github.com/SunnySuite/Sunny.jl/wiki/Getting-started-with-Julia#the-built-in-julia-package-manager).

### 3. Download course materials

[Download link](https://download-directory.github.io/?url=https%3A%2F%2Fgithub.com%2FSunnySuite%2FSunnyContributed%2Ftree%2Fmain%2Fworkshops%2F2024_09_ORNL)
