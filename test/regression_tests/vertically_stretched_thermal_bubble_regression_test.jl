function run_vertically_stretched_thermal_bubble_regression_test(arch)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    zF = range(-Lz, 0, length=Nz+1)
    grid = VerticallyStretchedCartesianGrid(size=(Nx, Ny, Nz), length=(Lx, Ly, Lz), zF=zF)

    closure = ConstantIsotropicDiffusivity(ν=4e-2, κ=4e-2)
    model = Model(architecture=arch, grid=grid, closure=closure, coriolis=FPlane(f=1e-4))

    model.tracers.T.data.parent .= 9.85
    model.tracers.S.data.parent .= 35.0

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    regression_data_filepath = joinpath(dirname(@__FILE__), "data", "thermal_bubble_regression.nc")

    ####
    #### Regression test
    ####

    time_step!(model, 10, Δt)

    ds = Dataset(regression_data_filepath, "r")

    uᶜ = ds["u"][:, :, :, end]
    vᶜ = ds["v"][:, :, :, end]
    wᶜ = ds["w"][:, :, :, end]
    Tᶜ = ds["T"][:, :, :, end]
    Sᶜ = ds["S"][:, :, :, end]

    field_names = ["u", "v", "w", "T", "S"]
    fields = [interior(model.velocities.u), interior(model.velocities.v), interior(model.velocities.w),
              interior(model.tracers.T), interior(model.tracers.S)]
    fields_correct = [uᶜ, vᶜ, wᶜ, Tᶜ, Sᶜ]
    summarize_regression_test(field_names, fields, fields_correct)

    # Now test that the model state matches the regression output.
    @test all(Array(interior(model.velocities.u)) .≈ uᶜ)
    @test all(Array(interior(model.velocities.v)) .≈ vᶜ)
    @test all(Array(interior(model.velocities.w)) .≈ wᶜ)
    @test all(Array(interior(model.tracers.T))    .≈ Tᶜ)
    @test all(Array(interior(model.tracers.S))    .≈ Sᶜ)

    return nothing
end
