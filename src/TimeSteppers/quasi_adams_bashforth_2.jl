using MPI

"""
    QuasiAdamsBashforth2TimeStepper{T, TG} <: AbstractTimeStepper

Holds tendency fields and the parameter `χ` for a modified second-order
Adams-Bashforth timestepping method.
"""
struct QuasiAdamsBashforth2TimeStepper{T, TG} <: AbstractTimeStepper
     χ :: T
    Gⁿ :: TG
    G⁻ :: TG
end

"""
    QuasiAdamsBashforth2TimeStepper(arch, grid, tracers, χ=0.1;
                                    Gⁿ = TendencyFields(arch, grid, tracers),
                                    G⁻ = TendencyFields(arch, grid, tracers))

Return an QuasiAdamsBashforth2TimeStepper object with tendency fields on `arch` and
`grid` with AB2 parameter `χ`. The tendency fields can be specified via optional
kwargs.
"""
function QuasiAdamsBashforth2TimeStepper(arch, grid, tracers, χ=0.1;
                                         Gⁿ = TendencyFields(arch, grid, tracers),
                                         G⁻ = TendencyFields(arch, grid, tracers))

    return QuasiAdamsBashforth2TimeStepper{eltype(grid), typeof(Gⁿ)}(χ, Gⁿ, G⁻)
end

#####
##### Time steppping
#####

"""
    time_step!(model::AbstractModel{<:QuasiAdamsBashforth2TimeStepper}, Δt; euler=false)

Step forward `model` one time step `Δt` with a 2nd-order Adams-Bashforth method and
pressure-correction substep. Setting `euler=true` will take a forward Euler time step.
"""
function time_step!(model::AbstractModel{<:QuasiAdamsBashforth2TimeStepper}, Δt; euler=false)
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"
    
    χ = ifelse(euler, convert(eltype(model.grid), -0.5), model.timestepper.χ)

    # Be paranoid and update state at iteration 0, in case run! is not used:
    model.clock.iteration == 0 && update_state!(model)

    calculate_tendencies!(model)

    ab2_step!(model, Δt, χ) # full step for tracers, fractional step for velocities.

    calculate_pressure_correction!(model, Δt)
    pressure_correct_velocities!(model, Δt)

    tick!(model.clock, Δt)
    update_state!(model)
    store_tendencies!(model)

    return nothing
end

#####
##### Time stepping in each step
#####

function ab2_step!(model, Δt, χ)

    # Split up work on grid according to MPI rank
    workgroup, worksize = work_layout(model.grid, :xyz; use_MPI=true)

    Nx, Ny, Nz = size(model.grid)

    # Determine number of planes each MPI rank works on, and the offset of its first plane
    comm = MPI.COMM_WORLD
    n_MPI_ranks = MPI.Comm_size(comm)
    Nz_per_rank = Nz ÷ n_MPI_ranks # work_layout checks that this is evenly divisible
    z_offset = MPI.Comm_rank(comm)*Nz_per_rank

    barrier = Event(device(model.architecture))

    step_field_kernel! = ab2_step_field!(device(model.architecture), workgroup, worksize)

    model_fields = fields(model)

    events = []

    for (i, field) in enumerate(model_fields)
        field_event = step_field_kernel!(field, Δt, χ,
                                         model.timestepper.Gⁿ[i],
                                         model.timestepper.G⁻[i],
                                         z_offset,
                                         dependencies=Event(device(model.architecture)))

        push!(events, field_event)
    end

    wait(device(model.architecture), MultiEvent(Tuple(events)))

    ### Gather results from each MPI rank to every rank
    # Create temporary array and kernels to write/read from it
    # I have done this because OffsetArrays did not mesh easily with MPI, so I copy the data to a
    # normal Array and back to facilitate the MPI gather. Once we figure out how to use OffsetArrays
    # with MPI, this will be simplified.
    tmp_field_for_MPI = Array{Float64}(undef, Nx, Ny, Nz)

    for (i, field) in enumerate(model_fields)
        # Copy field from OffsetArray to Array
        field_to_tmp!(tmp_field_for_MPI, field, z_offset, Nz_per_rank)

        # Send results from each rank to all the other ranks
        MPI.Allgather!(tmp_field_for_MPI, Nx*Ny*Nz_per_rank, comm)

        # Copy full field back from Array to OffsetArray
        tmp_to_field!(field, tmp_field_for_MPI)
    end

    return nothing
end

"""
Time step via

    `U^{n+1} = U^n + Δt ( (3/2 + χ) * G^{n} - (1/2 + χ) G^{n-1} )`

"""

# In this quick MPI implementation, each rank has the full array, but only does part of the work
# as determined by its z_offset
@kernel function ab2_step_field!(U, Δt, χ::FT, Gⁿ, G⁻, z_offset) where FT
    i, j, k = @index(Global, NTuple)
    k += z_offset

    @inbounds begin
        U[i, j, k] += Δt * (  (FT(1.5) + χ) * Gⁿ[i, j, k] - (FT(0.5) + χ) * G⁻[i, j, k] )

    end
end

# Copy field from OffsetArray to Array
# Only copy the line of real data (the work this MPI rank did) to the Array
function field_to_tmp!(tmpU, U, z_offset, Nz_per_rank)
    for k in (1 + z_offset):(Nz_per_rank + z_offset)
        for j in 1:size(tmpU,2)
            for i in 1:size(tmpU,1)
                tmpU[i, j, k] = U[i, j, k]
            end
        end
    end
end

# Copy full field back from Array to OffsetArray
function tmp_to_field!(U, tmpU)
    for k in 1:size(tmpU,3)
        for j in 1:size(tmpU,2)
            for i in 1:size(tmpU,1)
                U[i, j, k] = tmpU[i, j, k]
            end
        end
    end
end
